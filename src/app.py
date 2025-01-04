from flask import Flask, render_template, request, send_file
import os
from dehazing import Dehazing
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp','webp','tiff','gif','svg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['image']
    if file.filename == '':
        return 'No file selected', 400
    
    # 获取参数
    omega = float(request.form.get('omega', 0.75))
    window_size = int(request.form.get('windowSize', 15))
    sky_thresh = float(request.form.get('skyThresh', 0.7))
    sky_trans = float(request.form.get('skyTrans', 0.85))
    
    if file and allowed_file(file.filename):
        # 读取上传的图像
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 使用用户设置的参数执行去雾处理
        dehazer = Dehazing(
            omega=omega, 
            t0=0.2, 
            window_size=window_size,
            sky_thresh=sky_thresh,
            sky_trans=sky_trans
        )
        result = dehazer.dehaze(img)
        
        # 将处理后的图像转换为字节流
        is_success, buffer = cv2.imencode('.jpg', result)
        if not is_success:
            return 'Error processing image', 500
        
        io_buf = io.BytesIO(buffer)
        io_buf.seek(0)
        
        return send_file(
            io_buf,
            mimetype='image/jpeg',
            as_attachment=False,
            download_name='dehazed.jpg'
        )
    
    return 'Invalid file type', 400

if __name__ == '__main__':
    app.run(debug=True) 