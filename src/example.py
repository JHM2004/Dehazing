import cv2
from dehazing import Dehazing

# 创建去雾器实例，使用非常温和的参数
dehazer = Dehazing(omega=0.75, t0=0.2, window_size=15)

# 读取有雾图像
hazy_img = cv2.imread('hazy_image.jpg')

# 执行去雾
result = dehazer.dehaze(hazy_img)

# 保存和显示结果
cv2.imwrite('dehazed_result.jpg', result)
cv2.imshow('Original', hazy_img)
cv2.imshow('Dehazed', result)
cv2.waitKey(0)
cv2.destroyAllWindows() 