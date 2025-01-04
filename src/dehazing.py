import cv2
import numpy as np

class Dehazing:
    def __init__(self, omega=0.75, t0=0.2, window_size=15, sky_thresh=0.7, sky_trans=0.85):
        """
        增加天空处理相关参数
        """
        self.omega = omega          # 去雾强度
        self.t0 = t0               # 最小透射率
        self.window_size = window_size
        self.sky_thresh = sky_thresh  # 天空识别阈值
        self.sky_trans = sky_trans    # 天空透射率
        
    def get_dark_channel(self, img):
        """优化暗通道计算"""
        b, g, r = cv2.split(img)
        min_rgb = cv2.min(cv2.min(r, g), b)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                         (self.window_size, self.window_size))
        dark_channel = cv2.erode(min_rgb, kernel)
        return dark_channel

    def estimate_atmospheric_light(self, img, dark_channel):
        """优化大气光值估算"""
        h, w = dark_channel.shape
        flat_img = img.reshape(h * w, 3)
        flat_dark = dark_channel.ravel()
        
        # 取暗通道前0.1%最亮的点
        max_pixels = int(h * w * 0.001)
        indices = np.argpartition(flat_dark, -max_pixels)[-max_pixels:]
        
        # 在最亮的点中选择RGB均值最大的点
        candidate_pixels = flat_img.take(indices, axis=0)
        bright_values = np.mean(candidate_pixels, axis=1)
        atmospheric_light = candidate_pixels[np.argmax(bright_values)]
        
        return atmospheric_light

    def estimate_transmission(self, img, atmospheric_light):
        """更新天空区域处理"""
        normalized = img / atmospheric_light
        dark_channel = self.get_dark_channel(normalized)
        
        # 计算初始透射率
        transmission = 1 - self.omega * dark_channel
        
        # 计算亮度和颜色特征
        intensity = np.mean(img, axis=2)
        blue_channel = img[:,:,0]
        
        # 使用可调节的阈值识别天空区域，增加蓝色通道的权重
        sky_mask = (intensity > self.sky_thresh) & (blue_channel > np.mean(img, axis=2) * 0.9)
        
        # 使用可调节的透射率处理天空区域
        transmission[sky_mask] = np.maximum(transmission[sky_mask], self.sky_trans)
        
        # 非天空区域温和去雾
        non_sky_mask = ~sky_mask
        transmission[non_sky_mask] = transmission[non_sky_mask] * 0.9
        
        return transmission

    def guided_filter(self, img, p, r, eps):
        """引导滤波优化"""
        mean_i = cv2.boxFilter(img, cv2.CV_64F, (r,r))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (r,r))
        mean_ip = cv2.boxFilter(img*p, cv2.CV_64F, (r,r))
        cov_ip = mean_ip - mean_i * mean_p

        mean_ii = cv2.boxFilter(img*img, cv2.CV_64F, (r,r))
        var_i = mean_ii - mean_i * mean_i

        a = cov_ip / (var_i + eps)
        b = mean_p - a * mean_i

        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r,r))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r,r))

        return mean_a * img + mean_b

    def dehaze(self, img):
        """使用可调节的天空参数进行去雾处理"""
        norm_img = img.astype('float64') / 255
        
        dark_channel = self.get_dark_channel(norm_img)
        atmospheric_light = self.estimate_atmospheric_light(norm_img, dark_channel)
        
        transmission = self.estimate_transmission(norm_img, atmospheric_light)
        
        # 优化透射率图
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float64') / 255.0
        transmission = self.guided_filter(gray_img, transmission, 60, 0.0001)
        
        # 使用可调节的天空参数
        intensity = np.mean(norm_img, axis=2)
        sky_mask = (intensity > self.sky_thresh) & (norm_img[:,:,0] > np.mean(norm_img, axis=2) * 0.9)
        
        # 使用可调节的透射率范围，扩大范围
        transmission[sky_mask] = np.clip(transmission[sky_mask], self.sky_trans * 0.8, min(self.sky_trans * 1.2, 1.0))
        transmission[~sky_mask] = np.clip(transmission[~sky_mask], self.t0, 0.9)
        
        # 恢复无雾图像
        result = np.empty_like(norm_img)
        for i in range(3):
            result[:, :, i] = (norm_img[:, :, i] - atmospheric_light[i]) / \
                             transmission + atmospheric_light[i]
        
        # 对比度增强，使用更灵活的天空区域调整
        result[~sky_mask] = result[~sky_mask] * 1.1
        sky_enhance = 2.0 - self.sky_trans if self.sky_trans <= 1.0 else self.sky_trans
        result[sky_mask] = result[sky_mask] * sky_enhance
        
        # 颜色校正
        mean_intensity = np.mean(result, axis=2, keepdims=True)
        result = result + (result - mean_intensity) * 0.1
        
        result = np.clip(result * 255, 0, 255).astype('uint8')
        return result 