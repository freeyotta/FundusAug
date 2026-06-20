import numpy as np
import cv2
import torch
import random
from scipy import ndimage
from PIL import Image, ImageEnhance
import albumentations as A

class FundusHypothesisAug:
    def __init__(self, img_size=(512, 512)):
        self.img_size = img_size
        
        # 1. 几何变换 (Geometry)
        self.geo = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=1.0),
            A.Perspective(scale=(0.05, 0.1), p=1.0),
        ])

        # 2. 色彩变换 (Color)
        self.color = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0)
        ])

        # 3. 图像模糊 (Blur)
        self.blur = A.Compose([
            A.GaussianBlur(blur_limit=(35, 55), p=1.0)
        ])

        # 4. 压缩与重采样 (Resampling)
        self.low_res = A.Compose([
            A.Downscale(scale_min=0.2, scale_max=0.4, interpolation=cv2.INTER_NEAREST, p=1.0),
            A.Resize(height=img_size[0], width=img_size[1], interpolation=cv2.INTER_CUBIC)
        ])

        # 5. 运动抖动 (Motion Blur)
        self.motion_blur = A.Compose([
            A.MotionBlur(blur_limit=(25, 45), allow_shifted=True, p=1.0)
        ])

    # ==========================================================
    # 核心物理/数学机制辅助函数：生成高级二维高斯矩阵Mask
    # ==========================================================
    def _generate_gaussian_mask(self, shape, center, sigma, max_val=1.0):
        """二维高斯矩阵用于光照模拟"""
        h, w = shape[:2]
        y, x = np.ogrid[:h, :w]
        cx, cy = center
        gauss = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
        return gauss * max_val

    def _simulate_overexposure_core(self, image, intensity=0.7):
        """内部高级过曝仿真：中心高亮，边缘衰减，特征截断丢失"""
        h, w, c = image.shape
        center = (np.random.randint(0, w), np.random.randint(0, h))
        sigma = np.random.randint(min(h, w) // 6, min(h, w) // 3)

        mask = self._generate_gaussian_mask((h, w), center, sigma, max_val=intensity * 255)
        mask = np.expand_dims(mask, axis=2)  # (H, W, 1)

        exposed_img = image.astype(np.float32) + mask
        return np.clip(exposed_img, 0, 255).astype('uint8')

    def _simulate_shadow_core(self, image, min_darkness=0.3):
        """内部高级阴影仿真：柔和半影，局部点乘变暗"""
        h, w, c = image.shape
        center = (np.random.randint(0, w), np.random.randint(0, h))
        sigma = np.random.randint(min(h, w) // 4, min(h, w) // 1.5)

        gauss = self._generate_gaussian_mask((h, w), center, sigma, max_val=1.0)
        shadow_mask = 1.0 - (1.0 - min_darkness) * gauss
        shadow_mask = np.expand_dims(shadow_mask, axis=2)  # (H, W, 1)

        shadow_img = image.astype(np.float32) * shadow_mask
        return np.clip(shadow_img, 0, 255).astype('uint8')

    # --- TMI 源码白内障物理模拟私有函数 ---
    def _gaussian_kernel_official(self, img):
        kernel_5x5 = np.array([
            [1, 4, 7, 4, 1],
            [4, 16, 26, 16, 4],
            [7, 26, 41, 26, 7],
            [4, 16, 26, 16, 4],
            [1, 4, 7, 4, 1]
        ])
        kernel_5x5 = kernel_5x5 / kernel_5x5.sum()
        return ndimage.convolve(img, kernel_5x5)

    def cataract_tmi_official(self, image):
        im_A = cv2.resize(image, self.img_size)
        gray_A = cv2.cvtColor(im_A, cv2.COLOR_BGR2GRAY)
        mask_A = ndimage.binary_opening(gray_A > 10, structure=np.ones((8, 8))) * 255
        mask_A_3 = (mask_A / (mask_A.max() + 1e-8))[:, :, np.newaxis]

        h, w = self.img_size
        wp = random.randint(int(-w * 0.3), int(w * 0.3))
        hp = random.randint(int(-h * 0.3), int(h * 0.3))
        transmap = np.ones(shape=[h, w])
        transmap[w // 2 + wp, h // 2 + hp] = 0
        
        transmap = self._gaussian_kernel_official(ndimage.distance_transform_edt(transmap)) * mask_A
        transmap = transmap / (transmap.max() + 1e-8)

        randomR = random.choice([1, 3, 5, 7])
        randomS = random.randint(10, 30)
        fundus_blur = cv2.GaussianBlur(im_A, (randomR, randomR), randomS)

        B, G, R_ch = cv2.split(fundus_blur)
        panel = cv2.merge([
            transmap * (B.max() - B), 
            transmap * (G.max() - G), 
            transmap * (R_ch.max() - R_ch)
        ])
        
        panel_ratio = random.uniform(0.6, 0.8)
        sum_degrad = 0.8 * fundus_blur.astype(float) + panel * panel_ratio
        sum_degrad = np.clip(sum_degrad, 0, 255).astype('uint8')

        img_pil = Image.fromarray(sum_degrad)
        img_pil = ImageEnhance.Contrast(img_pil).enhance(random.uniform(0.9, 1.3))
        img_pil = ImageEnhance.Brightness(img_pil).enhance(random.uniform(0.9, 1.0))
        img_pil = ImageEnhance.Color(img_pil).enhance(random.uniform(0.9, 1.3))

        final_res = np.array(img_pil) * mask_A_3
        return final_res.astype('uint8')

 # --- 统一分发接口 (Forward Interface) ---
    def forward_augmentation(self, image, mode="Mixed"):
        if mode == "Geometry":
            return self.geo(image=image)['image']
        elif mode == "Color":
            return self.color(image=image)['image']
        elif mode == "Blur":
            return self.blur(image=image)['image']
        elif mode == "LowRes":
            return self.low_res(image=image)['image']
        elif mode == "Cataract":
            return self.cataract_tmi_official(image)
        elif mode == "MotionBlur":   
            return self.motion_blur(image=image)['image']
        elif mode == "Overexposure":  
            return self._simulate_overexposure_core(image, intensity=0.7)
        elif mode == "Shadow":        
            return self._simulate_shadow_core(image, min_darkness=0.3)
            
        elif mode == "Mixed":
            # Mixed 逻辑：完全随机的临床复合退化流水线
            res = image.copy()
            
            # 1. 几何空间层（50% 概率触发：模拟患者配合度不佳、对焦歪斜）
            if random.random() < 0.5:
                res = self.geo(image=res)['image']
                
            # 2. 色彩波动层（40% 概率触发：模拟不同设备传感器的光照/伽马环境变动）
            if random.random() < 0.4:
                res = self.color(image=res)['image']
                
            # 3. 镜头伪影层（从算子里随机挑一个，50% 概率触发）
            if random.random() < 0.5:
                artifact_mode = random.choice(["Blur", "MotionBlur", "LowRes"])
                if artifact_mode == "Blur":
                    res = self.blur(image=res)['image']
                elif artifact_mode == "MotionBlur":
                    res = self.motion_blur(image=res)['image']
                elif artifact_mode == "LowRes":
                    res = self.low_res(image=res)['image']
                    
            # 4. 高级局部光照层（从高斯过曝和阴影里随机挑一个，40% 概率触发）
            if random.random() < 0.4:
                light_mode = random.choice(["Overexposure", "Shadow"])
                if light_mode == "Overexposure":
                    res = self._simulate_overexposure_core(res, intensity=random.uniform(0.5, 0.8))
                elif light_mode == "Shadow":
                    res = self._simulate_shadow_core(res, min_darkness=random.uniform(0.2, 0.5))     
            return res       
        return image
