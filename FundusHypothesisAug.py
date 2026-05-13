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
        
        # 1. 几何与结构变换 (Geometry & Structural) - 模拟拍摄角度、眼球位置
        self.geo_ops = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=1.0),
            A.Perspective(scale=(0.05, 0.1), p=1.0),
        ])

        # 2. 色彩与对比度变换 (Color & Contrast) - 模拟不同相机传感器差异
        self.color_ops = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0)
        ])

        # 3. 局部变换与噪声 (Local Artifacts & Noise) - 模拟传感器老化、对焦模糊
        self.artifact_ops = A.Compose([
            A.OneOf([
                A.GaussianBlur(blur_limit=(5, 11)),
                A.GaussNoise(var_limit=(20, 100)),
                A.Sharpen(alpha=(0.2, 0.5)),
            ], p=1.0),
            A.CoarseDropout(max_holes=8, max_height=40, max_width=40, fill_value=0, p=0.5)
        ])

        # 4. 压缩与重采样 (Resampling) - 模拟图像传输损失、低分辨率相机
        self.resample_ops = A.Compose([
            A.Downscale(scale_min=0.2, scale_max=0.4, interpolation=cv2.INTER_NEAREST, p=1.0),
            A.Resize(height=img_size[0], width=img_size[1], interpolation=cv2.INTER_CUBIC)
        ])

    # --- TMI 源码私有辅助函数 ---
    def _gaussian_kernel_official(self, img):
        """还原 TMI 源码中的 5x5 高斯卷积核逻辑"""
        kernel_5x5 = np.array([
            [1, 4, 7, 4, 1],
            [4, 16, 26, 16, 4],
            [7, 26, 41, 26, 7],
            [4, 16, 26, 16, 4],
            [1, 4, 7, 4, 1]
        ])
        kernel_5x5 = kernel_5x5 / kernel_5x5.sum()
        return ndimage.convolve(img, kernel_5x5)

    # --- 5. 病理级物理模拟 (Medical-specific: Cataract) ---
    def cataract_tmi_official(self, image):
        """100% 还原 TMI 论文源码物理算子"""
        # 预处理：resize 并提取 mask
        im_A = cv2.resize(image, self.img_size)
        gray_A = cv2.cvtColor(im_A, cv2.COLOR_BGR2GRAY)
        mask_A = ndimage.binary_opening(gray_A > 10, structure=np.ones((8, 8))) * 255
        mask_A_3 = (mask_A / (mask_A.max() + 1e-8))[:, :, np.newaxis]

        h, w = self.img_size
        # 生成随机距离场
        wp = random.randint(int(-w * 0.3), int(w * 0.3))
        hp = random.randint(int(-h * 0.3), int(h * 0.3))
        transmap = np.ones(shape=[h, w])
        transmap[w // 2 + wp, h // 2 + hp] = 0
        
        # 核心物理模拟：卷积平滑距离变换
        transmap = self._gaussian_kernel_official(ndimage.distance_transform_edt(transmap)) * mask_A
        transmap = transmap / (transmap.max() + 1e-8)

        # 模拟模糊细节丢失
        randomR = random.choice([1, 3, 5, 7])
        randomS = random.randint(10, 30)
        fundus_blur = cv2.GaussianBlur(im_A, (randomR, randomR), randomS)

        # TMI 特有的 Channel Inversion 散射逻辑
        B, G, R_ch = cv2.split(fundus_blur)
        panel = cv2.merge([
            transmap * (B.max() - B), 
            transmap * (G.max() - G), 
            transmap * (R_ch.max() - R_ch)
        ])
        
        # 合成
        panel_ratio = random.uniform(0.6, 0.8)
        sum_degrad = 0.8 * fundus_blur.astype(float) + panel * panel_ratio
        sum_degrad = np.clip(sum_degrad, 0, 255).astype('uint8')

        # PIL 色彩/对比度增强
        img_pil = Image.fromarray(sum_degrad)
        img_pil = ImageEnhance.Contrast(img_pil).enhance(random.uniform(0.9, 1.3))
        img_pil = ImageEnhance.Brightness(img_pil).enhance(random.uniform(0.9, 1.0))
        img_pil = ImageEnhance.Color(img_pil).enhance(random.uniform(0.9, 1.3))

        final_res = np.array(img_pil) * mask_A_3
        return final_res.astype('uint8')

    # --- 统一分发接口 (Forward Interface) ---
    def forward_augmentation(self, image, mode="Mixed"):
        """
        根据模式执行不同的假设空间变换
        """
        if mode == "Geometry":
            return self.geo_ops(image=image)['image']
        elif mode == "Color":
            return self.color_ops(image=image)['image']
        elif mode == "Blur":
            return self.artifact_ops(image=image)['image']
        elif mode == "LowRes":
            return self.resample_ops(image=image)['image']
        elif mode == "Cataract":
            return self.cataract_tmi_official(image)
        elif mode == "Mixed":
            # 几何 + 白内障模拟 的强一致性组合
            res = self.geo_ops(image=image)['image']
            res = self.cataract_tmi_official(res)
            return res
        return image