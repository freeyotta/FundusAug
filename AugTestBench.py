import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from FundusHypothesisAug import FundusHypothesisAug

def run_bench(img_path='test_img.jpg'):
    # 初始化增强引擎
    aug_engine = FundusHypothesisAug(img_size=(512, 512))
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found. 请在当前目录下放置一张 {img_path}")
        return

    # 1. 读取原始图像
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 2. 适配 9 种模式（1种原始 + 8种增强）
    test_modes = [
        ("Original", "Original Source"),
        ("Geometry", "Geo: Rotate/Shift"),
        ("Color", "Color: Saturation/Gamma"),
        ("Blur", "Blur: Gaussian Blur"),
        ("LowRes", "Res: Downsample"),
        ("Cataract", "TMI: Physics Cataract"),
        ("MotionBlur", "Motion: Motion Blur"),
        ("Overexposure", "Light: Overexposure"),
        ("Shadow", "Shadow: Local Shadow")
    ]

    # 3. 3x3 布局
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()

    for i, (mode, title) in enumerate(test_modes):
        if mode == "Original":
            axes[i].imshow(image_rgb)
        else:
            # 兼容性处理：只有白内障物理算子必须接收 BGR，其他直接喂 RGB
            if mode == "Cataract":
                res = aug_engine.forward_augmentation(image_bgr, mode=mode)
                res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            else:
                res = aug_engine.forward_augmentation(image_rgb, mode=mode)
            
            axes[i].imshow(res)
        
        axes[i].set_title(title, fontsize=16, fontweight='bold')
        axes[i].axis('off')

    plt.tight_layout()
    output_fn = "result.jpg"
    plt.savefig(output_fn, dpi=300)
    print(f"Success! 包含 9 种模式的最新对比图已保存至: {output_fn}")
    plt.show()

if __name__ == "__main__":
    run_bench('test_img.jpg')
