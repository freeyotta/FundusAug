import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from FundusHypothesisAug import FundusHypothesisAug

def run_bench(img_path='test_img.jpg'):
    aug_engine = FundusHypothesisAug(img_size=(512, 512))
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return

    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    test_modes = [
        ("Original", "Original Source"),
        ("Geometry", "Geo: Rotate/Shift"),
        ("Color", "Color: Saturation/Gamma"),
        ("Blur", "Noise: Gaussian Blur"), 
        ("LowRes", "Res: Downsample"),
        ("Cataract", "TMI: Physics Cataract") 
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, (mode, title) in enumerate(test_modes):
        if mode == "Original":
            axes[i].imshow(image_rgb)
        else:
            # 执行增强逻辑
            if mode == "Cataract":
                res = aug_engine.forward_augmentation(image_bgr, mode=mode)
                res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            else:
                res = aug_engine.forward_augmentation(image_rgb, mode=mode)
            
            axes[i].imshow(res)
        
        axes[i].set_title(title, fontsize=16, fontweight='bold')
        axes[i].axis('off')

    plt.tight_layout()
    # 建议保存一个清晰的文件名发给老师
    output_fn = "result.jpg"
    plt.savefig(output_fn, dpi=300)
    print(f"Success! 新的对比图已保存: {output_fn}")
    plt.show()

if __name__ == "__main__":
    run_bench('test_img.jpg')