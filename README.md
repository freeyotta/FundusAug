# 眼底图像增强工具说明 (FundusHypothesisAug)

这是我为眼底图像研究写的一个数据增强工具包。

### 1. 怎么安装依赖？
你可以直接通过项目中的环境核对表进行一键安装。建议在虚拟环境中运行：

```bash
# 使用 pip 一键安装所有依赖库
pip install -r requirement.txt
```

如果手动安装，核心依赖包括：
`pip install opencv-python albumentations scipy matplotlib Pillow`

### 2. 这个包里有什么？

* **基础增强**：包括旋转、平移、亮度对比度调整、模糊、低分辨率模拟等常用操作。
* **物理模拟 (核心)**：这部分参考了 **TMI 论文的官方源码**，专门用来模拟**白内障**导致的视觉退化。它利用距离场和通道反转散射逻辑，还原了真实的物理退化过程。

### 3. 如何快速看效果？

1. 在项目文件夹里放一张叫 `test_img.jpg` 的原始眼底图。
2. 在终端运行测试脚本：`python AugTestBench.py`。
3. 运行完后，文件夹里会自动生成一张包含 6 种模式对比的图 `result.jpg`。

### 4. 怎么在代码里用？

你可以参考下面的例子快速集成到你的训练流程中：

```python
from FundusHypothesisAug import FundusHypothesisAug
import cv2

# 1. 初始化增强引擎
# 指定 img_size 确保输出尺寸一致，默认为 (512, 512)
aug = FundusHypothesisAug(img_size=(224, 224))

# 2. 读取图像 (OpenCV 读取默认为 BGR)
image = cv2.imread("test_img.jpg")

# 3. 调用增强接口
# mode 可以根据研究需求选择不同的假设空间变换
result = aug.forward_augmentation(image, mode="Cataract")

```

####  参数说明表

| 参数名 (Parameter) | 类型 | 默认值 | 说明 (Description) |
| --- | --- | --- | --- |
| **`img_size`** | `tuple` | `(512, 512)` | 初始化参数。定义输出图像的分辨率 (H, W)。 |
| **`image`** | `ndarray` | - | 输入图像。支持 OpenCV 格式 (H, W, C)，建议为 BGR 顺序。 |
| **`mode`** | `string` | `"Mixed"` | **增强模式选择**。决定图像进入哪种物理或几何假设空间。 |

####  可选模式 (`mode`) 详解

| 模式名 (Mode) | 增强逻辑与物理意义 |
| --- | --- |
| **`"Geometry"`** | 几何变换。包含随机旋转、平移、缩放及透视变换，模拟拍摄角度差异。 |
| **`"Color"`** | 色彩变换。模拟不同相机传感器的亮度、对比度、饱和度及 Gamma 差异。 |
| **`"Blur"`** | 模糊变换。模拟相机对焦不准或镜头污渍导致的模糊效果。 |
| **`"LowRes"`** | 低分辨率模拟。通过下采样再上采样，模拟低像素设备的成像质量。 |
| **`"Cataract"`** | **物理启发式白内障模拟 (核心)**。利用距离场模拟晶状体混浊，包含物理散射逻辑。 |
| **`"Mixed"`** | 混合模式。随机组合上述多种变换，产生最复杂的退化场景。 |

> **注意 (Tips)**:
> * `forward_augmentation` 返回的是 `numpy.ndarray` 格式。
> * 如果用于 **PyTorch** 训练，请确保在增强后使用 `transforms.ToTensor()` 将其转换为张量，并最后进行标准归一化。
> * **白内障模式**（Cataract）会对图像色彩产生较大物理扰动，建议仅在训练阶段开启。
> 
> 
