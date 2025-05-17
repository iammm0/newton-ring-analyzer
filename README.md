
# 🧠 牛顿环识别项目说明文档

本项目基于 PyTorch + OpenCV 实现牛顿环图像中圆环结构的自动识别与参数提取，支持单图、多图、深度模型和传统图像处理方法组合使用。

---

## 🎯 项目目标

- 自动检测图像中的牛顿环圆结构
- 提取有效参数（直径、圆心位置、数量）
- 支持完整/部分/偏移牛顿环图像
- 提供传统视觉 & 深度学习双路线融合方案

---



## 🛠 技术路径概览

| 组件         | 技术                  |
|--------------|-----------------------|
| 图像预处理    | OpenCV（模糊、边缘）  |
| 掩膜生成      | 自动生成伪掩膜 / 手工 |
| 图像分割模型  | PyTorch U-Net         |
| 圆检测辅助    | Hough变换 + 圆心拟合  |
| 参数提取      | `cv2.minEnclosingCircle` + 拟合优化 |



## 📁 项目结构调整建议

```
project_root/
├── scripts/                    # 执行脚本总目录
│   ├── single/                 # 针对单张图像执行的脚本
│   │   ├── main.py             # 使用模型检测
│   │   ├── main_with_fallback.py  # 模型+OpenCV融合检测
│   ├── batch/                  # 批量处理整个文件夹
│   │   ├── analyze_all.py
│   │   ├── analyze_all_with_fallback.py
│   │   ├── analyze_processed_images.py
│   ├── traditional/            # 仅使用传统方法（无模型）
│   │   ├── detect_by_opencv.py
│   ├── training/               # 训练模型脚本
│   │   ├── train_model.py
│   │   ├── fine_tune_model.py
│   ├── generate/               # 数据生成脚本
│   │   ├── generate_data.py
│   │   ├── generate_pseudo_masks.py
│   │   ├── preprocess_all.py
├── analysis/                   # 参数提取逻辑
├── model/                      # 模型结构和加载
├── preprocessing/             # 图像预处理逻辑
├── utils/                      # 数据集加载和合成工具
├── data/                       # 图像数据和掩膜
│   ├── test_images/
│   ├── processed_images/
│   ├── real_data/
├── saved_model.pth
├── fine_tuned_model.pth
├── README.md
```

---



## 🧪 执行命令索引

### ✅ 使用模型检测单张图片

```bash
python scripts/single/main.py data/test_images/test1.jpg --mode newton --use_model --model_path fine_tuned_model.pth
```

### ✅ 使用模型 + OpenCV 兜底检测单张图片

```bash
python scripts/single/main_with_fallback.py data/test_images/test1.jpg --model_path fine_tuned_model.pth
```

---

### 📂 批量检测整个文件夹（processed 或原图）

```bash
python scripts/batch/analyze_all.py
python scripts/batch/analyze_processed_images.py
python scripts/batch/analyze_all_with_fallback.py
```

---

### 🧠 使用 OpenCV 圆检测法（不依赖模型）

```bash
python scripts/traditional/detect_by_opencv.py data/test_images/test1.jpg
```

---

### 🧠 训练/微调模型

```bash
python scripts/training/train_model.py
python scripts/training/fine_tune_model.py
```

---

### 🛠 数据生成与预处理

```bash
python scripts/generate/generate_data.py
python scripts/generate/preprocess_all.py
python scripts/generate/generate_pseudo_masks.py
```

---



## 🧪 训练流程相关命令索引

### 📸 1. 生成合成训练数据（牛顿环图像 + 掩膜）

```bash
python scripts/generate/generate_data.py
```

输出到：

```bash
data/images/
data/masks/
```

------

### 🧼 2. 预处理图像（灰度/模糊/边缘）

```bash
python scripts/generate/preprocess_all.py
```

输出到：

```bash
data/processed_images/
```

------

### 🧩 3. 自动生成伪掩膜用于真实图微调训练（从 edges.png → mask）

```
python scripts/generate/generate_pseudo_masks.py
```

输出到：

```bash
real_data/images/
real_data/masks/
```

------

## 🧠 模型训练命令

### 📌 4. 初次训练（使用合成图像）

```bash
python scripts/training/train_model.py
```

输出模型文件：

```bash
saved_model.pth
```

------

### 🔁 5. 微调训练（使用真实图伪掩膜）

```bash
python scripts/training/fine_tune_model.py
```

输出模型文件：

```bash
fine_tuned_model.pth
```

## ✅ 总结

根据任务需求灵活选择：
- 单图检测 vs 批量检测
- 模型预测 vs OpenCV圆检测 vs 融合检测
- 数据增强、掩膜伪生成、微调训练等脚本组件组合

