
import os
from glob import glob

import torch
from PIL import Image

from analysis.circle_fitting import detect_newton_rings_opencv
from analysis.measure_parameters import analyze_newton_rings
from model.inference import load_model, predict_mask
from preprocessing.process_image import preprocess


def analyze_all_images_with_fallback(folder, model_path="fine_tuned_model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device=device)

    image_paths = glob(os.path.join(folder, "*.jpg")) + glob(os.path.join(folder, "*.png"))
    for img_path in image_paths:
        print(f"\n🖼️ 正在分析图像: {os.path.basename(img_path)}")

        # Step 1: 预处理 + 模型输入
        results = preprocess(img_path)
        input_image = Image.fromarray(results["gray"]).convert("RGB")

        # Step 2: 模型预测掩膜
        mask = predict_mask(model, input_image, device=device)

        if mask.sum() < 500:
            print("⚠️ 模型掩膜无效，使用传统 OpenCV 圆检测...")
            rings = detect_newton_rings_opencv(img_path)
            if rings:
                print(f"✅ 传统方法检测到 {len(rings)} 个圆：")
                for i, (x, y, r) in enumerate(rings):
                    print(f"  - 圆{i+1}: 中心=({x}, {y}), 半径={r}")
            else:
                print("❌ 无法检测到有效圆")
        else:
            print("✅ 模型检测成功，分析分割掩膜...")
            diameters = analyze_newton_rings(mask)
            print(f"识别出 {len(diameters)} 个圆，直径列表（像素）: {diameters}")

if __name__ == "__main__":
    analyze_all_images_with_fallback("data/test_images", model_path="../../fine_tuned_model.pth")
