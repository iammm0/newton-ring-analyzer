import os
from glob import glob

from sympy.printing.pytorch import torch

from preprocessing.process_image import preprocess
from model.inference import load_model, predict_mask
from analysis.measure_parameters import analyze_newton_rings, analyze_wedge_angle

def analyze_all_images(
    folder, mode="newton", model_path="saved_model.pth", device="cuda" if torch.cuda.is_available() else "cpu"
):
    model = load_model(model_path, device=device)
    image_paths = glob(os.path.join(folder, "*.jpg")) + glob(os.path.join(folder, "*.png"))

    for img_path in image_paths:
        print(f"\n🖼️ 分析图像: {os.path.basename(img_path)}")

        # Step 1: 预处理
        results = preprocess(img_path)
        processed_image = results["edges"]  # or "gray", based on what works best

        # Step 2: 模型预测
        mask = predict_mask(model, img_path, device=device)

        # Step 3: 参数提取
        if mode == "newton":
            diameters = analyze_newton_rings(mask)
            print(f"识别出 {len(diameters)} 个圆，直径（像素）: {diameters}")
        elif mode == "wedge":
            angles = analyze_wedge_angle(mask)
            print(f"识别出 {len(angles)} 条线，角度（度）: {angles}")
        else:
            print("❌ 未知分析模式。")

if __name__ == "__main__":
    import torch
    analyze_all_images(folder="data/test_images", mode="newton", model_path="../../saved_model.pth")
