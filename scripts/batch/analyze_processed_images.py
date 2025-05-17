import os

import cv2
import torch
from PIL import Image

from analysis.measure_parameters import analyze_newton_rings, analyze_wedge_angle
from model.inference import load_model, predict_mask


def analyze_processed_images(folder, mode="newton", model_path="saved_model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device=device)

    subdirs = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

    for sub in subdirs:
        edge_path = os.path.join(folder, sub, "gray.png")
        if not os.path.exists(edge_path):
            print(f"⚠️ 跳过 {sub}，未找到 edges.png")
            continue

        print(f"\n🖼️ 分析图像: {sub}")

        # 读取 edges 图像并转换为 RGB 送入模型
        edge_img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        edge_img_rgb = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2RGB)
        pil_image = Image.fromarray(edge_img_rgb)

        # 模型预测
        mask = predict_mask(model, pil_image, device=device)

        # 参数分析
        if mode == "newton":
            diameters = analyze_newton_rings(mask)
            print(f"直径列表（像素）: {diameters}")
        elif mode == "wedge":
            angles = analyze_wedge_angle(mask)
            print(f"角度列表（度）: {angles}")
        else:
            print("❌ 未知分析模式")

if __name__ == "__main__":
    analyze_processed_images("data/processed_images", mode="newton", model_path="../../saved_model.pth")
