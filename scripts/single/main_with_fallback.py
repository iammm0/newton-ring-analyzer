import argparse

import torch
from PIL import Image

from analysis.circle_fitting import detect_newton_rings_opencv
from analysis.measure_parameters import analyze_newton_rings
from model.inference import load_model, predict_mask
from preprocessing.process_image import preprocess


def main_with_fallback(image_path, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device=device)

    # Step 1: 预处理图像
    results = preprocess(image_path)
    input_image = Image.fromarray(results["gray"]).convert("RGB")

    # Step 2: 使用模型预测
    print("🔍 使用模型预测掩膜...")
    mask = predict_mask(model, input_image, device=device)

    # Step 3: 判断掩膜是否有效
    if mask.sum() < 500:
        print("⚠️ 模型预测区域过小，尝试使用传统方法检测圆...")
        rings = detect_newton_rings_opencv(image_path)
        if rings:
            print(f"✅ 传统方法识别出 {len(rings)} 个圆：")
            for i, (x, y, r) in enumerate(rings):
                print(f"  - 圆{i+1}: 中心=({x}, {y}), 半径={r}")
        else:
            print("❌ 传统方法也未检测到有效圆")
    else:
        print("✅ 模型检测成功，使用分割掩膜分析直径...")
        diameters = analyze_newton_rings(mask)
        print(f"识别出 {len(diameters)} 个圆，直径列表（像素）: {diameters}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="图像路径")
    parser.add_argument("--model_path", default="fine_tuned_model.pth", help="模型文件路径")
    args = parser.parse_args()
    main_with_fallback(args.image_path, args.model_path)
