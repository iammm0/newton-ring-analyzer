import argparse

from analysis.measure_parameters import analyze_newton_rings, analyze_wedge_angle
from model.inference import load_model, predict_mask
from preprocessing.process_image import preprocess


def main(image_path, mode, use_model=False, model_path=None):
    # 预处理
    processed = preprocess(image_path)
    edge_image = processed['edges']

    # 是否使用模型分割
    if use_model:
        model = load_model(model_path)
        mask = predict_mask(model, image_path)
    else:
        mask = edge_image  # 直接使用边缘图

    # 参数分析
    if mode == 'newton':
        diameters = analyze_newton_rings(mask)
        print(f"检测到 {len(diameters)} 个圆环")
        print("直径列表（像素）：", diameters)
    elif mode == 'wedge':
        angles = analyze_wedge_angle(mask)
        print(f"检测到 {len(angles)} 条线段")
        print("角度列表（度）：", angles)
    else:
        print("错误：不支持的模式。请使用 'newton' 或 'wedge'。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="输入图像路径")
    parser.add_argument("--mode", type=str, choices=["newton", "wedge"], default="newton", help="分析类型")
    parser.add_argument("--use_model", action="store_true", help="是否使用分割模型")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径")

    args = parser.parse_args()
    main(args.image_path, args.mode, args.use_model, args.model_path)
