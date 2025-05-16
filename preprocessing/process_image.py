import os

import cv2


def preprocess(image_path, enhance=True):
    """
    对输入图像进行灰度转换、增强、模糊和边缘提取。

    Returns:
        dict: 包含原图、灰度图、模糊图、边缘图
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"找不到图像: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if enhance:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = cv2.Canny(blurred, 50, 150)

    return {
        'original': image,
        'gray': gray,
        'blurred': blurred,
        'edges': edges
    }

def save_intermediate_results(results_dict, save_dir):
    """
    将多个处理结果图像保存到指定目录
    """
    os.makedirs(save_dir, exist_ok=True)
    for key, img in results_dict.items():
        out_path = os.path.join(save_dir, f"{key}.png")
        cv2.imwrite(out_path, img)
