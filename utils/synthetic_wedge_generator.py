import cv2
import numpy as np
import os

def generate_wedge_image_pair(image_size=(256, 256), angle_deg=10, line_gap=5):
    h, w = image_size
    image = np.zeros((h, w), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    # 计算两个直线角度和位置
    angle_rad = np.deg2rad(angle_deg)
    center_x = w // 2
    center_y = h // 2

    length = max(h, w)

    dx = int(length * np.cos(angle_rad / 2))
    dy = int(length * np.sin(angle_rad / 2))

    # 第一条线（向左上）
    pt1_a = (center_x - dx, center_y + dy)
    pt2_a = (center_x + dx, center_y - dy)
    cv2.line(image, pt1_a, pt2_a, 255, 1)
    cv2.line(mask, pt1_a, pt2_a, 255, 1)

    # 第二条线（向右上）
    pt1_b = (center_x - dx, center_y - dy)
    pt2_b = (center_x + dx, center_y + dy)
    cv2.line(image, pt1_b, pt2_b, 255, 1)
    cv2.line(mask, pt1_b, pt2_b, 255, 1)

    # 添加轻微噪声
    noise = np.random.normal(0, 8, (h, w)).astype(np.uint8)
    noisy_image = cv2.add(image, noise)

    return noisy_image, mask

def generate_wedge_dataset(save_dir, count=100):
    os.makedirs(f"{save_dir}/images", exist_ok=True)
    os.makedirs(f"{save_dir}/masks", exist_ok=True)

    for i in range(count):
        angle = np.random.uniform(2, 20)  # 模拟不同劈尖夹角
        img, mask = generate_wedge_image_pair(angle_deg=angle)
        cv2.imwrite(f"{save_dir}/images/wedge_{i:04}.png", img)
        cv2.imwrite(f"{save_dir}/masks/wedge_{i:04}.png", mask)

    print(f"✅ 已生成 {count} 组劈尖图像，保存至 {save_dir}")
