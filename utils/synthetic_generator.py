import cv2
import numpy as np
import os

def generate_newton_ring_pair(image_size=(256, 256), num_rings=6, noise_level=10):
    h, w = image_size
    center = (w // 2, h // 2)

    image = np.zeros((h, w), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    base_radius = 20
    for i in range(num_rings):
        radius = base_radius + i * 10
        cv2.circle(image, center, radius, 255 - i * 15, 1)
        cv2.circle(mask, center, radius, 255, 1)

    # 添加轻微噪声模拟现实图像
    noise = np.random.normal(0, noise_level, (h, w)).astype(np.uint8)
    noisy_image = cv2.add(image, noise)

    return noisy_image, mask

def generate_dataset(save_dir, count=100):
    os.makedirs(f"{save_dir}/images", exist_ok=True)
    os.makedirs(f"{save_dir}/masks", exist_ok=True)

    for i in range(count):
        img, mask = generate_newton_ring_pair()
        cv2.imwrite(f"{save_dir}/images/{i:04}.png", img)
        cv2.imwrite(f"{save_dir}/masks/{i:04}.png", mask)

    print(f"✅ 成功生成 {count} 组合成图像和掩膜，保存在 {save_dir}/images 与 /masks")
