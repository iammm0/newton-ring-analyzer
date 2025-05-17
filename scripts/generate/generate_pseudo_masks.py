import os
import cv2

input_dir = "../../data/processed_images"
output_mask_dir = "../../real_data/masks"
output_img_dir = "../../real_data/images"

os.makedirs(output_mask_dir, exist_ok=True)
os.makedirs(output_img_dir, exist_ok=True)

subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
created = 0

for sub in subdirs:
    edge_path = os.path.join(input_dir, sub, "edges.png")
    original_path = os.path.join(input_dir, sub, "original.png")

    if os.path.exists(edge_path) and os.path.exists(original_path):
        # 读取边缘图
        edge_img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(edge_img, 30, 255, cv2.THRESH_BINARY)

        # 保存掩膜图
        mask_save_path = os.path.join(output_mask_dir, f"{sub}.png")
        cv2.imwrite(mask_save_path, mask)

        # 保存原图
        img = cv2.imread(original_path)
        img_save_path = os.path.join(output_img_dir, f"{sub}.png")
        cv2.imwrite(img_save_path, img)

        created += 1

print(f"✅ 共生成 {created} 个伪掩膜和对应原图")
