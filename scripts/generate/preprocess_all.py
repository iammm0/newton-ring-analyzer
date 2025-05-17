import os
from glob import glob
from preprocessing.process_image import preprocess, save_intermediate_results

def preprocess_all_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_paths = glob(os.path.join(input_folder, "*.jpg")) + glob(os.path.join(input_folder, "*.png"))

    for img_path in image_paths:
        print(f"🔍 正在处理: {os.path.basename(img_path)}")

        results = preprocess(img_path)

        # 保存所有中间结果
        name = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(output_folder, name)
        save_intermediate_results(results, save_path)

    print(f"\n✅ 所有预处理结果已保存至: {output_folder}")

if __name__ == "__main__":
    preprocess_all_images("data/test_images", "data/processed_images")