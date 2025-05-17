from scripts.training.train_model import train_model

if __name__ == "__main__":
    train_model(
        image_dir="../../real_data/images",
        mask_dir="../../real_data/masks",
        save_path="../../fine_tuned_model.pth",
        lr=1e-5,
        epochs=10,
        batch_size=2
    )
