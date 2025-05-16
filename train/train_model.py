import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from model.unet import UNet
from utils.dataset import NewtonRingDataset


def train_model(
    image_dir, mask_dir, save_path,
    lr=1e-4, batch_size=4, epochs=20, device='cuda' if torch.cuda.is_available() else 'cpu'
):
    # 数据增强和预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = NewtonRingDataset(image_dir, mask_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)

            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), save_path)
    print(f"模型保存至: {save_path}")

if __name__ == "__main__":
    train_model(
        image_dir="data/images",
        mask_dir="data/masks",
        save_path="saved_model.pth",
        epochs=20
    )
