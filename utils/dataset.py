from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as T

class NewtonRingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 灰度掩膜

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)  # 确保为 [1, H, W]

        return image, mask
