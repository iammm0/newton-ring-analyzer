import torch
import torchvision.transforms as T
from PIL import Image
from model.unet import UNet

# 图像转换（与训练时保持一致）
transform = T.Compose([
    T.Resize((256, 256)),  # 根据训练图像大小来定
    T.ToTensor(),
])


def load_model(checkpoint_path, device='cpu'):
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_mask(model, image, device='cpu'):
    # 如果传入的是路径，打开图像；否则假设已是 PIL.Image
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        mask = output.squeeze().cpu().numpy()
        mask = (mask > 0.5).astype("uint8") * 255

    return mask

