import torch
import torch.nn as nn
from torchvision.transforms import v2
from torchvision.transforms import functional as VF


class ResizeAndPadToSquare(nn.Module):
    def __init__(self, long_size=640, fill=0):
        super().__init__()
        self.long_size = long_size
        self.fill = fill

    def forward(self, img):
        # img: PIL or Tensor, shape [C,H,W] after ToImage
        if isinstance(img, torch.Tensor):
            _, h, w = img.shape
        else:
            w, h = img.size

        scale = self.long_size / max(h, w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))

        img = VF.resize(img, size=[new_h, new_w])

        pad_h = self.long_size - new_h
        pad_w = self.long_size - new_w

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        img = VF.pad(img, [left, top, right, bottom], fill=self.fill)
        return img


class CLAHETransform(nn.Module):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def forward(self, x):
        import cv2
        import numpy as np

        if x.dim() == 3:
            x = x.unsqueeze(0)

        device = x.device
        dtype = x.dtype
        outs = []

        for img in x:
            img_np = img.detach().cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

            img_np = img_np.astype("float32") / 255.0
            img_t = torch.from_numpy(img_np).permute(2, 0, 1)
            outs.append(img_t)

        outs = torch.stack(outs, dim=0).to(device=device, dtype=dtype)
        return outs.squeeze(0) if outs.shape[0] == 1 else outs


class EdgeEnhancement(nn.Module):
    def __init__(self, intensity=0.3):
        super().__init__()
        self.intensity = intensity

        sobel_x = torch.tensor(
            [[[[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]]],
            dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[[[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]]]],
            dtype=torch.float32
        )

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, x):
        import torch.nn.functional as F

        if x.dim() == 3:
            x = x.unsqueeze(0)

        gray = x.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        edge = edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-6)

        out = x + self.intensity * edge
        out = out.clamp(0.0, 1.0)

        return out.squeeze(0) if out.shape[0] == 1 else out


def build_base_transform(image_size=640):
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        ResizeAndPadToSquare(long_size=image_size),
    ])


def build_geometric_transform(image_size=640):
    # 对 image+mask 同步做
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.85, 1.0),
            ratio=(0.90, 1.10),
        ),
        v2.RandomRotation(degrees=10),
    ])


def build_image_only_transform():
    return v2.Compose([
        # v2.RandomAutocontrast(p=0.2), 
        v2.RandomApply([CLAHETransform()], p=0.30),
        v2.RandomApply([EdgeEnhancement(intensity=0.20)], p=0.30),

        v2.RandomApply([
            v2.ColorJitter(
                brightness=0.12,
                contrast=0.12,
                saturation=0.05,
                hue=0.01
            )
        ], p=0.4),

        # 先去掉 grayscale
        # v2.RandomGrayscale(p=0.05),

        # blur 改成随机、低强度
        v2.RandomApply([
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6))
        ], p=0.2),

        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])