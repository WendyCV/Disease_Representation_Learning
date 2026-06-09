#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from torchvision.transforms.functional import to_pil_image

from augmentations import build_base_transform, build_geometric_transform, build_image_only_transform

import warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning)


def center_pad_to_square(
    image: Image.Image,
    fill: int | Tuple[int, int, int] = 0,
) -> Image.Image:
    width, height = image.size
    target_size = max(width, height)

    pad_w = target_size - width
    pad_h = target_size - height

    if pad_w == 0 and pad_h == 0:
        return image

    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    return ImageOps.expand(
        image,
        border=(left, top, right, bottom),
        fill=fill,
    )


def build_all_one_mask(image: Image.Image) -> Image.Image:
    width, height = image.size
    mask_np = np.ones((height, width), dtype=np.uint8) * 255
    return Image.fromarray(mask_np, mode="L")


def postprocess_mask(
    mask: Image.Image,
    threshold: int = 127,
    median_blur: int = 0,
    open_kernel: int = 0,
    close_kernel: int = 0,
) -> Image.Image:
    mask_np = np.array(mask.convert("L"), dtype=np.uint8)
    mask_np = (mask_np > threshold).astype(np.uint8) * 255

    if median_blur and median_blur >= 3:
        k = int(median_blur)
        if k % 2 == 0:
            k += 1
        mask_np = cv2.medianBlur(mask_np, k)

    if open_kernel and open_kernel >= 3:
        k = int(open_kernel)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)

    if close_kernel and close_kernel >= 3:
        k = int(close_kernel)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)

    mask_np = (mask_np > 127).astype(np.uint8) * 255
    return Image.fromarray(mask_np, mode="L")


def build_edge_suppression_mask(
    image: Image.Image,
    edge: int = 64,
    threshold: int = 30,
) -> np.ndarray:
    image_np = np.array(image.convert("RGB"))
    height, width, _ = image_np.shape
    edge = int(min(height / 10, width / 10, edge))

    valid_mask = np.ones((height, width), dtype=np.uint8)
    dark_pixels = np.all(image_np < threshold, axis=2)

    valid_mask[:edge, :] &= ~dark_pixels[:edge, :]
    valid_mask[-edge:, :] &= ~dark_pixels[-edge:, :]
    valid_mask[:, :edge] &= ~dark_pixels[:, :edge]
    valid_mask[:, -edge:] &= ~dark_pixels[:, -edge:]

    return valid_mask


def load_image_and_mask(
    image_path: str,
    mask_path: str | None,
    image_size: int,
    mask_mode: str = "sam2",
    missing_mask_policy: str = "ones",
    mask_threshold: int = 127,
    median_blur: int = 0,
    open_kernel: int = 0,
    close_kernel: int = 0,
    edge_mask: bool = False,
    min_valid_pixels: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor, Image.Image, Image.Image]:
    image = Image.open(image_path).convert("RGB")

    if mask_mode == "none":
        mask = build_all_one_mask(image)

    elif mask_mode == "sam2":
        if mask_path is None or not os.path.exists(mask_path):
            if missing_mask_policy == "ones":
                print(f"[WARN] Missing mask, use all-one mask: {mask_path}")
                mask = build_all_one_mask(image)
            else:
                raise FileNotFoundError(f"Missing mask: {mask_path}")
        else:
            mask = Image.open(mask_path).convert("L")
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.NEAREST)

            mask = postprocess_mask(
                mask,
                threshold=mask_threshold,
                median_blur=median_blur,
                open_kernel=open_kernel,
                close_kernel=close_kernel,
            )
    else:
        raise ValueError(f"Unsupported mask_mode: {mask_mode}")

    image = center_pad_to_square(image, fill=(0, 0, 0))
    mask = center_pad_to_square(mask, fill=0)

    if edge_mask:
        edge_valid = build_edge_suppression_mask(image).astype(np.uint8)
        mask_np = np.array(mask, dtype=np.uint8)
        mask_np = (mask_np // 255) * edge_valid * 255
        mask = Image.fromarray(mask_np.astype(np.uint8), mode="L")

    rgba_image = image.copy()
    rgba_image.putalpha(mask)

    base_transform = build_base_transform(image_size=image_size)
    rgba_tensor = base_transform(rgba_image)

    image_tensor = rgba_tensor[:3]
    mask_tensor = rgba_tensor[3:].float()
    mask_tensor = (mask_tensor > 0.5).float()

    if mask_tensor.sum().item() < min_valid_pixels:
        print("[WARN] Mask valid pixels too few, use all-one mask.")
        mask_tensor = torch.ones_like(mask_tensor)

    base_image_pil = tensor_to_pil_image(image_tensor)
    base_mask_pil = tensor_to_pil_mask(mask_tensor)

    return image_tensor, mask_tensor, base_image_pil, base_mask_pil


def build_augmented_view(
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    image_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    geometry_transform = build_geometric_transform(image_size=image_size)
    image_only_transform = build_image_only_transform()

    rgba_tensor = torch.cat([image_tensor, mask_tensor], dim=0)
    rgba_tensor = geometry_transform(rgba_tensor)

    view_image = rgba_tensor[:3]
    view_mask = (rgba_tensor[3:] > 0.5).float()

    view_image = image_only_transform(view_image)

    return view_image, view_mask


def tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0, 1)
    return to_pil_image(tensor)


def tensor_to_pil_mask(mask_tensor: torch.Tensor) -> Image.Image:
    mask = mask_tensor.detach().cpu()
    if mask.dim() == 3:
        mask = mask[0]
    mask_np = (mask.numpy() > 0.5).astype(np.uint8) * 255
    return Image.fromarray(mask_np, mode="L")


def build_overlay(
    image: Image.Image,
    mask: Image.Image,
    alpha: float = 0.45,
) -> Image.Image:
    image = image.convert("RGB")
    mask_np = np.array(mask.convert("L")) > 127

    img_np = np.array(image).astype(np.float32)
    red = np.array([255, 0, 0], dtype=np.float32)

    img_np[mask_np] = (1 - alpha) * img_np[mask_np] + alpha * red
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    return Image.fromarray(img_np, mode="RGB")


import math
from typing import Tuple
from PIL import Image, ImageDraw

def make_contact_sheet(
    items: list[Tuple[str, Image.Image]],
    cell_size: int = 220,
    cols: int = 6
) -> Image.Image:
    title_h = 28

    rows = math.ceil(len(items) / cols)

    sheet = Image.new(
        "RGB",
        (cols * cell_size, rows * (cell_size + title_h)),
        (255, 255, 255)
    )

    draw = ImageDraw.Draw(sheet)

    for i, (title, img) in enumerate(items):
        row = i // cols
        col = i % cols

        x = col * cell_size
        y = row * (cell_size + title_h)

        draw.text((x + 5, y + 6), title, fill=(0, 0, 0))

        img = img.convert("RGB").resize((cell_size, cell_size), Image.BILINEAR)
        sheet.paste(img, (x, y + title_h))

    return sheet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="./runs_audit/debug_augmentation")
    parser.add_argument("--prefix", type=str, default="sample009")

    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--mask-mode", type=str, default="sam2", choices=["sam2", "none"])
    parser.add_argument("--missing-mask-policy", type=str, default="ones", choices=["ones", "error"])

    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--median-blur", type=int, default=0)
    parser.add_argument("--open-kernel", type=int, default=0)
    parser.add_argument("--close-kernel", type=int, default=0)
    parser.add_argument("--edge-mask", action="store_true")
    parser.add_argument("--min-valid-pixels", type=int, default=4096)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    image_tensor, mask_tensor, base_image, base_mask = load_image_and_mask(
        image_path=args.image,
        mask_path=args.mask,
        image_size=args.image_size,
        mask_mode=args.mask_mode,
        missing_mask_policy=args.missing_mask_policy,
        mask_threshold=args.mask_threshold,
        median_blur=args.median_blur,
        open_kernel=args.open_kernel,
        close_kernel=args.close_kernel,
        edge_mask=args.edge_mask,
        min_valid_pixels=args.min_valid_pixels,
    )

    base_image.save(out_dir / f"{args.prefix}_base_image.png")
    base_mask.save(out_dir / f"{args.prefix}_base_mask.png")
    build_overlay(base_image, base_mask).save(out_dir / f"{args.prefix}_base_overlay.png")

    sheet_items: list[Tuple[str, Image.Image]] = [
        ("base_image", base_image),
        ("base_mask", base_mask.convert("RGB")),
        ("base_overlay", build_overlay(base_image, base_mask)),
    ]

    for i in range(1, args.num_views + 1):
        view_image_tensor, view_mask_tensor = build_augmented_view(
            image_tensor=image_tensor.clone(),
            mask_tensor=mask_tensor.clone(),
            image_size=args.image_size,
        )

        view_image = tensor_to_pil_image(view_image_tensor)
        view_mask = tensor_to_pil_mask(view_mask_tensor)
        view_overlay = build_overlay(view_image, view_mask)

        view_image.save(out_dir / f"{args.prefix}_view{i}_image.png")
        view_mask.save(out_dir / f"{args.prefix}_view{i}_mask.png")
        view_overlay.save(out_dir / f"{args.prefix}_view{i}_overlay.png")

        sheet_items.extend([
            (f"view{i}_image", view_image),
            (f"view{i}_mask", view_mask.convert("RGB")),
            (f"view{i}_overlay", view_overlay),
        ])

    contact_sheet = make_contact_sheet(sheet_items, cell_size=640)
    contact_sheet.save(out_dir / f"{args.prefix}_contact_sheet.png")

    print(f"[INFO] Done. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()