# -*- coding: utf-8 -*-
"""
Filter SSL images by foreground mask ratio after square padding.

Function:
1. Read mask files.
2. Pad mask to square using the same logic as image square padding.
3. Compute foreground ratio after padding:
       ratio = foreground_pixels / square_pixels
4. Save statistics CSV.
5. Optionally delete or move image/mask pairs whose ratio is too small or too large.

Recommended usage:

# 1) Dry-run only, do not delete anything
python tools/filter_ssl_by_mask_ratio.py ^
  --image_root data/unlabeled_train ^
  --mask_root data/foreground_masks ^
  --out_csv runs/mask_ratio_stats.csv ^
  --min_ratio 0.03 ^
  --max_ratio 0.60

# 2) Move bad samples to quarantine folder
python tools/filter_ssl_by_mask_ratio.py ^
  --image_root data/unlabeled_train ^
  --mask_root data/foreground_masks ^
  --out_csv runs/mask_ratio_stats.csv ^
  --min_ratio 0.03 ^
  --max_ratio 0.60 ^
  --move_to runs/removed_by_mask_ratio

# 3) Permanently delete bad samples, use carefully
python tools/filter_ssl_by_mask_ratio.py ^
  --image_root data/unlabeled_train ^
  --mask_root data/foreground_masks ^
  --out_csv runs/mask_ratio_stats.csv ^
  --min_ratio 0.03 ^
  --max_ratio 0.60 ^
  --delete
"""

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
MASK_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def center_pad_to_square_pil(img: Image.Image, fill=0) -> Image.Image:
    """
    Pad image/mask to square using center padding.
    This should match your training-time square padding logic.
    """
    w, h = img.size
    if w == h:
        return img

    size = max(w, h)

    if img.mode == "L":
        new_img = Image.new("L", (size, size), color=fill)
    elif img.mode == "RGB":
        new_img = Image.new("RGB", (size, size), color=fill if isinstance(fill, tuple) else (fill, fill, fill))
    else:
        new_img = Image.new(img.mode, (size, size), color=fill)

    left = (size - w) // 2
    top = (size - h) // 2
    new_img.paste(img, (left, top))
    return new_img


def build_file_index(root: Path, exts: List[str]) -> Dict[str, Path]:
    """
    Build an index from file stem to path.
    Skip preview visualization folders.
    Prefer real mask files over preview images.
    """
    index = {}

    # 优先级：优先使用 png / bmp / tif 这类更可能是真实 mask 的文件
    preferred_exts = [".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"]

    for ext in preferred_exts:
        if ext not in exts:
            continue

        for p in root.rglob(f"*{ext}"):
            if not p.is_file():
                continue

            # 关键：跳过 _preview 目录
            if "_preview" in [part.lower() for part in p.parts]:
                continue

            # 如果同名已经存在，不要被后面的文件覆盖
            if p.stem not in index:
                index[p.stem] = p

    return index


def find_mask_for_image(image_path: Path, mask_index: Dict[str, Path]) -> Optional[Path]:
    """
    Match mask by image stem.
    Example:
        image: DSC_0120.jpg
        mask : DSC_0120.png
    """
    return mask_index.get(image_path.stem, None)


def compute_mask_ratio_after_padding(mask_path: Path, threshold: int = 127) -> Tuple[float, int, int, int, int]:
    """
    Return:
        ratio_after_pad,
        fg_pixels_after_pad,
        total_pixels_after_pad,
        original_w,
        original_h
    """
    with Image.open(mask_path) as m:
        m = m.convert("L")
        original_w, original_h = m.size

        padded = center_pad_to_square_pil(m, fill=0)
        arr = np.array(padded, dtype=np.uint8)

        fg = arr > threshold
        fg_pixels = int(fg.sum())
        total_pixels = int(arr.size)

        ratio = fg_pixels / max(total_pixels, 1)

    return ratio, fg_pixels, total_pixels, original_w, original_h


def compute_mask_bbox_ratio_after_padding(mask_path: Path, threshold: int = 127) -> Tuple[float, float, int, int]:
    """
    Compute bbox area ratio after padding.

    Return:
        bbox_ratio,
        bbox_aspect_ratio,
        bbox_w,
        bbox_h

    bbox_ratio = bbox_area / padded_square_area
    bbox_aspect_ratio = bbox_h / bbox_w
    """
    with Image.open(mask_path) as m:
        m = m.convert("L")
        padded = center_pad_to_square_pil(m, fill=0)
        arr = np.array(padded, dtype=np.uint8)

    ys, xs = np.where(arr > threshold)
    if len(xs) == 0 or len(ys) == 0:
        return 0.0, 0.0, 0, 0

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    bbox_w = x2 - x1 + 1
    bbox_h = y2 - y1 + 1
    bbox_area = bbox_w * bbox_h
    total_area = arr.shape[0] * arr.shape[1]

    bbox_ratio = bbox_area / max(total_area, 1)
    bbox_aspect = bbox_h / max(bbox_w, 1)

    return float(bbox_ratio), float(bbox_aspect), int(bbox_w), int(bbox_h)


def classify_sample(ratio: float, min_ratio: float, max_ratio: float) -> str:
    if ratio < min_ratio:
        return "too_small"
    if ratio > max_ratio:
        return "too_large"
    return "keep"


def safe_move_file(src: Path, src_root: Path, dst_root: Path) -> Path:
    """
    Move file while preserving relative path.
    """
    rel = src.relative_to(src_root)
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return dst


def delete_or_move_pair(
    image_path: Path,
    mask_path: Path,
    image_root: Path,
    mask_root: Path,
    move_to: Optional[Path],
    delete: bool,
) -> Tuple[str, str]:
    """
    Return:
        image_action, mask_action
    """
    if move_to is not None:
        image_dst_root = move_to / "images"
        mask_dst_root = move_to / "masks"

        image_dst = safe_move_file(image_path, image_root, image_dst_root)
        mask_dst = safe_move_file(mask_path, mask_root, mask_dst_root)

        return f"moved_to:{image_dst}", f"moved_to:{mask_dst}"

    if delete:
        image_path.unlink(missing_ok=True)
        mask_path.unlink(missing_ok=True)
        return "deleted", "deleted"

    return "dry_run", "dry_run"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_root", type=str, default="./data/unlabeled_train/images", help="Root directory of SSL images.")
    parser.add_argument("--mask_root", type=str, default="./data/unlabeled_train/foreground_masks", help="Root directory of foreground masks.")
    parser.add_argument("--out_csv", type=str, default="./runs/analysis_stage1_mask_ratio/mask_ratio_stats.csv", help="Output CSV path.")

    parser.add_argument("--min_ratio", type=float, default=0.06, help="Delete/move if padded mask ratio < min_ratio.")
    parser.add_argument("--max_ratio", type=float, default=0.55, help="Delete/move if padded mask ratio > max_ratio.")
    parser.add_argument("--threshold", type=int, default=127, help="Mask binarization threshold.")

    parser.add_argument("--move_to", type=str, default=None, help="Move bad image/mask pairs to this folder instead of deleting. Recommended.")
    parser.add_argument("--delete", action="store_true", help="Permanently delete bad image/mask pairs. Dangerous. Default is dry-run.")
    parser.add_argument("--delete_missing_mask", action="store_true", help="Also delete/move images that do not have a matched mask.")

    return parser.parse_args()


def main():
    args = parse_args()

    image_root = Path(args.image_root).resolve()
    mask_root = Path(args.mask_root).resolve()
    out_csv = Path(args.out_csv).resolve()
    move_to = Path(args.move_to).resolve() if args.move_to else out_csv.parent

    if not image_root.exists():
        raise FileNotFoundError(f"image_root not found: {image_root}")

    if not mask_root.exists():
        raise FileNotFoundError(f"mask_root not found: {mask_root}")

    if move_to is not None and args.delete:
        raise ValueError("Please use only one of --move_to or --delete, not both.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    image_paths: List[Path] = []
    for ext in IMAGE_EXTS:
        image_paths.extend([p for p in image_root.rglob(f"*{ext}") if p.is_file()])

    image_paths = sorted(image_paths)
    mask_index = build_file_index(mask_root, MASK_EXTS)

    rows = []

    total = 0
    keep_count = 0
    too_small_count = 0
    too_large_count = 0
    missing_mask_count = 0
    affected_count = 0

    print("=" * 80)
    print("Mask ratio filtering")
    print(f"image_root        : {image_root}")
    print(f"mask_root         : {mask_root}")
    print(f"out_csv           : {out_csv}")
    print(f"min_ratio         : {args.min_ratio}")
    print(f"max_ratio         : {args.max_ratio}")
    print(f"threshold         : {args.threshold}")
    print(f"move_to           : {move_to}")
    print(f"delete            : {args.delete}")
    print(f"delete_missing_mask: {args.delete_missing_mask}")
    print("=" * 80)

    for image_path in image_paths:
        total += 1
        mask_path = find_mask_for_image(image_path, mask_index)

        if mask_path is None:
            missing_mask_count += 1
            status = "missing_mask"
            image_action = "dry_run"
            mask_action = "none"

            if args.delete_missing_mask:
                if move_to is not None:
                    image_dst_root = move_to / "images_missing_mask"
                    image_dst = safe_move_file(image_path, image_root, image_dst_root)
                    image_action = f"moved_to:{image_dst}"
                    affected_count += 1
                elif args.delete:
                    image_path.unlink(missing_ok=True)
                    image_action = "deleted"
                    affected_count += 1

            rows.append({
                "image_path": str(image_path),
                "mask_path": "",
                "status": status,
                "mask_ratio_after_pad": "",
                "fg_pixels_after_pad": "",
                "total_pixels_after_pad": "",
                "mask_original_w": "",
                "mask_original_h": "",
                "bbox_ratio_after_pad": "",
                "bbox_aspect_h_over_w": "",
                "bbox_w": "",
                "bbox_h": "",
                "image_action": image_action,
                "mask_action": mask_action,
            })
            continue

        ratio, fg_pixels, total_pixels, ow, oh = compute_mask_ratio_after_padding(
            mask_path, threshold=args.threshold
        )

        bbox_ratio, bbox_aspect, bbox_w, bbox_h = compute_mask_bbox_ratio_after_padding(
            mask_path, threshold=args.threshold
        )

        status = classify_sample(ratio, args.min_ratio, args.max_ratio)

        if status == "keep":
            keep_count += 1
            image_action = "keep"
            mask_action = "keep"
        else:
            if status == "too_small":
                too_small_count += 1
            elif status == "too_large":
                too_large_count += 1

            image_action, mask_action = delete_or_move_pair(
                image_path=image_path,
                mask_path=mask_path,
                image_root=image_root,
                mask_root=mask_root,
                move_to=move_to,
                delete=args.delete,
            )

            if image_action != "dry_run":
                affected_count += 1

        rows.append({
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "status": status,
            "mask_ratio_after_pad": f"{ratio:.8f}",
            "fg_pixels_after_pad": fg_pixels,
            "total_pixels_after_pad": total_pixels,
            "mask_original_w": ow,
            "mask_original_h": oh,
            "bbox_ratio_after_pad": f"{bbox_ratio:.8f}",
            "bbox_aspect_h_over_w": f"{bbox_aspect:.6f}",
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
            "image_action": image_action,
            "mask_action": mask_action,
        })

    fieldnames = [
        "image_path",
        "mask_path",
        "status",
        "mask_ratio_after_pad",
        "fg_pixels_after_pad",
        "total_pixels_after_pad",
        "mask_original_w",
        "mask_original_h",
        "bbox_ratio_after_pad",
        "bbox_aspect_h_over_w",
        "bbox_w",
        "bbox_h",
        "image_action",
        "mask_action",
    ]

    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 80)
    print("Summary")
    print(f"Total images       : {total}")
    print(f"Keep               : {keep_count}")
    print(f"Too small          : {too_small_count}")
    print(f"Too large          : {too_large_count}")
    print(f"Missing mask       : {missing_mask_count}")
    print(f"Affected files     : {affected_count}")
    print(f"CSV saved to       : {out_csv}")
    print("=" * 80)

    if not args.delete and move_to is None:
        print("Current mode: DRY-RUN. No files were deleted or moved.")
        print("Use --move_to to move bad samples, or --delete to permanently delete them.")


if __name__ == "__main__":
    main()