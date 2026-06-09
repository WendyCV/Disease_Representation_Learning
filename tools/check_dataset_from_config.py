# -*- coding: utf-8 -*-
"""
check_dataset_from_config.py

Purpose:
    Quickly check whether dataset reading problems cause PyTorch DataLoader worker crashes.
    This script does NOT train a model. It only scans images and masks from a config file.

Typical usage:
    python check_dataset_from_config.py --config configs/plantseg_stage1.yaml

Optional:
    python check_dataset_from_config.py --config configs/plantseg_stage1.yaml --max_samples 1000
    python check_dataset_from_config.py --config configs/plantseg_stage1.yaml --output bad_samples.csv

Supported config style examples:

Example 1:
    data_root: D:/datasets/PlantSeg
    image_dir: images
    mask_dir: masks

Example 2:
    dataset:
      root: D:/datasets/PlantSeg
      image_dir: images
      mask_dir: masks

Example 3:
    train:
      images: D:/datasets/PlantSeg/images/train
      masks: D:/datasets/PlantSeg/masks/train

Example 4:
    images_dir: D:/datasets/PlantSeg/images
    masks_dir: D:/datasets/PlantSeg/masks

The script tries to infer common key names automatically.
"""

import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageFile
import numpy as np


ImageFile.LOAD_TRUNCATED_IMAGES = True


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_config(config_path: str) -> Dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = path.suffix.lower()

    if suffix in [".yaml", ".yml"]:
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is not installed. Please install it with: pip install pyyaml"
            )
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(f"Unsupported config format: {suffix}. Use .yaml, .yml, or .json")


def flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict[str, object]:
    items = {}
    if not isinstance(d, dict):
        return items

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def find_first_existing_path(config: Dict, candidate_keys: List[str]) -> Optional[Path]:
    flat = flatten_dict(config)

    for key in candidate_keys:
        for real_key, value in flat.items():
            if real_key.lower().endswith(key.lower()) or real_key.lower() == key.lower():
                if isinstance(value, str):
                    p = Path(value)
                    if p.exists():
                        return p

    return None


def find_value(config: Dict, candidate_keys: List[str]) -> Optional[str]:
    flat = flatten_dict(config)

    for key in candidate_keys:
        for real_key, value in flat.items():
            if real_key.lower().endswith(key.lower()) or real_key.lower() == key.lower():
                if isinstance(value, str):
                    return value

    return None


def resolve_data_dirs(config: Dict) -> Tuple[Path, Optional[Path]]:
    """
    Try to infer image_dir and mask_dir from config.
    Returns:
        image_dir, mask_dir
    """

    image_keys = [
        "data.train_dir",
    ]

    mask_keys = [
        "data.mask_root_dir",
    ]

    image_dir = find_first_existing_path(config, image_keys)
    mask_dir = find_first_existing_path(config, mask_keys)

    return image_dir, mask_dir


def collect_files(root: Path, exts: set) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def build_mask_index(mask_dir: Path) -> Dict[str, Path]:
    """
    Build mask index by stem.
    Example:
        image: abc.jpg
        mask candidates:
            abc.png
            abc_mask.png
            abc.jpg
    """
    mask_files = collect_files(mask_dir, MASK_EXTS)
    index = {}

    for p in mask_files:
        stem = p.stem
        index[stem] = p

        # Also support names like xxx_mask
        if stem.endswith("_mask"):
            index[stem[:-5]] = p

        if stem.endswith("-mask"):
            index[stem[:-5]] = p

    return index


def find_mask_for_image(image_path: Path, mask_index: Dict[str, Path]) -> Optional[Path]:
    stem = image_path.stem

    if stem in mask_index:
        return mask_index[stem]

    candidates = [
        stem + "_mask",
        stem + "-mask",
        stem.replace("image", "mask"),
        stem.replace("img", "mask"),
    ]

    for c in candidates:
        if c in mask_index:
            return mask_index[c]

    return None


def check_image(path: Path) -> Tuple[bool, Optional[Tuple[int, int]], Optional[str]]:
    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            img = img.convert("RGB")
            size = img.size
            _ = np.array(img)

        return True, size, None

    except Exception as e:
        return False, None, f"{type(e).__name__}: {str(e)}"


def check_mask(path: Path) -> Tuple[bool, Optional[Tuple[int, int]], Optional[Dict], Optional[str]]:
    try:
        with Image.open(path) as mask:
            mask.verify()

        with Image.open(path) as mask:
            mask = mask.convert("L")
            size = mask.size
            arr = np.array(mask)

        info = {
            "min": int(arr.min()) if arr.size > 0 else None,
            "max": int(arr.max()) if arr.size > 0 else None,
            "nonzero": int(np.count_nonzero(arr)),
            "total": int(arr.size),
            "nonzero_ratio": float(np.count_nonzero(arr) / arr.size) if arr.size > 0 else 0.0,
        }

        return True, size, info, None

    except Exception as e:
        return False, None, None, f"{type(e).__name__}: {str(e)}"


def write_bad_sample(writer, image_path, mask_path, problem_type, detail):
    writer.writerow({
        "image_path": str(image_path) if image_path else "",
        "mask_path": str(mask_path) if mask_path else "",
        "problem_type": problem_type,
        "detail": detail,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml/json config file.")
    parser.add_argument("--output", type=str, default="./runs_audit/plantseg_singleclass_bad_samples.csv", help="Output CSV file.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum number of samples to check. -1 means all.")
    parser.add_argument("--allow_missing_mask", action="store_true", help="Do not treat missing masks as errors.")
    parser.add_argument("--allow_empty_mask", action="store_true", help="Do not treat all-black masks as errors.")
    parser.add_argument("--check_size_match", action="store_true", help="Check whether image and mask sizes match.")
    args = parser.parse_args()

    config = load_config(args.config)

    image_dir, mask_dir = resolve_data_dirs(config)

    print("=" * 80)
    print("[INFO] Dataset checking started")
    print(f"[INFO] Config      : {args.config}")
    print(f"[INFO] Image dir   : {image_dir}")
    print(f"[INFO] Mask dir    : {mask_dir if mask_dir else 'None'}")
    print(f"[INFO] Output CSV  : {args.output}")
    print("=" * 80)

    image_files = collect_files(image_dir, IMAGE_EXTS)

    if args.max_samples > 0:
        image_files = image_files[:args.max_samples]

    print(f"[INFO] Found image files: {len(image_files)}")

    if len(image_files) == 0:
        print("[ERROR] No image files found. Please check image_dir in config.")
        sys.exit(1)

    mask_index = {}
    if mask_dir is not None:
        mask_index = build_mask_index(mask_dir)
        print(f"[INFO] Found indexed mask files: {len(mask_index)}")
    else:
        print("[WARN] No mask directory inferred. Only images will be checked.")

    bad_count = 0
    missing_mask_count = 0
    empty_mask_count = 0
    size_mismatch_count = 0
    image_error_count = 0
    mask_error_count = 0

    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["image_path", "mask_path", "problem_type", "detail"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, image_path in enumerate(image_files):
            if idx % 200 == 0:
                print(f"[INFO] Checking {idx}/{len(image_files)}...")

            ok_img, image_size, image_err = check_image(image_path)

            if not ok_img:
                bad_count += 1
                image_error_count += 1
                write_bad_sample(writer, image_path, None, "BAD_IMAGE", image_err)
                continue

            if mask_dir is None:
                continue

            mask_path = find_mask_for_image(image_path, mask_index)

            if mask_path is None:
                if not args.allow_missing_mask:
                    bad_count += 1
                    missing_mask_count += 1
                    write_bad_sample(
                        writer,
                        image_path,
                        None,
                        "MISSING_MASK",
                        f"No mask found for image stem: {image_path.stem}",
                    )
                continue

            ok_mask, mask_size, mask_info, mask_err = check_mask(mask_path)

            if not ok_mask:
                bad_count += 1
                mask_error_count += 1
                write_bad_sample(writer, image_path, mask_path, "BAD_MASK", mask_err)
                continue

            if args.check_size_match and image_size != mask_size:
                bad_count += 1
                size_mismatch_count += 1
                write_bad_sample(
                    writer,
                    image_path,
                    mask_path,
                    "SIZE_MISMATCH",
                    f"image_size={image_size}, mask_size={mask_size}",
                )

            if mask_info is not None:
                if mask_info["nonzero"] == 0 and not args.allow_empty_mask:
                    bad_count += 1
                    empty_mask_count += 1
                    write_bad_sample(
                        writer,
                        image_path,
                        mask_path,
                        "EMPTY_MASK",
                        f"mask is all black. nonzero_ratio={mask_info['nonzero_ratio']}",
                    )

    print("=" * 80)
    print("[SUMMARY]")
    print(f"Checked images       : {len(image_files)}")
    print(f"Bad samples          : {bad_count}")
    print(f"Bad images           : {image_error_count}")
    print(f"Missing masks        : {missing_mask_count}")
    print(f"Bad masks            : {mask_error_count}")
    print(f"Empty masks          : {empty_mask_count}")
    print(f"Size mismatches      : {size_mismatch_count}")
    print(f"Bad sample CSV       : {args.output}")
    print("=" * 80)

    if bad_count > 0:
        print("[RESULT] Dataset has suspicious or bad samples. Please inspect the CSV file.")
        sys.exit(2)
    else:
        print("[RESULT] No obvious dataset reading problem found.")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[FATAL] Script crashed:")
        traceback.print_exc()
        sys.exit(1)