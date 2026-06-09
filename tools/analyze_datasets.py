#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze dataset characteristics for YOLO-style detection datasets.

Supported inputs:
- images directory
- labels directory in YOLO txt format:
    class_id x_center y_center width height
- optional masks directory:
    same stem as image; non-zero pixels are treated as mask foreground

The script supports multiple datasets in one run:
    --dataset "Name|images_dir|labels_dir|masks_dir"
or:
    --dataset "Name|images_dir|labels_dir"

Main outputs:
- dataset_summary.csv
- bbox_instances.csv
- image_density_summary.csv
- size_group_summary.csv
- class_distribution_summary.csv
- spatial_distribution_summary.csv
- mask_quality_summary.csv
- several plots

Author: generated for dataset comparison of Durian / PlantDoc / PlantSeg.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class DatasetSpec:
    name: str
    images_dir: Path
    labels_dir: Path
    masks_dir: Optional[Path] = None


def parse_dataset_spec(spec: str) -> DatasetSpec:
    """
    Parse dataset spec:
        name|images_dir|labels_dir|masks_dir
    or:
        name|images_dir|labels_dir
    """
    parts = spec.split("|")
    if len(parts) not in (3, 4):
        raise ValueError(
            "Each --dataset must use format: "
            "'name|images_dir|labels_dir' or 'name|images_dir|labels_dir|masks_dir'. "
            f"Got: {spec}"
        )

    name = parts[0].strip()
    images_dir = Path(parts[1].strip()).resolve()
    labels_dir = Path(parts[2].strip()).resolve()
    masks_dir = Path(parts[3].strip()).resolve() if len(parts) == 4 and parts[3].strip() else None

    if not name:
        raise ValueError(f"Dataset name is empty in spec: {spec}")
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir does not exist: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels_dir does not exist: {labels_dir}")
    if masks_dir is not None and not masks_dir.exists():
        raise FileNotFoundError(f"masks_dir does not exist: {masks_dir}")

    return DatasetSpec(name=name, images_dir=images_dir, labels_dir=labels_dir, masks_dir=masks_dir)


def list_images(images_dir: Path) -> List[Path]:
    images = [
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(images)


def build_stem_index(root: Path, exts: set[str]) -> Dict[str, Path]:
    """
    Build a stem -> path index. If duplicated stems exist, keep the first sorted path.
    """
    index: Dict[str, Path] = {}
    if root is None or not root.exists():
        return index

    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            index.setdefault(p.stem, p)

    return index


def read_image_size(image_path: Path) -> Tuple[int, int]:
    with Image.open(image_path) as img:
        w, h = img.size
    return int(w), int(h)


def read_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Read YOLO label:
        cls x y w h
    values are normalized.
    """
    if not label_path.exists():
        return []

    rows = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f.readlines(), start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                print(f"[WARN] Invalid label line: {label_path}:{line_no} -> {line}")
                continue

            try:
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
            except ValueError:
                print(f"[WARN] Cannot parse label line: {label_path}:{line_no} -> {line}")
                continue

            rows.append((cls, x, y, w, h))

    return rows


def yolo_to_xyxy(
    x: float,
    y: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    x1 = (x - w / 2.0) * img_w
    y1 = (y - h / 2.0) * img_h
    x2 = (x + w / 2.0) * img_w
    y2 = (y + h / 2.0) * img_h
    return x1, y1, x2, y2


def clip_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    x1 = float(np.clip(x1, 0, img_w))
    y1 = float(np.clip(y1, 0, img_h))
    x2 = float(np.clip(x2, 0, img_w))
    y2 = float(np.clip(y2, 0, img_h))
    return x1, y1, x2, y2


def align_bbox_square_pad(
    x: float,
    y: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float, int, int]:
    """
    Simulate center padding to square.

    Original image W x H is padded to S x S, S=max(W,H).
    Return normalized x,y,w,h under S x S coordinate.
    """
    s = max(img_w, img_h)
    pad_x = (s - img_w) / 2.0
    pad_y = (s - img_h) / 2.0

    x1, y1, x2, y2 = yolo_to_xyxy(x, y, w, h, img_w, img_h)
    x1 += pad_x
    x2 += pad_x
    y1 += pad_y
    y2 += pad_y

    new_w = max((x2 - x1) / s, 0.0)
    new_h = max((y2 - y1) / s, 0.0)
    new_x = ((x1 + x2) / 2.0) / s
    new_y = ((y1 + y2) / 2.0) / s

    return new_x, new_y, new_w, new_h, s, s


def align_bbox_original(
    x: float,
    y: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float, int, int]:
    return x, y, w, h, img_w, img_h


def get_size_group(
    area_ratio: float,
    small_thr: float,
    medium_thr: float,
    full_thr: float,
) -> str:
    if area_ratio < small_thr:
        return "small"
    if area_ratio < medium_thr:
        return "medium"
    if area_ratio < full_thr:
        return "large"
    return "full_image"


def load_mask_binary(
    mask_path: Path,
    img_w: int,
    img_h: int,
    alignment: str,
    mask_threshold: int,
) -> Optional[np.ndarray]:
    """
    Load mask as binary array.

    If mask size differs from image size, resize to image size first.
    If alignment == square_pad, pad mask to square.
    """
    if mask_path is None or not mask_path.exists():
        return None

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        print(f"[WARN] Failed to read mask: {mask_path}")
        return None

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if mask.shape[1] != img_w or mask.shape[0] != img_h:
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    binary = (mask > mask_threshold).astype(np.uint8)

    if alignment == "square_pad":
        s = max(img_w, img_h)
        padded = np.zeros((s, s), dtype=np.uint8)
        pad_x = int(round((s - img_w) / 2.0))
        pad_y = int(round((s - img_h) / 2.0))
        padded[pad_y:pad_y + img_h, pad_x:pad_x + img_w] = binary
        binary = padded

    return binary


def compute_mask_components(binary_mask: np.ndarray) -> Dict[str, float]:
    """
    Compute connected component statistics for a binary mask.
    """
    if binary_mask is None:
        return {
            "mask_available": 0,
            "mask_area_ratio": np.nan,
            "components_count": np.nan,
            "components_area_mean": np.nan,
            "components_area_median": np.nan,
            "small_component_ratio": np.nan,
        }

    h, w = binary_mask.shape
    total_pixels = h * w
    mask_area = int(binary_mask.sum())
    mask_area_ratio = mask_area / total_pixels if total_pixels > 0 else np.nan

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8),
        connectivity=8,
    )

    component_areas = []
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area > 0:
            component_areas.append(area / total_pixels)

    if len(component_areas) == 0:
        return {
            "mask_available": 1,
            "mask_area_ratio": mask_area_ratio,
            "components_count": 0,
            "components_area_mean": 0.0,
            "components_area_median": 0.0,
            "small_component_ratio": 0.0,
        }

    component_areas_np = np.array(component_areas, dtype=np.float32)
    small_component_ratio = float((component_areas_np < 0.001).mean())

    return {
        "mask_available": 1,
        "mask_area_ratio": float(mask_area_ratio),
        "components_count": int(len(component_areas)),
        "components_area_mean": float(component_areas_np.mean()),
        "components_area_median": float(np.median(component_areas_np)),
        "small_component_ratio": small_component_ratio,
    }


def bbox_mask_fill_ratio(
    binary_mask: Optional[np.ndarray],
    x: float,
    y: float,
    w: float,
    h: float,
    aligned_w: int,
    aligned_h: int,
) -> float:
    """
    Compute mask_area_inside_bbox / bbox_area.
    """
    if binary_mask is None:
        return np.nan

    x1, y1, x2, y2 = yolo_to_xyxy(x, y, w, h, aligned_w, aligned_h)
    x1, y1, x2, y2 = clip_xyxy(x1, y1, x2, y2, aligned_w, aligned_h)

    xi1 = int(math.floor(x1))
    yi1 = int(math.floor(y1))
    xi2 = int(math.ceil(x2))
    yi2 = int(math.ceil(y2))

    if xi2 <= xi1 or yi2 <= yi1:
        return np.nan

    crop = binary_mask[yi1:yi2, xi1:xi2]
    bbox_area = crop.size
    if bbox_area <= 0:
        return np.nan

    return float(crop.sum() / bbox_area)


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) > 0 else np.nan


def safe_median(series: pd.Series) -> float:
    return float(series.median()) if len(series) > 0 else np.nan


def summarize_dataset(
    dataset_name: str,
    image_df: pd.DataFrame,
    bbox_df: pd.DataFrame,
    small_thr: float,
    medium_thr: float,
    full_thr: float,
) -> Dict[str, object]:
    num_images = len(image_df)
    num_boxes = len(bbox_df)
    num_classes = int(bbox_df["class_id"].nunique()) if num_boxes > 0 else 0

    images_with_labels = int((image_df["box_count"] > 0).sum())
    empty_images = int((image_df["box_count"] == 0).sum())

    summary = {
        "dataset": dataset_name,
        "num_images": num_images,
        "num_boxes": num_boxes,
        "num_classes": num_classes,
        "images_with_labels": images_with_labels,
        "empty_images": empty_images,
        "boxes_per_image_mean": safe_mean(image_df["box_count"]),
        "boxes_per_image_median": safe_median(image_df["box_count"]),
        "boxes_per_image_p90": float(image_df["box_count"].quantile(0.90)) if num_images > 0 else np.nan,
        "boxes_per_image_max": int(image_df["box_count"].max()) if num_images > 0 else 0,
        "single_object_image_ratio": float((image_df["box_count"] == 1).mean()) if num_images > 0 else np.nan,
        "multi_object_image_ratio": float((image_df["box_count"] >= 2).mean()) if num_images > 0 else np.nan,
        "dense_image_ratio_ge5": float((image_df["box_count"] >= 5).mean()) if num_images > 0 else np.nan,
        "dense_image_ratio_ge10": float((image_df["box_count"] >= 10).mean()) if num_images > 0 else np.nan,
        "image_aspect_ratio_mean": safe_mean(image_df["image_aspect_ratio"]),
        "image_aspect_ratio_median": safe_median(image_df["image_aspect_ratio"]),
    }

    if num_boxes == 0:
        extra = {
            "area_ratio_mean": np.nan,
            "area_ratio_median": np.nan,
            "area_ratio_p25": np.nan,
            "area_ratio_p75": np.nan,
            "area_ratio_p90": np.nan,
            "small_ratio": np.nan,
            "medium_ratio": np.nan,
            "large_ratio": np.nan,
            "full_image_ratio": np.nan,
            "center_region_ratio": np.nan,
            "center_distance_mean": np.nan,
            "edge_touch_ratio": np.nan,
            "full_or_edge_ratio": np.nan,
            "aspect_ratio_mean": np.nan,
            "aspect_ratio_median": np.nan,
            "wide_box_ratio": np.nan,
            "tall_box_ratio": np.nan,
        }
    else:
        extra = {
            "area_ratio_mean": safe_mean(bbox_df["area_ratio"]),
            "area_ratio_median": safe_median(bbox_df["area_ratio"]),
            "area_ratio_p25": float(bbox_df["area_ratio"].quantile(0.25)),
            "area_ratio_p75": float(bbox_df["area_ratio"].quantile(0.75)),
            "area_ratio_p90": float(bbox_df["area_ratio"].quantile(0.90)),
            "small_ratio": float((bbox_df["size_group"] == "small").mean()),
            "medium_ratio": float((bbox_df["size_group"] == "medium").mean()),
            "large_ratio": float((bbox_df["size_group"] == "large").mean()),
            "full_image_ratio": float((bbox_df["size_group"] == "full_image").mean()),
            "center_region_ratio": float(bbox_df["center_region"].mean()),
            "center_distance_mean": safe_mean(bbox_df["center_distance"]),
            "edge_touch_ratio": float(bbox_df["edge_touch"].mean()),
            "full_or_edge_ratio": float(((bbox_df["size_group"] == "full_image") | bbox_df["edge_touch"]).mean()),
            "aspect_ratio_mean": safe_mean(bbox_df["bbox_aspect_ratio"]),
            "aspect_ratio_median": safe_median(bbox_df["bbox_aspect_ratio"]),
            "wide_box_ratio": float((bbox_df["bbox_aspect_ratio"] > 2.0).mean()),
            "tall_box_ratio": float((bbox_df["bbox_aspect_ratio"] < 0.5).mean()),
        }

    summary.update(extra)
    return summary


def analyze_one_dataset(
    spec: DatasetSpec,
    alignment: str,
    small_thr: float,
    medium_thr: float,
    full_thr: float,
    edge_thr: float,
    center_min: float,
    center_max: float,
    mask_threshold: int,
    target_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"[INFO] Analyzing dataset: {spec.name}")
    print(f"       images: {spec.images_dir}")
    print(f"       labels: {spec.labels_dir}")
    print(f"       masks : {spec.masks_dir}")

    images = list_images(spec.images_dir)
    label_index = build_stem_index(spec.labels_dir, {".txt"})
    mask_index = build_stem_index(spec.masks_dir, MASK_EXTS) if spec.masks_dir else {}

    image_rows = []
    bbox_rows = []
    mask_rows = []

    for image_path in images:
        stem = image_path.stem
        img_w, img_h = read_image_size(image_path)
        label_path = label_index.get(stem, spec.labels_dir / f"{stem}.txt")
        mask_path = mask_index.get(stem, None)

        labels = read_yolo_label(label_path)

        if alignment == "square_pad":
            aligned_w = aligned_h = max(img_w, img_h)
        elif alignment == "original":
            aligned_w, aligned_h = img_w, img_h
        else:
            raise ValueError(f"Unsupported alignment: {alignment}")

        binary_mask = None
        mask_stats = {
            "mask_available": 0,
            "mask_area_ratio": np.nan,
            "components_count": np.nan,
            "components_area_mean": np.nan,
            "components_area_median": np.nan,
            "small_component_ratio": np.nan,
        }

        if mask_path is not None:
            binary_mask = load_mask_binary(
                mask_path=mask_path,
                img_w=img_w,
                img_h=img_h,
                alignment=alignment,
                mask_threshold=mask_threshold,
            )
            mask_stats = compute_mask_components(binary_mask)

        image_rows.append({
            "dataset": spec.name,
            "image": str(image_path),
            "stem": stem,
            "image_width": img_w,
            "image_height": img_h,
            "aligned_width": aligned_w,
            "aligned_height": aligned_h,
            "image_aspect_ratio": img_w / img_h if img_h > 0 else np.nan,
            "label_path": str(label_path) if label_path.exists() else "",
            "mask_path": str(mask_path) if mask_path is not None else "",
            "box_count": len(labels),
            **mask_stats,
        })

        mask_rows.append({
            "dataset": spec.name,
            "image": str(image_path),
            "stem": stem,
            "mask_path": str(mask_path) if mask_path is not None else "",
            **mask_stats,
        })

        for idx, (cls, x, y, w, h) in enumerate(labels):
            if alignment == "square_pad":
                ax, ay, aw, ah, awidth, aheight = align_bbox_square_pad(
                    x, y, w, h, img_w, img_h
                )
            else:
                ax, ay, aw, ah, awidth, aheight = align_bbox_original(
                    x, y, w, h, img_w, img_h
                )

            area_ratio = max(aw, 0.0) * max(ah, 0.0)
            size_group = get_size_group(
                area_ratio=area_ratio,
                small_thr=small_thr,
                medium_thr=medium_thr,
                full_thr=full_thr,
            )

            x1 = ax - aw / 2.0
            y1 = ay - ah / 2.0
            x2 = ax + aw / 2.0
            y2 = ay + ah / 2.0

            edge_touch = (
                x1 <= edge_thr or
                y1 <= edge_thr or
                x2 >= (1.0 - edge_thr) or
                y2 >= (1.0 - edge_thr)
            )

            center_region = (
                center_min <= ax <= center_max and
                center_min <= ay <= center_max
            )

            center_distance = math.sqrt((ax - 0.5) ** 2 + (ay - 0.5) ** 2)
            bbox_aspect_ratio = aw / ah if ah > 0 else np.nan

            fill_ratio = bbox_mask_fill_ratio(
                binary_mask=binary_mask,
                x=ax,
                y=ay,
                w=aw,
                h=ah,
                aligned_w=awidth,
                aligned_h=aheight,
            )

            # Estimate bbox pixel size on the target square input, e.g., 640x640.
            # This does not resize images. It only converts normalized bbox size
            # after alignment into target-input pixel scale.
            bbox_w_px_target = aw * target_size
            bbox_h_px_target = ah * target_size
            bbox_area_px_target = bbox_w_px_target * bbox_h_px_target

            bbox_rows.append({
                "dataset": spec.name,
                "image": str(image_path),
                "stem": stem,
                "bbox_index": idx,
                "class_id": cls,
                "x_center": ax,
                "y_center": ay,
                "width": aw,
                "height": ah,
                "area_ratio": area_ratio,
                "bbox_w_px_target": bbox_w_px_target,
                "bbox_h_px_target": bbox_h_px_target,
                "bbox_area_px_target": bbox_area_px_target,
                "target_size": target_size,
                "size_group": size_group,
                "bbox_aspect_ratio": bbox_aspect_ratio,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "edge_touch": bool(edge_touch),
                "center_region": bool(center_region),
                "center_distance": center_distance,
                "mask_fill_ratio": fill_ratio,
                "image_width": img_w,
                "image_height": img_h,
                "aligned_width": awidth,
                "aligned_height": aheight,
            })
    
    image_df = pd.DataFrame(image_rows)
    bbox_df = pd.DataFrame(bbox_rows)
    mask_df = pd.DataFrame(mask_rows)

    return image_df, bbox_df, mask_df


def make_summary_tables(
    image_df: pd.DataFrame,
    bbox_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    small_thr: float,
    medium_thr: float,
    full_thr: float,
) -> Dict[str, pd.DataFrame]:
    dataset_summaries = []
    for dataset_name in image_df["dataset"].unique():
        one_img = image_df[image_df["dataset"] == dataset_name]
        one_box = bbox_df[bbox_df["dataset"] == dataset_name]
        dataset_summaries.append(
            summarize_dataset(
                dataset_name=dataset_name,
                image_df=one_img,
                bbox_df=one_box,
                small_thr=small_thr,
                medium_thr=medium_thr,
                full_thr=full_thr,
            )
        )

    dataset_summary = pd.DataFrame(dataset_summaries)

    if len(bbox_df) > 0:
        size_group_summary = (
            bbox_df.groupby(["dataset", "size_group"])
            .size()
            .reset_index(name="box_count")
        )
        size_group_summary["box_ratio"] = size_group_summary.groupby("dataset")["box_count"].transform(
            lambda x: x / x.sum()
        )

        class_distribution = (
            bbox_df.groupby(["dataset", "class_id"])
            .agg(
                box_count=("class_id", "size"),
                area_ratio_mean=("area_ratio", "mean"),
                area_ratio_median=("area_ratio", "median"),
                small_ratio=("size_group", lambda s: float((s == "small").mean())),
                medium_ratio=("size_group", lambda s: float((s == "medium").mean())),
                large_ratio=("size_group", lambda s: float((s == "large").mean())),
                full_image_ratio=("size_group", lambda s: float((s == "full_image").mean())),
            )
            .reset_index()
        )

        image_per_class = (
            bbox_df.groupby(["dataset", "class_id"])["stem"]
            .nunique()
            .reset_index(name="image_count")
        )
        class_distribution = class_distribution.merge(
            image_per_class,
            on=["dataset", "class_id"],
            how="left",
        )

        spatial_distribution = (
            bbox_df.groupby("dataset")
            .agg(
                center_x_mean=("x_center", "mean"),
                center_x_std=("x_center", "std"),
                center_y_mean=("y_center", "mean"),
                center_y_std=("y_center", "std"),
                center_distance_mean=("center_distance", "mean"),
                center_region_ratio=("center_region", "mean"),
                edge_touch_ratio=("edge_touch", "mean"),
            )
            .reset_index()
        )
    else:
        size_group_summary = pd.DataFrame()
        class_distribution = pd.DataFrame()
        spatial_distribution = pd.DataFrame()

    image_density_summary = (
        image_df.groupby("dataset")
        .agg(
            num_images=("image", "size"),
            boxes_per_image_mean=("box_count", "mean"),
            boxes_per_image_median=("box_count", "median"),
            boxes_per_image_p90=("box_count", lambda x: float(x.quantile(0.90))),
            boxes_per_image_max=("box_count", "max"),
            single_object_image_ratio=("box_count", lambda x: float((x == 1).mean())),
            multi_object_image_ratio=("box_count", lambda x: float((x >= 2).mean())),
            dense_image_ratio_ge5=("box_count", lambda x: float((x >= 5).mean())),
            dense_image_ratio_ge10=("box_count", lambda x: float((x >= 10).mean())),
        )
        .reset_index()
    )

    if len(mask_df) > 0 and "mask_available" in mask_df.columns:
        mask_quality_summary = (
            mask_df.groupby("dataset")
            .agg(
                images_with_mask=("mask_available", "sum"),
                mask_area_ratio_mean=("mask_area_ratio", "mean"),
                mask_area_ratio_median=("mask_area_ratio", "median"),
                components_per_image_mean=("components_count", "mean"),
                components_per_image_median=("components_count", "median"),
                components_area_mean=("components_area_mean", "mean"),
                small_component_ratio_mean=("small_component_ratio", "mean"),
            )
            .reset_index()
        )

        if len(bbox_df) > 0 and "mask_fill_ratio" in bbox_df.columns:
            fill_summary = (
                bbox_df.groupby("dataset")
                .agg(
                    bbox_mask_fill_ratio_mean=("mask_fill_ratio", "mean"),
                    bbox_mask_fill_ratio_median=("mask_fill_ratio", "median"),
                    bbox_mask_fill_ratio_p25=("mask_fill_ratio", lambda x: float(x.dropna().quantile(0.25)) if x.dropna().shape[0] > 0 else np.nan),
                    bbox_mask_fill_ratio_p75=("mask_fill_ratio", lambda x: float(x.dropna().quantile(0.75)) if x.dropna().shape[0] > 0 else np.nan),
                )
                .reset_index()
            )
            mask_quality_summary = mask_quality_summary.merge(fill_summary, on="dataset", how="left")
    else:
        mask_quality_summary = pd.DataFrame()

    return {
        "dataset_summary": dataset_summary,
        "image_density_summary": image_density_summary,
        "size_group_summary": size_group_summary,
        "class_distribution_summary": class_distribution,
        "spatial_distribution_summary": spatial_distribution,
        "mask_quality_summary": mask_quality_summary,
    }


def save_tables(
    out_dir: Path,
    prefix: str,
    image_df: pd.DataFrame,
    bbox_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    image_df.to_csv(out_dir / f"{prefix}_image_instances.csv", index=False, encoding="utf-8-sig")
    bbox_df.to_csv(out_dir / f"{prefix}_bbox_instances.csv", index=False, encoding="utf-8-sig")
    mask_df.to_csv(out_dir / f"{prefix}_mask_instances.csv", index=False, encoding="utf-8-sig")

    for name, df in tables.items():
        df.to_csv(out_dir / f"{prefix}_{name}.csv", index=False, encoding="utf-8-sig")


def plot_area_histogram(bbox_df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    if len(bbox_df) == 0:
        return

    plt.figure(figsize=(8, 5))
    for dataset_name, sub in bbox_df.groupby("dataset"):
        vals = sub["area_ratio"].dropna().clip(lower=1e-6)
        plt.hist(vals, bins=50, alpha=0.4, label=dataset_name)

    plt.xscale("log")
    plt.xlabel("BBox area ratio, log scale")
    plt.ylabel("Count")
    plt.title("BBox area ratio distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_area_ratio_histogram.png", dpi=300)
    plt.close()


def plot_size_group_bar(size_group_summary: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    if len(size_group_summary) == 0:
        return

    pivot = size_group_summary.pivot(
        index="dataset",
        columns="size_group",
        values="box_ratio",
    ).fillna(0.0)

    order = [c for c in ["small", "medium", "large", "full_image"] if c in pivot.columns]
    pivot = pivot[order]

    ax = pivot.plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("Box ratio")
    ax.set_title("Size group ratio by dataset")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_size_group_barplot.png", dpi=300)
    plt.close()


def plot_boxes_per_image(image_df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    if len(image_df) == 0:
        return

    plt.figure(figsize=(8, 5))
    for dataset_name, sub in image_df.groupby("dataset"):
        plt.hist(sub["box_count"], bins=40, alpha=0.4, label=dataset_name)

    plt.xlabel("Boxes per image")
    plt.ylabel("Image count")
    plt.title("Boxes per image distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_boxes_per_image_histogram.png", dpi=300)
    plt.close()


def plot_bbox_center_scatter(bbox_df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    if len(bbox_df) == 0:
        return

    datasets = list(bbox_df["dataset"].unique())
    n = len(datasets)

    plt.figure(figsize=(5 * n, 5))
    for i, dataset_name in enumerate(datasets, start=1):
        sub = bbox_df[bbox_df["dataset"] == dataset_name]
        plt.subplot(1, n, i)
        plt.scatter(sub["x_center"], sub["y_center"], s=4, alpha=0.35)
        plt.xlim(0, 1)
        plt.ylim(1, 0)
        plt.xlabel("x center")
        plt.ylabel("y center")
        plt.title(dataset_name)
        plt.grid(True, linewidth=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_bbox_center_scatter.png", dpi=300)
    plt.close()


def plot_class_distribution(class_distribution: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    if len(class_distribution) == 0:
        return

    datasets = list(class_distribution["dataset"].unique())

    for dataset_name in datasets:
        sub = class_distribution[class_distribution["dataset"] == dataset_name].copy()
        sub = sub.sort_values("box_count", ascending=False)

        plt.figure(figsize=(10, 5))
        plt.bar(sub["class_id"].astype(str), sub["box_count"])
        plt.xlabel("Class ID")
        plt.ylabel("Box count")
        plt.title(f"Class distribution: {dataset_name}")
        plt.xticks(rotation=60, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_{dataset_name}_class_distribution_barplot.png", dpi=300)
        plt.close()


def save_metadata(
    out_dir: Path,
    prefix: str,
    args: argparse.Namespace,
    specs: List[DatasetSpec],
) -> None:
    meta = {
        "datasets": [
            {
                "name": s.name,
                "images_dir": str(s.images_dir),
                "labels_dir": str(s.labels_dir),
                "masks_dir": str(s.masks_dir) if s.masks_dir else None,
            }
            for s in specs
        ],
        "args": vars(args),
    }
    with open(out_dir / f"{prefix}_analysis_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze and compare YOLO-style datasets: images, labels, optional masks."
    )

    parser.add_argument(
        "--dataset",
        action="append",
        default=[
            "DurianLeaf|./data/labeled_train|./data/labeled_train|./data/unlabeled_train/foreground_masks",
            "PlantDoc|./data/PlantDoc/images|./data/PlantDoc/labels|./data/PlantDoc/foreground_masks",
            "PlantSeg|./data/PlantSeg/images|./data/PlantSeg/labels|./data/PlantSeg/foreground_masks"
        ],
        help=(
            "Dataset spec. Use format: "
            "'name|images_dir|labels_dir' or 'name|images_dir|labels_dir|masks_dir'. "
            "Can be used multiple times."
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="./runs_dataset_analysis/compare_all",
        help="Output directory."
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="all",
        help="Output filename prefix."
    )

    parser.add_argument(
        "--target-size",
        type=int,
        default=640,
        help="Target square image size used to estimate bbox pixel size after alignment."
    )

    parser.add_argument(
        "--alignment",
        type=str,
        default="square_pad",
        choices=["original", "square_pad"],
        help=(
            "Coordinate alignment mode. "
            "'original': use original image coordinate. "
            "'square_pad': simulate center padding to square before computing normalized bbox metrics."
        ),
    )

    parser.add_argument(
        "--small-thr",
        type=float,
        default=0.03,
        help="Area ratio threshold for small bbox."
    )

    parser.add_argument(
        "--medium-thr",
        type=float,
        default=0.15,
        help="Area ratio threshold for medium bbox. Large begins at this value."
    )

    parser.add_argument(
        "--full-thr",
        type=float,
        default=0.50,
        help="Area ratio threshold for full-image bbox."
    )

    parser.add_argument(
        "--edge-thr",
        type=float,
        default=0.02,
        help="Threshold for edge-touching bbox."
    )

    parser.add_argument(
        "--center-min",
        type=float,
        default=0.25,
        help="Minimum normalized center coordinate for center-region check."
    )

    parser.add_argument(
        "--center-max",
        type=float,
        default=0.75,
        help="Maximum normalized center coordinate for center-region check."
    )

    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=0,
        help="Mask threshold. Pixels greater than this value are foreground."
    )

    parser.add_argument(
        "--plots",
        action="store_true",
        help="If set, save diagnostic plots."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [parse_dataset_spec(s) for s in args.dataset]

    all_image_dfs = []
    all_bbox_dfs = []
    all_mask_dfs = []

    print("[INFO] Analysis settings")
    print(f"       alignment : {args.alignment}")
    print(f"       small_thr : {args.small_thr}")
    print(f"       medium_thr: {args.medium_thr}")
    print(f"       full_thr  : {args.full_thr}")
    print(f"       out_dir   : {out_dir}")
    print(f"       prefix    : {args.prefix}")

    for spec in specs:
        image_df, bbox_df, mask_df = analyze_one_dataset(
            spec=spec,
            alignment=args.alignment,
            small_thr=args.small_thr,
            medium_thr=args.medium_thr,
            full_thr=args.full_thr,
            edge_thr=args.edge_thr,
            center_min=args.center_min,
            center_max=args.center_max,
            mask_threshold=args.mask_threshold,
            target_size=args.target_size,
        )
        all_image_dfs.append(image_df)
        all_bbox_dfs.append(bbox_df)
        all_mask_dfs.append(mask_df)

    image_df_all = pd.concat(all_image_dfs, ignore_index=True) if all_image_dfs else pd.DataFrame()
    bbox_df_all = pd.concat(all_bbox_dfs, ignore_index=True) if all_bbox_dfs else pd.DataFrame()
    mask_df_all = pd.concat(all_mask_dfs, ignore_index=True) if all_mask_dfs else pd.DataFrame()

    tables = make_summary_tables(
        image_df=image_df_all,
        bbox_df=bbox_df_all,
        mask_df=mask_df_all,
        small_thr=args.small_thr,
        medium_thr=args.medium_thr,
        full_thr=args.full_thr,
    )

    save_tables(
        out_dir=out_dir,
        prefix=args.prefix,
        image_df=image_df_all,
        bbox_df=bbox_df_all,
        mask_df=mask_df_all,
        tables=tables,
    )

    save_metadata(
        out_dir=out_dir,
        prefix=args.prefix,
        args=args,
        specs=specs,
    )

    if args.plots:
        plot_area_histogram(bbox_df_all, out_dir, args.prefix)
        plot_size_group_bar(tables["size_group_summary"], out_dir, args.prefix)
        plot_boxes_per_image(image_df_all, out_dir, args.prefix)
        plot_bbox_center_scatter(bbox_df_all, out_dir, args.prefix)
        plot_class_distribution(tables["class_distribution_summary"], out_dir, args.prefix)

    print("[INFO] Done.")
    print(f"[INFO] Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()