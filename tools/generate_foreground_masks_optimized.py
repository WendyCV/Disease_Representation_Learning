# -*- coding: utf-8 -*-
"""
Generate foreground masks for durian leaf SSL images using SAM2. Balanced v3.

This version is optimized for FOREGROUND segmentation rather than single-leaf instance segmentation.
Main changes compared with the previous prompted+GrabCut version:
1. Do not force images to vertical orientation and do not overwrite original images.
2. Use multiple broad/overlapping box prompts instead of a single center-biased prompt.
3. Merge multiple valid SAM2 masks so a leaf cluster can be retained as one foreground region.
4. Reduce center bias and avoid selecting only one largest leaf.
5. Keep multiple connected components after refinement.
6. Save binary masks and preview images for visual inspection.

Recommended command:
python tools/generate_foreground_masks_optimized.py \
  --input_root ./data/unlabeled_train/images \
  --output_root ./data/unlabeled_train/foreground_masks_v3 \
  --sam2_root E:/sam2 \
  --max_size 1024 \
  --overwrite
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

warnings.filterwarnings(action="ignore", category=UserWarning)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


# =========================================================
# Configuration
# =========================================================
DEFAULT_CONFIG = {
    "RUNTIME": {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "use_autocast_when_cuda": True,
        "max_size": 1024,
        "overwrite": False,
        "save_preview": True,
        "save_init_mask": False,
    },

    # Candidate boxes. These are deliberately broad and overlapping.
    # The goal is foreground leaf/leaf-cluster segmentation, not selecting one central object.
    "BOX": {
        "relative_boxes": [
            [0.02, 0.02, 0.98, 0.98],  # almost full image
            [0.06, 0.04, 0.94, 0.96],  # broad center
            [0.00, 0.05, 0.62, 0.95],  # left region
            [0.38, 0.05, 1.00, 0.95],  # right region
            [0.08, 0.00, 0.92, 0.62],  # upper region
            [0.08, 0.38, 0.92, 1.00],  # lower region
            [0.00, 0.00, 0.72, 0.72],  # upper-left broad
            [0.28, 0.00, 1.00, 0.72],  # upper-right broad
            [0.00, 0.28, 0.72, 1.00],  # lower-left broad
            [0.28, 0.28, 1.00, 1.00],  # lower-right broad
        ],
        "use_grabcut_boxes": True,
        "grabcut_rects": [
            [0.02, 0.02, 0.98, 0.98],
            [0.08, 0.05, 0.92, 0.95],
            [0.00, 0.05, 0.65, 0.95],
            [0.35, 0.05, 1.00, 0.95],
        ],
        "grabcut_iter_count": 3,
        "grabcut_min_component_area": 800,
        "grabcut_max_boxes": 5,
        "grabcut_expand_x": 0.18,
        "grabcut_expand_y": 0.22,
    },

    "SAM2": {
        "multimask_output": True,
    },

    # Mask candidate filtering before merging.
    # BALANCED version:
    # - v1 was too permissive and could merge background blobs.
    # - v2 was too strict and point prompts often collapsed to a single leaf or tiny fragment.
    # - This version returns to box-dominant prompting and uses leafness only as a soft score.
    "CANDIDATE": {
        "min_area_ratio": 0.004,
        "max_area_ratio": 0.58,
        "min_bbox_fill": 0.06,
        "min_solidity": 0.08,
        "max_edge_touch": 0.52,
        "score_keep_delta": 0.32,
        "min_score_to_keep": 0.16,
        "max_candidates_to_merge": 10,
        "nms_iou_threshold": 0.86,

        # Leafness constraints are SOFT in this version.
        # Strict aspect/contrast constraints caused missed dark leaves and tiny-fragment fallback.
        "min_oriented_aspect": 1.05,
        "soft_oriented_aspect": 1.30,
        "min_boundary_contrast": 4.0,
        "soft_boundary_contrast": 10.0,

        # Point prompting is disabled by default.
        # In the bad cases, positive points made SAM2 segment only one local leaf/fragment.
        # Enable manually only for single-leaf close-up images.
        "use_point_prompts": False,
        "max_positive_points": 4,
        "seed_min_area": 900,
        "seed_h_low": 22,
        "seed_h_high": 100,
        "seed_s_min": 18,
        "seed_v_min": 18,
        "seed_v_max": 235,
        "use_corner_negative_points": False,
        "corner_negative_margin": 5,
    },

    # Final foreground refinement.
    "MASK": {
        "min_region_area": 420,
        "max_components": 20,
        "close_kernel": 9,
        "open_kernel": 3,
        "dilate_kernel": 5,
        "dilate_iterations": 1,
        "fill_holes": True,
        "min_final_ratio": 0.035,
        "max_final_ratio": 0.65,
    },

    "PREVIEW": {
        "fill_color": [0, 255, 0],
        "candidate_contour_color": [0, 128, 255],
        "final_contour_color": [255, 0, 255],
        "box_color": [255, 128, 0],
        "fill_alpha": 0.38,
        "contour_thickness": 2,
        "box_thickness": 1,
    },
}


# =========================================================
# Basic IO
# =========================================================
def load_image_rgb(path: Path, max_size: Optional[int] = None, force_vertical: bool = False) -> np.ndarray:
    """
    Read image as RGB.

    Important:
    - EXIF orientation is corrected.
    - The original file is never overwritten.
    - By default, landscape images are NOT rotated. This avoids changing the acquisition geometry.
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")

    if force_vertical:
        w, h = img.size
        if w > h:
            img = img.rotate(90, expand=True)

    if max_size is not None:
        w, h = img.size
        max_side = max(w, h)
        if max_side > max_size:
            scale = max_size / float(max_side)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = img.resize((new_w, new_h), Image.BILINEAR)

    return np.asarray(img)


def iter_images(input_roots: Sequence[Path]) -> Iterable[Path]:
    for root in input_roots:
        if not root.exists():
            print(f"[WARN] input_root not found, skipped: {root}")
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS:
                # Do not accidentally process preview or mask folders if nested.
                parts_lower = [x.lower() for x in p.parts]
                if "_preview" in parts_lower or "_init_mask" in parts_lower:
                    continue
                yield p


def save_mask_png(mask: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out = (ensure_binary(mask) * 255).astype(np.uint8)
    Image.fromarray(out).save(save_path)


# =========================================================
# Mask utilities
# =========================================================
def ensure_binary(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def kernel_ellipse(k: int) -> np.ndarray:
    k = int(k)
    if k <= 1:
        k = 1
    if k % 2 == 0:
        k += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask = ensure_binary(mask)
    h, w = mask.shape[:2]
    if mask.sum() == 0:
        return mask
    flood = (mask * 255).astype(np.uint8).copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    filled = (mask * 255) | flood_inv
    return (filled > 127).astype(np.uint8)


def remove_small_regions(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask = ensure_binary(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == i] = 1
    return out


def keep_top_components(mask: np.ndarray, max_components: int, min_area: int) -> np.ndarray:
    mask = ensure_binary(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            comps.append((i, area))
    comps = sorted(comps, key=lambda x: x[1], reverse=True)[:max_components]
    out = np.zeros_like(mask, dtype=np.uint8)
    for idx, _ in comps:
        out[labels == idx] = 1
    return out


def bbox_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    mask = ensure_binary(mask)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def bbox_area(box: Optional[np.ndarray]) -> float:
    if box is None:
        return 0.0
    x0, y0, x1, y1 = box
    return float(max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1))


def area_ratio(mask: np.ndarray, shape: Tuple[int, int]) -> float:
    h, w = shape[:2]
    return float(ensure_binary(mask).sum() / max(h * w, 1))


def bbox_fill_ratio(mask: np.ndarray) -> float:
    mask = ensure_binary(mask)
    box = bbox_from_mask(mask)
    if box is None:
        return 0.0
    return float(mask.sum() / max(bbox_area(box), 1.0))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = ensure_binary(a)
    b = ensure_binary(b)
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / union if union > 0 else 0.0


def solidity_score(mask: np.ndarray) -> float:
    mask = ensure_binary(mask)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    area_sum = 0.0
    hull_sum = 0.0
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area <= 1:
            continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        area_sum += area
        hull_sum += max(hull_area, 1.0)
    if hull_sum <= 1:
        return 0.0
    return float(np.clip(area_sum / hull_sum, 0.0, 1.0))


def edge_touch_ratio(mask: np.ndarray, edge_width: int = 18) -> float:
    mask = ensure_binary(mask)
    h, w = mask.shape[:2]
    edge = np.zeros_like(mask, dtype=np.uint8)
    ew = max(1, min(edge_width, h // 4, w // 4))
    edge[:ew, :] = 1
    edge[-ew:, :] = 1
    edge[:, :ew] = 1
    edge[:, -ew:] = 1
    return float((mask & edge).sum() / max(mask.sum(), 1))




def oriented_aspect_ratio(mask: np.ndarray) -> float:
    """
    Compute an orientation-invariant aspect ratio using the minimum-area rectangle.
    Leaf instances are often elongated. Large smooth background blobs tend to have
    weaker elongated structure, so this is used as a soft leafness cue.
    """
    mask = ensure_binary(mask)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0

    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) <= 2:
        return 0.0

    rect = cv2.minAreaRect(cnt)
    rw, rh = rect[1]
    if rw < 1 or rh < 1:
        return 0.0
    return float(max(rw, rh) / max(min(rw, rh), 1e-6))


def boundary_contrast(image_rgb: np.ndarray, mask: np.ndarray, ring: int = 5) -> float:
    """
    Estimate color contrast between the inner and outer boundary rings in LAB space.
    A real leaf boundary usually has a clearer inside/outside transition than a
    flat background region. This is a soft cue, not a hard segmentation rule.
    """
    mask = ensure_binary(mask)
    if mask.sum() == 0:
        return 0.0

    k = kernel_ellipse(max(3, ring))
    eroded = cv2.erode(mask, k, iterations=1)
    dilated = cv2.dilate(mask, k, iterations=1)

    inner_ring = ((mask > 0) & (eroded == 0)).astype(np.uint8)
    outer_ring = ((dilated > 0) & (mask == 0)).astype(np.uint8)

    if inner_ring.sum() < 10 or outer_ring.sum() < 10:
        return 0.0

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    inner_mean = lab[inner_ring > 0].mean(axis=0)
    outer_mean = lab[outer_ring > 0].mean(axis=0)
    return float(np.linalg.norm(inner_mean - outer_mean))


def propose_positive_points(image_rgb: np.ndarray, cfg: Dict) -> List[List[float]]:
    """
    Propose positive SAM2 point prompts from coarse green leaf-like regions.

    This is intentionally simple and conservative. It is not used as the final
    mask; it only provides seed points so SAM2 is less likely to choose a large
    smooth background region when the box prompt is ambiguous.
    """
    ccfg = cfg["CANDIDATE"]
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    green = (
        (h >= int(ccfg["seed_h_low"])) &
        (h <= int(ccfg["seed_h_high"])) &
        (s >= int(ccfg["seed_s_min"])) &
        (v >= int(ccfg["seed_v_min"])) &
        (v <= int(ccfg["seed_v_max"]))
    ).astype(np.uint8)

    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel_ellipse(7))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel_ellipse(5))
    green = remove_small_regions(green, int(ccfg["seed_min_area"]))
    green = ensure_binary(green)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(green, connectivity=8)
    comps = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(ccfg["seed_min_area"]):
            continue
        comp = (labels == i).astype(np.uint8)
        # Prefer components that are not dominated by image borders.
        edge = edge_touch_ratio(comp)
        fill = bbox_fill_ratio(comp)
        aspect = oriented_aspect_ratio(comp)
        score = area * (1.0 + 0.15 * min(aspect, 5.0)) * max(fill, 0.2) * (1.0 - 0.4 * min(edge, 1.0))
        comps.append((score, i, area))

    comps = sorted(comps, key=lambda x: x[0], reverse=True)[:int(ccfg["max_positive_points"])]

    points: List[List[float]] = []
    for _, idx, _ in comps:
        comp = (labels == idx).astype(np.uint8)
        dist = cv2.distanceTransform((comp * 255).astype(np.uint8), cv2.DIST_L2, 5)
        if dist.max() <= 0:
            ys, xs = np.where(comp > 0)
            if len(xs) == 0:
                continue
            x = float(xs.mean())
            y = float(ys.mean())
        else:
            y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
            x = float(x)
            y = float(y)
        points.append([x, y])

    return points


def build_point_prompt_sets(image_rgb: np.ndarray, cfg: Dict) -> List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    Build point prompt sets for SAM2. The first item is a box-only fallback.
    Additional items contain one positive point plus optional corner negatives.
    """
    ccfg = cfg["CANDIDATE"]
    prompt_sets: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = [(None, None)]
    if not ccfg.get("use_point_prompts", True):
        return prompt_sets

    pos_points = propose_positive_points(image_rgb, cfg)
    if not pos_points:
        return prompt_sets

    h, w = image_rgb.shape[:2]
    neg_points = np.empty((0, 2), dtype=np.float32)
    neg_labels = np.empty((0,), dtype=np.int32)
    if ccfg.get("use_corner_negative_points", True):
        m = int(ccfg.get("corner_negative_margin", 5))
        neg_points = np.array([
            [m, m],
            [w - 1 - m, m],
            [m, h - 1 - m],
            [w - 1 - m, h - 1 - m],
        ], dtype=np.float32)
        neg_labels = np.zeros((len(neg_points),), dtype=np.int32)

    for p in pos_points:
        pos = np.array([p], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        if len(neg_points) > 0:
            coords = np.concatenate([pos, neg_points], axis=0)
            point_labels = np.concatenate([labels, neg_labels], axis=0)
        else:
            coords = pos
            point_labels = labels
        prompt_sets.append((coords, point_labels))

    return prompt_sets


def final_mask_is_reasonable(mask: np.ndarray, image_rgb: np.ndarray, cfg: Dict) -> bool:
    """
    Soft final sanity check.

    v2 used a strict aspect/contrast check, which caused valid dark leaves to be rejected.
    Here we mainly reject: extremely tiny masks, very large masks, and masks dominated by image borders.
    """
    mcfg = cfg["MASK"]
    ccfg = cfg["CANDIDATE"]
    ar = area_ratio(mask, image_rgb.shape)
    edge = edge_touch_ratio(mask)
    fill = bbox_fill_ratio(mask)

    if ar < mcfg["min_final_ratio"] or ar > mcfg["max_final_ratio"]:
        return False
    if edge > max(0.58, float(ccfg["max_edge_touch"]) + 0.08):
        return False
    if fill < 0.045:
        return False
    return True

def close_open_refine(mask: np.ndarray, cfg: Dict) -> np.ndarray:
    mcfg = cfg["MASK"]
    mask = ensure_binary(mask)
    mask = remove_small_regions(mask, mcfg["min_region_area"])
    if mcfg["fill_holes"]:
        mask = fill_holes(mask)

    if mcfg["close_kernel"] > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_ellipse(mcfg["close_kernel"]))
        mask = ensure_binary(mask)
    if mcfg["open_kernel"] > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_ellipse(mcfg["open_kernel"]))
        mask = ensure_binary(mask)
    if mcfg["dilate_iterations"] > 0 and mcfg["dilate_kernel"] > 1:
        mask = cv2.dilate(mask, kernel_ellipse(mcfg["dilate_kernel"]), iterations=mcfg["dilate_iterations"])
        mask = ensure_binary(mask)

    if mcfg["fill_holes"]:
        mask = fill_holes(mask)
    mask = keep_top_components(mask, mcfg["max_components"], mcfg["min_region_area"])
    return ensure_binary(mask)


# =========================================================
# Candidate boxes
# =========================================================
def relative_box_to_abs(rel_box: Sequence[float], shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape[:2]
    x0r, y0r, x1r, y1r = rel_box
    x0 = int(np.clip(x0r * w, 0, w - 1))
    y0 = int(np.clip(y0r * h, 0, h - 1))
    x1 = int(np.clip(x1r * w, 0, w - 1))
    y1 = int(np.clip(y1r * h, 0, h - 1))
    if x1 <= x0:
        x1 = min(w - 1, x0 + 1)
    if y1 <= y0:
        y1 = min(h - 1, y0 + 1)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def expand_box(box: np.ndarray, shape: Tuple[int, int], sx: float, sy: float) -> np.ndarray:
    h, w = shape[:2]
    x0, y0, x1, y1 = box.astype(float)
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    x0 -= sx * bw
    x1 += sx * bw
    y0 -= sy * bh
    y1 += sy * bh
    return np.array([
        int(np.clip(x0, 0, w - 1)),
        int(np.clip(y0, 0, h - 1)),
        int(np.clip(x1, 0, w - 1)),
        int(np.clip(y1, 0, h - 1)),
    ], dtype=np.float32)


def get_grabcut_mask(image_rgb: np.ndarray, rect: Sequence[float], iter_count: int) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    x0, y0, x1, y1 = rect
    x = max(0, int(x0))
    y = max(0, int(y0))
    rw = max(2, int(x1 - x0))
    rh = max(2, int(y1 - y0))
    if x + rw >= w:
        rw = max(2, w - x - 1)
    if y + rh >= h:
        rh = max(2, h - y - 1)

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image_rgb, mask, (x, y, rw, rh), bgd_model, fgd_model,
                    iterCount=iter_count, mode=cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.zeros((h, w), dtype=np.uint8)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    return fg


def propose_grabcut_boxes(image_rgb: np.ndarray, cfg: Dict) -> List[np.ndarray]:
    bcfg = cfg["BOX"]
    if not bcfg["use_grabcut_boxes"]:
        return []
    h, w = image_rgb.shape[:2]
    boxes = []
    for rel in bcfg["grabcut_rects"]:
        rect = relative_box_to_abs(rel, image_rgb.shape)
        coarse = get_grabcut_mask(image_rgb, rect, iter_count=bcfg["grabcut_iter_count"])
        coarse = remove_small_regions(coarse, bcfg["grabcut_min_component_area"])
        coarse = fill_holes(coarse)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(coarse, connectivity=8)
        comps = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < bcfg["grabcut_min_component_area"]:
                continue
            comp = np.zeros_like(coarse, dtype=np.uint8)
            comp[labels == i] = 1
            box = bbox_from_mask(comp)
            if box is None:
                continue
            ar = area / max(h * w, 1)
            fill = bbox_fill_ratio(comp)
            # Avoid obvious background blobs while still allowing leaf clusters.
            if ar > 0.78 or fill < 0.05:
                continue
            comps.append((area, box))

        comps = sorted(comps, key=lambda x: x[0], reverse=True)[:bcfg["grabcut_max_boxes"]]
        for _, box in comps:
            boxes.append(expand_box(box, image_rgb.shape, bcfg["grabcut_expand_x"], bcfg["grabcut_expand_y"]))
    return boxes


def unique_boxes(boxes: List[np.ndarray], shape: Tuple[int, int]) -> List[np.ndarray]:
    out = []
    seen = set()
    h, w = shape[:2]
    for box in boxes:
        x0, y0, x1, y1 = box.astype(int).tolist()
        key = (round(x0 / max(w, 1), 2), round(y0 / max(h, 1), 2),
               round(x1 / max(w, 1), 2), round(y1 / max(h, 1), 2))
        if key in seen:
            continue
        seen.add(key)
        out.append(box.astype(np.float32))
    return out


def build_candidate_boxes(image_rgb: np.ndarray, cfg: Dict) -> List[np.ndarray]:
    boxes = [relative_box_to_abs(rb, image_rgb.shape) for rb in cfg["BOX"]["relative_boxes"]]
    boxes.extend(propose_grabcut_boxes(image_rgb, cfg))
    return unique_boxes(boxes, image_rgb.shape)


# =========================================================
# SAM2
# =========================================================
def inject_sam2_path(sam2_root: str) -> None:
    sam2_root = str(sam2_root)
    if os.path.isdir(sam2_root) and sam2_root not in sys.path:
        sys.path.insert(0, sam2_root)


def build_predictor(model_cfg: str, checkpoint: str, device: str):
    from sam2.build_sam import build_sam2  # type: ignore
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

    sam2_model = build_sam2(config_file=model_cfg, ckpt_path=checkpoint, device=device)
    return SAM2ImagePredictor(sam2_model)


def score_candidate(mask: np.ndarray, sam_score: float, image_rgb: np.ndarray, cfg: Dict) -> Tuple[float, Dict[str, float]]:
    """
    Balanced foreground score.

    Important change from v2:
    - Aspect and boundary contrast are no longer hard selectors.
    - Area score is not allowed to dominate, otherwise smooth green background wins.
    - Very small fragments are penalized more strongly, avoiding the tiny-top-leaf fallback.
    """
    ccfg = cfg["CANDIDATE"]
    mask = ensure_binary(mask)

    ar = area_ratio(mask, image_rgb.shape)
    fill = bbox_fill_ratio(mask)
    solid = solidity_score(mask)
    edge = edge_touch_ratio(mask)
    aspect = oriented_aspect_ratio(mask)
    contrast = boundary_contrast(image_rgb, mask, ring=5)

    # Prefer moderate-size foregrounds, but still allow leaf clusters.
    if ar <= 0.28:
        area_pref = np.clip(ar / 0.28, 0.0, 1.0)
    else:
        area_pref = np.clip(1.0 - (ar - 0.28) / 0.34, 0.0, 1.0)

    aspect_score = np.clip((aspect - 1.0) / 2.2, 0.0, 1.0)
    contrast_score = np.clip(contrast / float(ccfg["soft_boundary_contrast"]), 0.0, 1.0)

    # A balanced score: neither area nor contrast alone should decide.
    score = (
        0.20 * area_pref +
        0.18 * fill +
        0.12 * solid +
        0.20 * float(sam_score) +
        0.13 * aspect_score +
        0.17 * contrast_score -
        0.22 * edge
    )

    # Penalties. These are softer than v2 except for tiny fragments.
    if ar < ccfg["min_area_ratio"]:
        score -= 1.80
    if ar < 0.025:
        score -= 0.80
    if ar > ccfg["max_area_ratio"]:
        score -= 1.15
    if fill < ccfg["min_bbox_fill"]:
        score -= 0.65
    if solid < ccfg["min_solidity"]:
        score -= 0.25
    if edge > ccfg["max_edge_touch"]:
        score -= 0.50
    if aspect < ccfg["min_oriented_aspect"]:
        score -= 0.25
    if contrast < ccfg["min_boundary_contrast"]:
        score -= 0.30

    metrics = {
        "area_ratio": float(ar),
        "fill": float(fill),
        "solidity": float(solid),
        "edge_touch": float(edge),
        "oriented_aspect": float(aspect),
        "boundary_contrast": float(contrast),
        "sam_score": float(sam_score),
    }
    return float(score), metrics

def predict_and_merge_foreground(predictor, image_rgb: np.ndarray, cfg: Dict) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[Dict]]:
    ccfg = cfg["CANDIDATE"]
    boxes = build_candidate_boxes(image_rgb, cfg)
    prompt_sets = build_point_prompt_sets(image_rgb, cfg)
    predictor.set_image(image_rgb)

    candidates = []
    for box_i, box in enumerate(boxes):
        for prompt_i, (point_coords, point_labels) in enumerate(prompt_sets):
            try:
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box[None, :],
                    multimask_output=cfg["SAM2"]["multimask_output"],
                )
            except Exception as e:
                print(f"[WARN] SAM2 predict failed on box={box_i}, prompt={prompt_i}: {e}")
                continue

            for mask, sam_score in zip(masks, scores):
                mask = ensure_binary(mask)
                if mask.sum() == 0:
                    continue
                mask = remove_small_regions(mask, cfg["MASK"]["min_region_area"])
                mask = fill_holes(mask)
                mask = ensure_binary(mask)
                if mask.sum() == 0:
                    continue

                score, metrics = score_candidate(mask, float(sam_score), image_rgb, cfg)
                candidates.append({
                    "box_index": box_i,
                    "prompt_index": prompt_i,
                    "box": box,
                    "mask": mask,
                    "score": score,
                    "metrics": metrics,
                })

    if not candidates:
        h, w = image_rgb.shape[:2]
        empty = np.zeros((h, w), dtype=np.uint8)
        return empty, empty, boxes, []

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    best_score = candidates[0]["score"]

    selected = []
    for cand in candidates:
        if len(selected) >= ccfg["max_candidates_to_merge"]:
            break
        keep_by_score = cand["score"] >= best_score - ccfg["score_keep_delta"]
        keep_by_abs = cand["score"] >= ccfg["min_score_to_keep"]
        if not (keep_by_score or keep_by_abs):
            continue

        # Avoid adding nearly duplicate masks from different boxes/prompts.
        duplicated = False
        for prev in selected:
            if mask_iou(cand["mask"], prev["mask"]) >= ccfg["nms_iou_threshold"]:
                duplicated = True
                break
        if duplicated:
            continue
        selected.append(cand)

    # If filtering was too strict, keep the best candidate.
    if not selected:
        selected = [candidates[0]]

    merged = np.zeros_like(candidates[0]["mask"], dtype=np.uint8)
    init_for_preview = np.zeros_like(merged, dtype=np.uint8)
    for cand in selected:
        merged = ensure_binary(merged | cand["mask"])
        init_for_preview = ensure_binary(init_for_preview | cand["mask"])

    refined = close_open_refine(merged, cfg)

    # Safety fallback: if merged result is unreasonable, do NOT collapse to a tiny fragment.
    # Try a smaller subset of the top candidates first, then use the best reasonable single mask.
    if not final_mask_is_reasonable(refined, image_rgb, cfg):
        rescued = False
        for k in [6, 4, 2, 1]:
            trial = np.zeros_like(candidates[0]["mask"], dtype=np.uint8)
            trial_selected = []
            for cand in candidates[:k]:
                m = cand["mask"]
                # avoid very small fragment rescue
                if area_ratio(m, image_rgb.shape) < cfg["MASK"]["min_final_ratio"]:
                    continue
                trial = ensure_binary(trial | m)
                trial_selected.append(cand)
            trial = close_open_refine(trial, cfg)
            if trial_selected and final_mask_is_reasonable(trial, image_rgb, cfg):
                refined = trial
                init_for_preview = trial.copy()
                selected = trial_selected
                rescued = True
                break

        if not rescued:
            # Last fallback: choose the largest reasonable candidate among top-ranked masks,
            # rather than the tiny highest-score fragment.
            reasonable = []
            for cand in candidates[:30]:
                m = close_open_refine(cand["mask"], cfg)
                if final_mask_is_reasonable(m, image_rgb, cfg):
                    reasonable.append((area_ratio(m, image_rgb.shape), cand, m))
            if reasonable:
                reasonable = sorted(reasonable, key=lambda x: x[0], reverse=True)
                _, cand, m = reasonable[0]
                refined = m
                init_for_preview = ensure_binary(cand["mask"])
                selected = [cand]
            else:
                # keep the original refined result rather than returning empty; preview will expose it.
                pass

    return refined, init_for_preview, boxes, selected


# =========================================================
# Preview
# =========================================================
def draw_contours(canvas: np.ndarray, mask: np.ndarray, color: Sequence[int], thickness: int) -> None:
    cnts, _ = cv2.findContours(ensure_binary(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, cnts, -1, tuple(int(c) for c in color), thickness)


def make_preview(image_rgb: np.ndarray, init_mask: np.ndarray, final_mask: np.ndarray, boxes: List[np.ndarray], selected: List[Dict], cfg: Dict) -> np.ndarray:
    pcfg = cfg["PREVIEW"]
    canvas = image_rgb.copy()
    fill_color = np.array(pcfg["fill_color"], dtype=np.uint8)
    alpha = float(pcfg["fill_alpha"])

    canvas[final_mask > 0] = ((1 - alpha) * canvas[final_mask > 0] + alpha * fill_color).astype(np.uint8)
    draw_contours(canvas, init_mask, pcfg["candidate_contour_color"], pcfg["contour_thickness"])
    draw_contours(canvas, final_mask, pcfg["final_contour_color"], pcfg["contour_thickness"])

    # Draw only a limited number of boxes to avoid clutter.
    for b in boxes[:12]:
        x0, y0, x1, y1 = b.astype(int)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), tuple(pcfg["box_color"]), pcfg["box_thickness"])

    # Add small text summary.
    txt = f"merged={len(selected)} ratio={area_ratio(final_mask, image_rgb.shape):.3f}"
    cv2.putText(canvas, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


# =========================================================
# CLI / Main
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate foreground masks using SAM2 multi-box multi-mask merge.")
    parser.add_argument("--input_root", type=str, nargs="+", default=["./data/unlabeled_train/images"], help="One or more input image folders.")
    parser.add_argument("--output_root", type=str, default="./data/unlabeled_train/foreground_masks_v3", help="Output mask folder.")
    parser.add_argument("--sam2_root", type=str, default=r"E:\sam2", help="Local SAM2 repository root.")
    parser.add_argument("--model_cfg", type=str, default=None, help="SAM2 model config path.")
    parser.add_argument("--checkpoint", type=str, default=None, help="SAM2 checkpoint path.")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu. Default: auto.")
    parser.add_argument("--max_size", type=int, default=1024, help="Maximum image side for mask generation.")
    parser.add_argument("--sample_size", type=int, default=None, help="Randomly sample N images for mask generation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing masks.")
    parser.add_argument("--no_preview", action="store_true", help="Do not save preview images.")
    parser.add_argument("--save_init_mask", action="store_true", help="Save merged candidate mask before final refinement.")
    parser.add_argument("--force_vertical", action="store_true", help="Rotate landscape images to vertical in memory only. Not recommended by default.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DEFAULT_CONFIG.copy()
    cfg["RUNTIME"] = DEFAULT_CONFIG["RUNTIME"].copy()
    cfg["RUNTIME"]["max_size"] = args.max_size
    cfg["RUNTIME"]["overwrite"] = bool(args.overwrite)
    cfg["RUNTIME"]["save_preview"] = not bool(args.no_preview)
    cfg["RUNTIME"]["save_init_mask"] = bool(args.save_init_mask)
    if args.device is not None:
        cfg["RUNTIME"]["device"] = args.device

    sam2_root = args.sam2_root
    inject_sam2_path(sam2_root)

    model_cfg = args.model_cfg or os.path.join(sam2_root, r"sam2\configs\sam2.1\sam2.1_hiera_s.yaml")
    checkpoint = args.checkpoint or os.path.join(sam2_root, r"checkpoints\sam2.1_hiera_small.pt")

    input_roots = [Path(p) for p in args.input_root]
    output_root = Path(args.output_root)
    preview_root = output_root / "_preview"
    init_root = output_root / "_init_mask"

    output_root.mkdir(parents=True, exist_ok=True)
    if cfg["RUNTIME"]["save_preview"]:
        preview_root.mkdir(parents=True, exist_ok=True)
    if cfg["RUNTIME"]["save_init_mask"]:
        init_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Foreground SAM2 mask generation - balanced v3")
    print("input_root :", input_roots)
    print("output_root:", output_root)
    print("sam2_root  :", sam2_root)
    print("model_cfg  :", model_cfg)
    print("checkpoint :", checkpoint)
    print("device     :", cfg["RUNTIME"]["device"])
    print("max_size   :", cfg["RUNTIME"]["max_size"])
    print("overwrite  :", cfg["RUNTIME"]["overwrite"])
    print("preview    :", cfg["RUNTIME"]["save_preview"])
    print("=" * 80)

    predictor = build_predictor(model_cfg, checkpoint, cfg["RUNTIME"]["device"])
    image_paths = list(iter_images(input_roots))
    print(f"Found {len(image_paths)} images.")

    if args.seed is not None:
        random.seed(args.seed)

    if args.sample_size is not None and args.sample_size > 0:
        if args.sample_size < len(image_paths):
            image_paths = random.sample(image_paths, args.sample_size)
            print(f"Sample {len(image_paths)} images.")
        else:
            print(f"[WARN] sample_size={args.sample_size} >= total images={len(image_paths)}, use all images.")

    use_autocast = (
        cfg["RUNTIME"]["device"] == "cuda" and cfg["RUNTIME"]["use_autocast_when_cuda"]
    )

    for img_path in tqdm(image_paths, desc="Generating foreground masks", ncols=140):
        out_mask_path = output_root / f"{img_path.stem}.png"
        if out_mask_path.exists() and not cfg["RUNTIME"]["overwrite"]:
            continue

        image_rgb = load_image_rgb(
            img_path,
            max_size=cfg["RUNTIME"]["max_size"],
            force_vertical=bool(args.force_vertical),
        )

        with torch.inference_mode():
            if use_autocast:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    final_mask, init_mask, boxes, selected = predict_and_merge_foreground(predictor, image_rgb, cfg)
            else:
                final_mask, init_mask, boxes, selected = predict_and_merge_foreground(predictor, image_rgb, cfg)

        final_mask = ensure_binary(final_mask)
        save_mask_png(final_mask, out_mask_path)

        if cfg["RUNTIME"]["save_init_mask"]:
            save_mask_png(init_mask, init_root / f"{img_path.stem}.png")

        if cfg["RUNTIME"]["save_preview"]:
            preview = make_preview(image_rgb, init_mask, final_mask, boxes, selected, cfg)
            Image.fromarray(preview).save(preview_root / f"{img_path.stem}.jpg")

    print("Done. Foreground masks saved to:", output_root)


if __name__ == "__main__":
    main()
