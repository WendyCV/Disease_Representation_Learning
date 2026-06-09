import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ============================================================
# Basic utils
# ============================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_name(name: str) -> str:
    name = re.sub(r"[^\w\-.]+", "_", str(name))
    return name[:180]


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else float("nan")


def write_csv(path: Path, rows: List[dict]):
    ensure_dir(path.parent)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base: Path, p: str) -> Path:
    pp = Path(str(p))
    if pp.is_absolute():
        return pp
    return base / pp


# ============================================================
# Dataset parsing
# ============================================================

def parse_dataset_yaml(data_yaml: Path):
    data = load_yaml(data_yaml)

    if "path" not in data:
        raise ValueError("dataset yaml must contain `path`.")

    dataset_root = Path(data["path"])
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()

    names = data.get("names")
    if names is None:
        raise ValueError("dataset yaml must contain `names`.")

    if isinstance(names, dict):
        class_names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    elif isinstance(names, list):
        class_names = names
    else:
        raise ValueError("Unsupported names format in dataset yaml.")

    return data, dataset_root, class_names


def get_split_dirs(data: dict, dataset_root: Path, split: str):
    if split not in data or data[split] is None:
        raise ValueError(f"dataset yaml must contain `{split}` path.")

    img_dir = resolve_path(dataset_root, data[split])

    parts = list(img_dir.parts)
    if "images" not in parts:
        raise ValueError(f"Cannot infer label dir from image dir: {img_dir}")

    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    label_dir = Path(*parts)

    return img_dir, label_dir


def collect_images(img_dir: Path) -> List[Path]:
    images = []
    for p in img_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            images.append(p)
    return sorted(images)


def find_label_for_image(img_path: Path, img_root: Path, label_root: Path) -> Path:
    rel = img_path.relative_to(img_root).with_suffix(".txt")
    return label_root / rel


# ============================================================
# Geometry
# ============================================================

def yolo_to_xyxy(xc, yc, bw, bh, img_w, img_h):
    x1 = (xc - bw / 2.0) * img_w
    y1 = (yc - bh / 2.0) * img_h
    x2 = (xc + bw / 2.0) * img_w
    y2 = (yc + bh / 2.0) * img_h

    x1 = max(0.0, min(float(img_w), x1))
    y1 = max(0.0, min(float(img_h), y1))
    x2 = max(0.0, min(float(img_w), x2))
    y2 = max(0.0, min(float(img_h), y2))

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area2 = (
        np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    )

    union = area1 + area2 - inter
    return inter / np.maximum(union, 1e-9)


def box_area(box: np.ndarray):
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def box_wh(box: np.ndarray):
    return float(box[2] - box[0]), float(box[3] - box[1])


def area_bin(area_ratio: float):
    if area_ratio < 0.003:
        return "tiny"
    if area_ratio < 0.02:
        return "small"
    if area_ratio < 0.05:
        return "medium_small"
    if area_ratio < 0.15:
        return "medium"
    return "large"


def touches_edge(box: np.ndarray, img_w: int, img_h: int, edge_px: int):
    return (
        box[0] <= edge_px
        or box[1] <= edge_px
        or box[2] >= img_w - edge_px
        or box[3] >= img_h - edge_px
    )


def edge_distance_px(box: np.ndarray, img_w: int, img_h: int):
    return float(min(box[0], box[1], img_w - box[2], img_h - box[3]))


def clip_box_to_int(box: np.ndarray, img_w: int, img_h: int):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]

    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w, x2))
    y2 = max(0, min(img_h, y2))

    if x2 <= x1:
        x2 = min(img_w, x1 + 1)
    if y2 <= y1:
        y2 = min(img_h, y1 + 1)

    return x1, y1, x2, y2


# ============================================================
# Label reading
# ============================================================

def read_yolo_label(label_path: Path, img_w: int, img_h: int) -> Tuple[List[dict], List[dict]]:
    """
    Return:
      valid_rows, invalid_rows
    """
    if not label_path.exists():
        return [], []

    valid = []
    invalid = []

    lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    for line_id, line in enumerate(lines, start=1):
        raw = line.strip()

        if not raw:
            continue

        parts = raw.split()

        if len(parts) != 5:
            invalid.append(
                {
                    "line_id": line_id,
                    "raw_line": raw,
                    "reason": "not_5_columns",
                }
            )
            continue

        try:
            cls = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:])
        except Exception:
            invalid.append(
                {
                    "line_id": line_id,
                    "raw_line": raw,
                    "reason": "parse_error",
                }
            )
            continue

        if bw <= 0 or bh <= 0:
            invalid.append(
                {
                    "line_id": line_id,
                    "raw_line": raw,
                    "reason": "non_positive_width_or_height",
                }
            )
            continue

        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            invalid.append(
                {
                    "line_id": line_id,
                    "raw_line": raw,
                    "reason": "normalized_value_out_of_range",
                }
            )
            continue

        box = yolo_to_xyxy(xc, yc, bw, bh, img_w, img_h)

        if box[2] <= box[0] or box[3] <= box[1]:
            invalid.append(
                {
                    "line_id": line_id,
                    "raw_line": raw,
                    "reason": "invalid_xyxy_after_conversion",
                }
            )
            continue

        area_ratio = box_area(box) / float(img_w * img_h)
        w_px, h_px = box_wh(box)

        valid.append(
            {
                "line_id": line_id,
                "cls": cls,
                "xc": xc,
                "yc": yc,
                "bw": bw,
                "bh": bh,
                "xyxy": box,
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
                "w_px": w_px,
                "h_px": h_px,
                "aspect_w_h": safe_div(w_px, h_px),
                "area_ratio": float(area_ratio),
                "area_bin": area_bin(area_ratio),
                "raw_line": raw,
            }
        )

    return valid, invalid


# ============================================================
# Drawing
# ============================================================

CLASS_COLORS = {
    0: (0, 255, 255),   # algal_leaf_spot
    1: (0, 255, 0),     # no_disease
    2: (255, 0, 0),     # leaf_blight
    3: (0, 128, 255),   # leaf_spot
}


def draw_box(img, box, color, text, thickness=2):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = clip_box_to_int(box, w, h)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        img,
        text,
        (x1, max(18, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def save_crop(img, box, out_path: Path, text: str):
    h, w = img.shape[:2]

    bw = box[2] - box[0]
    bh = box[3] - box[1]

    pad_x = max(20, bw * 1.0)
    pad_y = max(20, bh * 1.0)

    crop_box = np.array(
        [
            max(0, box[0] - pad_x),
            max(0, box[1] - pad_y),
            min(w, box[2] + pad_x),
            min(h, box[3] + pad_y),
        ],
        dtype=np.float32,
    )

    cx1, cy1, cx2, cy2 = clip_box_to_int(crop_box, w, h)
    crop = img[cy1:cy2, cx1:cx2].copy()

    local_box = np.array(
        [
            box[0] - cx1,
            box[1] - cy1,
            box[2] - cx1,
            box[3] - cy1,
        ],
        dtype=np.float32,
    )

    draw_box(crop, local_box, (0, 0, 255), text, 2)

    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), crop)


def save_candidate_visual(img, labels, candidates, class_names, out_path: Path):
    canvas = img.copy()

    candidate_keys = {
        (c["line_id"], c["class_name"], c["issue_type"])
        for c in candidates
        if "line_id" in c
    }

    for r in labels:
        cls_id = r["cls"]
        cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"cls_{cls_id}"
        color = CLASS_COLORS.get(cls_id, (180, 180, 180))
        thickness = 2

        is_candidate = any(
            r["line_id"] == c.get("line_id")
            for c in candidates
        )

        if is_candidate:
            thickness = 4
            color = (0, 0, 255)

        draw_box(
            canvas,
            r["xyxy"],
            color,
            f"{r['line_id']}:{cls_name}:{r['area_bin']}",
            thickness,
        )

    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), canvas)


# ============================================================
# Split profiling
# ============================================================

def summarize_split(split, img_dir, label_dir, class_names, args):
    images = collect_images(img_dir)

    class_rows = []
    box_rows = []
    image_rows = []
    invalid_rows = []

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            image_rows.append(
                {
                    "split": split,
                    "image": str(img_path),
                    "status": "image_read_failed",
                    "num_boxes": 0,
                }
            )
            continue

        img_h, img_w = img.shape[:2]
        label_path = find_label_for_image(img_path, img_dir, label_dir)

        labels, invalids = read_yolo_label(label_path, img_w, img_h)

        for inv in invalids:
            inv_row = dict(inv)
            inv_row.update(
                {
                    "split": split,
                    "image": str(img_path),
                    "label": str(label_path),
                }
            )
            invalid_rows.append(inv_row)

        class_count = {name: 0 for name in class_names}

        for r in labels:
            cls_id = r["cls"]
            cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"cls_{cls_id}"

            if cls_name in class_count:
                class_count[cls_name] += 1

            box_rows.append(
                {
                    "split": split,
                    "image": str(img_path),
                    "label": str(label_path),
                    "line_id": r["line_id"],
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "area_ratio": r["area_ratio"],
                    "area_bin": r["area_bin"],
                    "w_px": r["w_px"],
                    "h_px": r["h_px"],
                    "aspect_w_h": r["aspect_w_h"],
                    "touch_edge": int(touches_edge(r["xyxy"], img_w, img_h, args.edge_px)),
                    "edge_distance_px": edge_distance_px(r["xyxy"], img_w, img_h),
                    "x1": r["x1"],
                    "y1": r["y1"],
                    "x2": r["x2"],
                    "y2": r["y2"],
                }
            )

        disease_classes = [
            name for name in class_count.keys()
            if name != args.no_disease_class
        ]
        disease_box_count = sum(class_count[c] for c in disease_classes)

        image_rows.append(
            {
                "split": split,
                "image": str(img_path),
                "label": str(label_path),
                "status": "ok",
                "img_w": img_w,
                "img_h": img_h,
                "num_boxes": len(labels),
                "num_classes_present": sum(1 for v in class_count.values() if v > 0),
                "disease_box_count": disease_box_count,
                "has_no_disease": int(class_count.get(args.no_disease_class, 0) > 0),
                "has_disease": int(disease_box_count > 0),
                "has_no_disease_and_disease": int(
                    class_count.get(args.no_disease_class, 0) > 0 and disease_box_count > 0
                ),
                **{f"count_{k}": v for k, v in class_count.items()},
            }
        )

    return image_rows, box_rows, invalid_rows


def aggregate_profile(box_rows, image_rows):
    profile_rows = []

    # class + area distribution
    groups = {}
    for r in box_rows:
        key = (r["split"], r["class_name"], r["area_bin"])
        groups.setdefault(
            key,
            {
                "split": r["split"],
                "class_name": r["class_name"],
                "area_bin": r["area_bin"],
                "box_count": 0,
                "touch_edge_count": 0,
                "area_ratio_sum": 0.0,
            },
        )
        g = groups[key]
        g["box_count"] += 1
        g["touch_edge_count"] += int(r["touch_edge"])
        g["area_ratio_sum"] += float(r["area_ratio"])

    for g in groups.values():
        g["touch_edge_ratio"] = safe_div(g["touch_edge_count"], g["box_count"])
        g["mean_area_ratio"] = safe_div(g["area_ratio_sum"], g["box_count"])
        del g["area_ratio_sum"]
        profile_rows.append(g)

    image_summary = {}
    for r in image_rows:
        key = r["split"]
        image_summary.setdefault(
            key,
            {
                "split": r["split"],
                "image_count": 0,
                "total_boxes": 0,
                "multi_class_image_count": 0,
                "no_disease_and_disease_count": 0,
                "empty_label_image_count": 0,
            },
        )
        s = image_summary[key]
        s["image_count"] += 1
        s["total_boxes"] += int(r.get("num_boxes", 0))
        if int(r.get("num_classes_present", 0)) > 1:
            s["multi_class_image_count"] += 1
        if int(r.get("has_no_disease_and_disease", 0)) == 1:
            s["no_disease_and_disease_count"] += 1
        if int(r.get("num_boxes", 0)) == 0:
            s["empty_label_image_count"] += 1

    for s in image_summary.values():
        s["mean_boxes_per_image"] = safe_div(s["total_boxes"], s["image_count"])
        s["multi_class_image_ratio"] = safe_div(s["multi_class_image_count"], s["image_count"])
        s["no_disease_and_disease_ratio"] = safe_div(s["no_disease_and_disease_count"], s["image_count"])
        s["empty_label_image_ratio"] = safe_div(s["empty_label_image_count"], s["image_count"])

    return profile_rows, list(image_summary.values())


# ============================================================
# Candidate detection
# ============================================================

def add_candidate(candidates, split, img_path, label_path, row, class_names, issue_type, priority, reason, suggestion):
    cls_id = row["cls"]
    cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"cls_{cls_id}"

    candidates.append(
        {
            "split": split,
            "priority": priority,
            "issue_type": issue_type,
            "image": str(img_path),
            "label": str(label_path),
            "line_id": row["line_id"],
            "class_id": cls_id,
            "class_name": cls_name,
            "area_ratio": row["area_ratio"],
            "area_bin": row["area_bin"],
            "w_px": row["w_px"],
            "h_px": row["h_px"],
            "aspect_w_h": row["aspect_w_h"],
            "x1": row["x1"],
            "y1": row["y1"],
            "x2": row["x2"],
            "y2": row["y2"],
            "reason": reason,
            "suggestion": suggestion,
            "manual_action": "",
            "manual_note": "",
        }
    )


def audit_test_labels(data, dataset_root, class_names, args, out_dir):
    split = args.target_split
    img_dir, label_dir = get_split_dirs(data, dataset_root, split)
    images = collect_images(img_dir)

    candidates = []
    invalid_rows = []
    duplicate_rows = []
    overlap_rows = []
    image_issue_rows = []

    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    focus_classes = set(args.focus_classes.split(","))

    for idx, img_path in enumerate(images, start=1):
        print(f"[AUDIT] {idx}/{len(images)} {img_path.name}")

        img = cv2.imread(str(img_path))
        if img is None:
            image_issue_rows.append(
                {
                    "split": split,
                    "image": str(img_path),
                    "issue_type": "image_read_failed",
                    "suggestion": "Check whether image file is corrupted.",
                }
            )
            continue

        img_h, img_w = img.shape[:2]
        label_path = find_label_for_image(img_path, img_dir, label_dir)
        labels, invalids = read_yolo_label(label_path, img_w, img_h)

        for inv in invalids:
            invalid_rows.append(
                {
                    "split": split,
                    "image": str(img_path),
                    "label": str(label_path),
                    **inv,
                }
            )

        if not labels:
            image_issue_rows.append(
                {
                    "split": split,
                    "image": str(img_path),
                    "label": str(label_path),
                    "issue_type": "empty_label",
                    "suggestion": "Check whether this is a true empty-label image. If image contains leaf/disease, add labels.",
                }
            )
            continue

        class_counts = {}
        for r in labels:
            cls_name = class_names[r["cls"]] if 0 <= r["cls"] < len(class_names) else f"cls_{r['cls']}"
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        disease_count = sum(v for k, v in class_counts.items() if k != args.no_disease_class)

        if class_counts.get(args.no_disease_class, 0) > 0 and disease_count > 0:
            image_issue_rows.append(
                {
                    "split": split,
                    "image": str(img_path),
                    "label": str(label_path),
                    "issue_type": "no_disease_coexists_with_disease",
                    "suggestion": "no_disease should normally be used for whole healthy leaf only. If disease labels exist, check whether no_disease box should be removed.",
                }
            )

        # Per-box candidate rules
        for r in labels:
            cls_id = r["cls"]
            cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"cls_{cls_id}"
            touch_edge = touches_edge(r["xyxy"], img_w, img_h, args.edge_px)

            if cls_name not in focus_classes:
                continue

            # leaf_blight audit rules
            if cls_name == "leaf_blight":
                if r["area_bin"] == "tiny":
                    add_candidate(
                        candidates,
                        split,
                        img_path,
                        label_path,
                        r,
                        class_names,
                        "leaf_blight_tiny_box",
                        "P1",
                        "leaf_blight box is tiny.",
                        "Keep only if it is a clear necrotic/dry spot. Delete if it is reflection, shadow, leaf_spot transition, algal edge, or blurry dot.",
                    )

                elif r["area_bin"] == "small":
                    add_candidate(
                        candidates,
                        split,
                        img_path,
                        label_path,
                        r,
                        class_names,
                        "leaf_blight_small_box",
                        "P2",
                        "leaf_blight box is small.",
                        "Keep clear small necrotic/dry lesions. Delete ambiguous dark transition or reflection.",
                    )

                if touch_edge:
                    add_candidate(
                        candidates,
                        split,
                        img_path,
                        label_path,
                        r,
                        class_names,
                        "leaf_blight_touch_edge",
                        "P2",
                        "leaf_blight box touches image edge.",
                        "Check whether this is a true truncated lesion. If it is leaf edge shadow or weak dark tip, delete.",
                    )

                if r["aspect_w_h"] > args.aspect_high or r["aspect_w_h"] < args.aspect_low:
                    add_candidate(
                        candidates,
                        split,
                        img_path,
                        label_path,
                        r,
                        class_names,
                        "leaf_blight_unusual_aspect",
                        "P3",
                        "leaf_blight box has unusual aspect ratio.",
                        "Check whether box is too long/thin. Shrink or split if it contains non-lesion area.",
                    )

            # algal_leaf_spot audit rules
            if cls_name == "algal_leaf_spot":
                if r["area_bin"] == "tiny":
                    add_candidate(
                        candidates,
                        split,
                        img_path,
                        label_path,
                        r,
                        class_names,
                        "algal_tiny_box",
                        "P1",
                        "algal_leaf_spot box is tiny.",
                        "Keep only if it is a clear algal lesion. Delete tiny noise or uncertain point.",
                    )

                if r["area_bin"] == "large":
                    add_candidate(
                        candidates,
                        split,
                        img_path,
                        label_path,
                        r,
                        class_names,
                        "algal_large_box",
                        "P2",
                        "algal_leaf_spot box is large.",
                        "Check whether it covers lesion cluster only. Do not box the whole leaf or large healthy background.",
                    )

                if touch_edge:
                    add_candidate(
                        candidates,
                        split,
                        img_path,
                        label_path,
                        r,
                        class_names,
                        "algal_touch_edge",
                        "P3",
                        "algal_leaf_spot box touches image edge.",
                        "Check whether it is true lesion at edge, not leaf boundary or background.",
                    )

        # Same-class duplicate / overlap check
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                iou = float(box_iou_one_to_many(a["xyxy"], np.array([b["xyxy"]], dtype=np.float32))[0])

                a_name = class_names[a["cls"]] if 0 <= a["cls"] < len(class_names) else f"cls_{a['cls']}"
                b_name = class_names[b["cls"]] if 0 <= b["cls"] < len(class_names) else f"cls_{b['cls']}"

                if a["cls"] == b["cls"] and iou >= args.same_class_overlap_iou:
                    duplicate_rows.append(
                        {
                            "split": split,
                            "image": str(img_path),
                            "label": str(label_path),
                            "class_name": a_name,
                            "line_id_a": a["line_id"],
                            "line_id_b": b["line_id"],
                            "iou": iou,
                            "issue_type": "same_class_high_overlap",
                            "suggestion": "Check duplicate boxes. Merge or delete one if they describe the same lesion.",
                        }
                    )

                if a["cls"] != b["cls"] and iou >= args.cross_class_overlap_iou:
                    overlap_rows.append(
                        {
                            "split": split,
                            "image": str(img_path),
                            "label": str(label_path),
                            "class_a": a_name,
                            "class_b": b_name,
                            "line_id_a": a["line_id"],
                            "line_id_b": b["line_id"],
                            "iou": iou,
                            "issue_type": "cross_class_high_overlap",
                            "suggestion": "Check class confusion. If symptoms overlap, split boxes more carefully or relabel wrong class.",
                        }
                    )

        # Save visuals only for images with candidate issues
        image_candidates = [
            c for c in candidates
            if c["image"] == str(img_path)
        ]

        if image_candidates:
            save_candidate_visual(
                img=img,
                labels=labels,
                candidates=image_candidates,
                class_names=class_names,
                out_path=out_dir / "visual_review" / f"{safe_name(img_path.name)}",
            )

            for c in image_candidates:
                box = np.array([c["x1"], c["y1"], c["x2"], c["y2"]], dtype=np.float32)
                crop_name = f"{safe_name(Path(img_path).stem)}_line{c['line_id']}_{c['class_name']}_{c['issue_type']}.jpg"
                save_crop(
                    img=img,
                    box=box,
                    out_path=out_dir / "crop_review" / c["class_name"] / c["issue_type"] / crop_name,
                    text=f"{c['line_id']}:{c['class_name']}:{c['issue_type']}",
                )

    return candidates, invalid_rows, duplicate_rows, overlap_rows, image_issue_rows


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="./data/det_dataset.yaml", help="YOLO dataset yaml")
    parser.add_argument("--out", default="./runs_audit", help="output directory")

    parser.add_argument("--target-split", default="test", help="split to audit")
    parser.add_argument("--compare-splits", default="val,test", help="splits for profile comparison")

    parser.add_argument("--focus-classes", default="leaf_blight,algal_leaf_spot")
    parser.add_argument("--no-disease-class", default="no_disease")

    parser.add_argument("--edge-px", type=int, default=6)
    parser.add_argument("--same-class-overlap-iou", type=float, default=0.80)
    parser.add_argument("--cross-class-overlap-iou", type=float, default=0.30)

    parser.add_argument("--aspect-low", type=float, default=0.20)
    parser.add_argument("--aspect-high", type=float, default=5.00)

    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    data, dataset_root, class_names = parse_dataset_yaml(data_yaml)

    out_dir = Path(args.out).resolve() / f"{args.target_split}_label_quality_audit"
    ensure_dir(out_dir)

    print("[DATA]", data_yaml)
    print("[ROOT]", dataset_root)
    print("[CLASSES]", class_names)
    print("[OUT]", out_dir)

    # ------------------------------------------------------------
    # 1) val/test profile
    # ------------------------------------------------------------
    all_image_rows = []
    all_box_rows = []
    all_invalid_rows = []

    compare_splits = [s.strip() for s in args.compare_splits.split(",") if s.strip()]

    for split in compare_splits:
        img_dir, label_dir = get_split_dirs(data, dataset_root, split)
        print(f"\n[PROFILE] split={split}")
        print("[IMG]", img_dir)
        print("[LAB]", label_dir)

        image_rows, box_rows, invalid_rows = summarize_split(
            split=split,
            img_dir=img_dir,
            label_dir=label_dir,
            class_names=class_names,
            args=args,
        )

        all_image_rows.extend(image_rows)
        all_box_rows.extend(box_rows)
        all_invalid_rows.extend(invalid_rows)

    profile_rows, image_summary_rows = aggregate_profile(all_box_rows, all_image_rows)

    write_csv(out_dir / "01_split_image_summary.csv", all_image_rows)
    write_csv(out_dir / "02_split_box_summary.csv", all_box_rows)
    write_csv(out_dir / "03_split_class_area_profile.csv", profile_rows)
    write_csv(out_dir / "04_split_image_level_profile.csv", image_summary_rows)
    write_csv(out_dir / "05_invalid_label_lines.csv", all_invalid_rows)

    # ------------------------------------------------------------
    # 2) test label audit
    # ------------------------------------------------------------
    print(f"\n[AUDIT TARGET SPLIT] {args.target_split}")

    candidates, invalid_rows, duplicate_rows, overlap_rows, image_issue_rows = audit_test_labels(
        data=data,
        dataset_root=dataset_root,
        class_names=class_names,
        args=args,
        out_dir=out_dir,
    )

    # Sort candidates by priority and class
    priority_order = {"P1": 0, "P2": 1, "P3": 2}
    candidates = sorted(
        candidates,
        key=lambda r: (
            priority_order.get(r["priority"], 9),
            r["class_name"],
            r["issue_type"],
            Path(r["image"]).name,
            r.get("line_id", 0),
        ),
    )

    write_csv(out_dir / "10_test_label_review_candidates.csv", candidates)
    write_csv(out_dir / "11_test_invalid_label_lines.csv", invalid_rows)
    write_csv(out_dir / "12_test_same_class_overlap_candidates.csv", duplicate_rows)
    write_csv(out_dir / "13_test_cross_class_overlap_candidates.csv", overlap_rows)
    write_csv(out_dir / "14_test_image_level_issues.csv", image_issue_rows)

    # ------------------------------------------------------------
    # 3) simple issue summary
    # ------------------------------------------------------------
    issue_summary = {}

    for r in candidates:
        key = (r["split"], r["priority"], r["class_name"], r["issue_type"])
        issue_summary.setdefault(
            key,
            {
                "split": r["split"],
                "priority": r["priority"],
                "class_name": r["class_name"],
                "issue_type": r["issue_type"],
                "count": 0,
            },
        )
        issue_summary[key]["count"] += 1

    write_csv(out_dir / "15_test_label_review_summary.csv", list(issue_summary.values()))

    manifest = {
        "data": str(data_yaml),
        "dataset_root": str(dataset_root),
        "class_names": class_names,
        "args": vars(args),
        "outputs": {
            "split_image_summary": str(out_dir / "01_split_image_summary.csv"),
            "split_box_summary": str(out_dir / "02_split_box_summary.csv"),
            "split_class_area_profile": str(out_dir / "03_split_class_area_profile.csv"),
            "split_image_level_profile": str(out_dir / "04_split_image_level_profile.csv"),
            "test_label_review_candidates": str(out_dir / "10_test_label_review_candidates.csv"),
            "visual_review": str(out_dir / "visual_review"),
            "crop_review": str(out_dir / "crop_review"),
        },
    }

    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    readme = f"""# Test label quality audit

This audit is designed for correcting the `{args.target_split}` split labels.

## Main files

1. Dataset profile:
- 01_split_image_summary.csv
- 02_split_box_summary.csv
- 03_split_class_area_profile.csv
- 04_split_image_level_profile.csv

2. Test label review:
- 10_test_label_review_candidates.csv
- 11_test_invalid_label_lines.csv
- 12_test_same_class_overlap_candidates.csv
- 13_test_cross_class_overlap_candidates.csv
- 14_test_image_level_issues.csv
- 15_test_label_review_summary.csv

3. Visual check:
- visual_review/
- crop_review/

## Recommended manual actions

For leaf_blight:
- keep: clear necrotic spot, clear dry edge, clear dry tip
- delete: reflection, shadow, slight dark tip, blurry tiny point, leaf_spot transition, algal edge
- shrink: if box includes too much healthy/yellowing area
- merge: if continuous necrotic region is split into multiple boxes
- split: if distant independent lesions are inside one large box

For algal_leaf_spot:
- keep: clear algal lesion or lesion cluster
- delete: tiny noise, uncertain point, background spot
- shrink: if box covers too much healthy leaf
- merge: if adjacent algal lesions form a coherent cluster
- split: if distant clusters are grouped into one box

For no_disease:
- no_disease should normally represent a whole healthy leaf.
- If disease labels also exist in the same image, check whether no_disease should be removed.
"""

    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print("\n[DONE]")
    print("[OUT]", out_dir)
    print("[KEY FILES]")
    print("  03_split_class_area_profile.csv")
    print("  04_split_image_level_profile.csv")
    print("  10_test_label_review_candidates.csv")
    print("  12_test_same_class_overlap_candidates.csv")
    print("  13_test_cross_class_overlap_candidates.csv")
    print("  visual_review/")
    print("  crop_review/")


if __name__ == "__main__":
    main()