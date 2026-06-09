import sys
import argparse
import csv
import json
import re
from pathlib import Path
from typing import List

import cv2
import numpy as np
import yaml

# ============================================================
# Fix project import path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import YOLO


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


def public_rows(rows: List[dict]) -> List[dict]:
    clean = []
    for r in rows:
        clean.append({k: v for k, v in r.items() if not str(k).startswith("_")})
    return clean


# ============================================================
# Dataset
# ============================================================

def parse_dataset_yaml(data_yaml: Path):
    data = load_yaml(data_yaml)

    names = data.get("names")
    if names is None:
        raise ValueError("dataset yaml must contain `names`.")

    if isinstance(names, dict):
        class_names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    elif isinstance(names, list):
        class_names = names
    else:
        raise ValueError("Unsupported names format in dataset yaml.")

    return data, class_names


def get_split_img_dir(data: dict, data_root: Path, split: str) -> Path:
    if split not in data or data[split] is None:
        raise ValueError(f"dataset yaml must contain `{split}` path.")

    split_path = Path(str(data[split]))

    if split_path.is_absolute():
        parts = list(split_path.parts)
        if "images" in parts:
            idx = len(parts) - 1 - parts[::-1].index("images")
            rel = Path(*parts[idx:])
            return data_root / rel
        return split_path

    return data_root / split_path


def infer_label_dir_from_img_dir(img_dir: Path) -> Path:
    parts = list(img_dir.parts)
    if "images" not in parts:
        raise ValueError(f"Cannot infer label dir from image dir: {img_dir}")

    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts)


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
# Label reading
# ============================================================

def read_yolo_label(label_path: Path, img_w: int, img_h: int) -> List[dict]:
    if not label_path.exists():
        return []

    rows = []
    lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    for line_id, line in enumerate(lines, start=1):
        raw = line.strip()
        if not raw:
            continue

        parts = raw.split()
        if len(parts) != 5:
            continue

        try:
            cls = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:])
        except Exception:
            continue

        x1 = (xc - bw / 2.0) * img_w
        y1 = (yc - bh / 2.0) * img_h
        x2 = (xc + bw / 2.0) * img_w
        y2 = (yc + bh / 2.0) * img_h

        x1 = max(0.0, min(float(img_w), x1))
        y1 = max(0.0, min(float(img_h), y1))
        x2 = max(0.0, min(float(img_w), x2))
        y2 = max(0.0, min(float(img_h), y2))

        if x2 <= x1 or y2 <= y1:
            continue

        area_ratio = ((x2 - x1) * (y2 - y1)) / float(img_w * img_h)

        rows.append(
            {
                "line_id": line_id,
                "cls": cls,
                "xyxy": np.array([x1, y1, x2, y2], dtype=np.float32),
                "area_ratio": float(area_ratio),
                "raw_line": raw,
            }
        )

    return rows


# ============================================================
# Geometry
# ============================================================

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


def max_iou_to_rows(box: np.ndarray, rows: List[dict]):
    if not rows:
        return 0.0, None

    boxes = np.array([r["xyxy"] for r in rows], dtype=np.float32)
    ious = box_iou_one_to_many(box, boxes)
    best_idx = int(np.argmax(ious))
    return float(ious[best_idx]), rows[best_idx]


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
# Drawing
# ============================================================

CLASS_COLORS = {
    0: (0, 255, 255),
    1: (0, 255, 0),
    2: (255, 0, 0),
    3: (0, 128, 255),
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


def _row_box(row: dict, prefix: str = "") -> np.ndarray:
    if prefix:
        return np.array(
            [
                float(row[f"{prefix}_x1"]),
                float(row[f"{prefix}_y1"]),
                float(row[f"{prefix}_x2"]),
                float(row[f"{prefix}_y2"]),
            ],
            dtype=np.float32,
        )

    return np.array(
        [
            float(row["x1"]),
            float(row["y1"]),
            float(row["x2"]),
            float(row["y2"]),
        ],
        dtype=np.float32,
    )


def save_label_diff_visual(
    img: np.ndarray,
    image_added: List[dict],
    image_removed: List[dict],
    image_modified: List[dict],
    out_path: Path,
):
    """
    Only draw different boxes.

    Color:
      red      = added new leaf_blight
      green    = removed old leaf_blight
      orange   = modified old leaf_blight boundary
      magenta  = modified new leaf_blight boundary
    """
    canvas = img.copy()

    for i, r in enumerate(image_added, start=1):
        box = _row_box(r)
        draw_box(
            canvas,
            box,
            (0, 0, 255),
            f"ADD#{i} {r.get('area_bin', '')}",
            3,
        )

    for i, r in enumerate(image_removed, start=1):
        box = _row_box(r)
        draw_box(
            canvas,
            box,
            (0, 200, 0),
            f"REMOVE#{i} {r.get('area_bin', '')}",
            3,
        )

    for i, r in enumerate(image_modified, start=1):
        old_box = _row_box(r, "old")
        new_box = _row_box(r, "new")

        draw_box(
            canvas,
            old_box,
            (0, 180, 255),
            f"MOD#{i} OLD",
            2,
        )

        draw_box(
            canvas,
            new_box,
            (255, 0, 255),
            f"MOD#{i} NEW",
            3,
        )

    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), canvas)


def save_crop_for_box(img, box, out_path: Path, text: str, color=(0, 0, 255)):
    h, w = img.shape[:2]

    bw = box[2] - box[0]
    bh = box[3] - box[1]

    pad_x = max(20, bw * 0.9)
    pad_y = max(20, bh * 0.9)

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

    local = np.array([box[0] - cx1, box[1] - cy1, box[2] - cx1, box[3] - cy1], dtype=np.float32)
    draw_box(crop, local, color, text, 2)

    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), crop)


# ============================================================
# Model loading / prediction
# ============================================================

def resolve_weight(weight_or_run: str) -> Path:
    p = Path(weight_or_run)

    if p.is_file():
        return p

    candidate = p / "weights" / "best.pt"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Cannot find weight. Use best.pt path or run dir containing weights/best.pt: {weight_or_run}"
    )


def predict_model(model: YOLO, image_paths: List[Path], args) -> dict:
    preds_by_img = {str(p): [] for p in image_paths}

    results_iter = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.nms_iou,
        max_det=args.max_det,
        device=args.device,
        verbose=False,
        stream=True,
    )

    results = list(results_iter)

    if len(results) != len(image_paths):
        print(f"[WARN] result count mismatch: results={len(results)}, images={len(image_paths)}")

    for img_path, result in zip(image_paths, results):
        rows = []

        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            cls = result.boxes.cls.cpu().numpy().astype(np.int64)
            conf = result.boxes.conf.cpu().numpy().astype(np.float32)

            for b, c, cf in zip(xyxy, cls, conf):
                rows.append({"cls": int(c), "conf": float(cf), "xyxy": b})

        preds_by_img[str(img_path)] = rows

    return preds_by_img


# ============================================================
# Prediction matching
# ============================================================

def best_pred_for_gt(gt_box: np.ndarray, preds: List[dict], gt_cls: int):
    best_same = {"iou": 0.0, "conf": float("nan"), "cls": None, "xyxy": None}
    best_any = {"iou": 0.0, "conf": float("nan"), "cls": None, "xyxy": None}
    best_wrong = {"iou": 0.0, "conf": float("nan"), "cls": None, "xyxy": None}

    if not preds:
        return best_same, best_any, best_wrong

    boxes = np.array([p["xyxy"] for p in preds], dtype=np.float32)
    ious = box_iou_one_to_many(gt_box, boxes)

    for i, p in enumerate(preds):
        iou = float(ious[i])

        if iou > best_any["iou"]:
            best_any = {"iou": iou, "conf": p["conf"], "cls": p["cls"], "xyxy": p["xyxy"]}

        if p["cls"] == gt_cls and iou > best_same["iou"]:
            best_same = {"iou": iou, "conf": p["conf"], "cls": p["cls"], "xyxy": p["xyxy"]}

        if p["cls"] != gt_cls and iou > best_wrong["iou"]:
            best_wrong = {"iou": iou, "conf": p["conf"], "cls": p["cls"], "xyxy": p["xyxy"]}

    return best_same, best_any, best_wrong


def best_gt_for_pred(pred_box: np.ndarray, gts: List[dict], pred_cls: int, class_names: List[str]):
    best_same = {
        "iou": 0.0,
        "gt_index": None,
        "gt_cls": None,
        "gt_cls_name": "",
        "gt_box": None,
        "gt_area_ratio": float("nan"),
        "gt_area_bin": "",
    }

    best_any = {
        "iou": 0.0,
        "gt_index": None,
        "gt_cls": None,
        "gt_cls_name": "",
        "gt_box": None,
        "gt_area_ratio": float("nan"),
        "gt_area_bin": "",
    }

    if not gts:
        return best_same, best_any

    gt_boxes = np.array([g["xyxy"] for g in gts], dtype=np.float32)
    ious = box_iou_one_to_many(pred_box, gt_boxes)

    for i, gt in enumerate(gts):
        iou = float(ious[i])
        gt_cls = gt["cls"]
        gt_cls_name = class_names[gt_cls] if 0 <= gt_cls < len(class_names) else f"cls_{gt_cls}"

        if iou > best_any["iou"]:
            best_any = {
                "iou": iou,
                "gt_index": i,
                "gt_cls": gt_cls,
                "gt_cls_name": gt_cls_name,
                "gt_box": gt["xyxy"],
                "gt_area_ratio": gt["area_ratio"],
                "gt_area_bin": area_bin(gt["area_ratio"]),
            }

        if gt_cls == pred_cls and iou > best_same["iou"]:
            best_same = {
                "iou": iou,
                "gt_index": i,
                "gt_cls": gt_cls,
                "gt_cls_name": gt_cls_name,
                "gt_box": gt["xyxy"],
                "gt_area_ratio": gt["area_ratio"],
                "gt_area_bin": area_bin(gt["area_ratio"]),
            }

    return best_same, best_any


def match_predictions_to_gts(
    model_tag: str,
    img_path: Path,
    label_path: Path,
    preds: List[dict],
    gts: List[dict],
    class_names: List[str],
    img_w: int,
    img_h: int,
    args,
):
    rows = []
    matched_gt_indices = set()

    sorted_preds = sorted(preds, key=lambda x: float(x["conf"]), reverse=True)

    for pred_index, pred in enumerate(sorted_preds):
        pred_cls = int(pred["cls"])
        pred_cls_name = class_names[pred_cls] if 0 <= pred_cls < len(class_names) else f"cls_{pred_cls}"
        pred_box = pred["xyxy"]
        pred_conf = float(pred["conf"])

        pred_w, pred_h = box_wh(pred_box)
        pred_area_ratio = (pred_w * pred_h) / float(img_w * img_h)
        pred_area_bin = area_bin(pred_area_ratio)

        best_same, best_any = best_gt_for_pred(pred_box, gts, pred_cls, class_names)

        prediction_status = "FP"
        fp_type = ""
        matched_gt_index = ""

        if best_same["iou"] >= args.match_iou:
            if best_same["gt_index"] not in matched_gt_indices:
                prediction_status = "TP"
                matched_gt_index = best_same["gt_index"]
                matched_gt_indices.add(best_same["gt_index"])
            else:
                fp_type = "FP_duplicate_same_class"
        else:
            if best_any["iou"] >= args.overlap_iou and best_any["gt_cls"] != pred_cls:
                fp_type = "FP_wrong_class_over_gt"
            elif best_same["iou"] >= args.overlap_iou:
                fp_type = "FP_same_class_low_iou"
            else:
                fp_type = "FP_background_or_missing_gt"

        rows.append(
            {
                "split": "",
                "model_tag": model_tag,
                "image": str(img_path),
                "label": str(label_path),
                "pred_index": pred_index,
                "pred_class_id": pred_cls,
                "pred_class_name": pred_cls_name,
                "pred_conf": pred_conf,
                "pred_area_ratio": pred_area_ratio,
                "pred_area_bin": pred_area_bin,
                "pred_w_px": pred_w,
                "pred_h_px": pred_h,
                "pred_aspect_w_h": safe_div(pred_w, pred_h),
                "pred_x1": float(pred_box[0]),
                "pred_y1": float(pred_box[1]),
                "pred_x2": float(pred_box[2]),
                "pred_y2": float(pred_box[3]),
                "prediction_status": prediction_status,
                "fp_type": fp_type,
                "matched_gt_index": matched_gt_index,
                "best_same_gt_iou": best_same["iou"],
                "best_same_gt_index": "" if best_same["gt_index"] is None else best_same["gt_index"],
                "best_same_gt_class_name": best_same["gt_cls_name"],
                "best_same_gt_area_ratio": best_same["gt_area_ratio"],
                "best_same_gt_area_bin": best_same["gt_area_bin"],
                "best_any_gt_iou": best_any["iou"],
                "best_any_gt_index": "" if best_any["gt_index"] is None else best_any["gt_index"],
                "best_any_gt_class_id": "" if best_any["gt_cls"] is None else best_any["gt_cls"],
                "best_any_gt_class_name": best_any["gt_cls_name"],
                "best_any_gt_area_ratio": best_any["gt_area_ratio"],
                "best_any_gt_area_bin": best_any["gt_area_bin"],
                "_xyxy": pred_box,
                "_best_gt_xyxy": best_any["gt_box"],
            }
        )

    return rows


def is_unique_fp(fp_row: dict, other_fp_rows: List[dict], args) -> bool:
    box = fp_row["_xyxy"]
    cls_id = fp_row["pred_class_id"]

    candidates = [r for r in other_fp_rows if r["pred_class_id"] == cls_id]

    if not candidates:
        return True

    boxes = np.array([r["_xyxy"] for r in candidates], dtype=np.float32)
    ious = box_iou_one_to_many(box, boxes)

    return float(np.max(ious)) < args.unique_fp_iou


# ============================================================
# Label diff
# ============================================================

def compare_leaf_blight_labels_for_split(
    split: str,
    data: dict,
    old_root: Path,
    new_root: Path,
    class_names: List[str],
    args,
    out_dir: Path,
):
    focus_cls = class_names.index(args.focus_class)

    old_img_dir = get_split_img_dir(data, old_root, split)
    new_img_dir = get_split_img_dir(data, new_root, split)

    old_label_dir = infer_label_dir_from_img_dir(old_img_dir)
    new_label_dir = infer_label_dir_from_img_dir(new_img_dir)

    new_images = collect_images(new_img_dir)

    diff_added = []
    diff_removed = []
    diff_modified = []
    changed_summary = []
    changed_image_paths = []

    print(f"\n[LABEL-DIFF] split={split}")
    print("[OLD IMG]", old_img_dir)
    print("[NEW IMG]", new_img_dir)
    print("[OLD LAB]", old_label_dir)
    print("[NEW LAB]", new_label_dir)
    print("[IMAGES]", len(new_images))

    for img_path in new_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]

        rel = img_path.relative_to(new_img_dir)
        old_img_path = old_img_dir / rel

        old_label_path = find_label_for_image(
            old_img_path if old_img_path.exists() else img_path,
            old_img_dir if old_img_path.exists() else new_img_dir,
            old_label_dir,
        )
        new_label_path = find_label_for_image(img_path, new_img_dir, new_label_dir)

        old_labels = read_yolo_label(old_label_path, img_w, img_h)
        new_labels = read_yolo_label(new_label_path, img_w, img_h)

        old_focus = [r for r in old_labels if r["cls"] == focus_cls]
        new_focus = [r for r in new_labels if r["cls"] == focus_cls]
        old_other = [r for r in old_labels if r["cls"] != focus_cls]
        new_other = [r for r in new_labels if r["cls"] != focus_cls]

        image_added = []
        image_removed = []
        image_modified = []

        # Added new leaf_blight
        for new_idx, nb in enumerate(new_focus):
            best_iou_old_focus, best_old_focus = max_iou_to_rows(nb["xyxy"], old_focus)
            best_iou_old_other, best_old_other = max_iou_to_rows(nb["xyxy"], old_other)

            if best_iou_old_focus < args.label_add_remove_iou:
                change_subtype = "added_new_leaf_blight"
                if best_iou_old_other >= args.label_class_change_iou:
                    old_cls = best_old_other["cls"]
                    old_cls_name = class_names[old_cls] if 0 <= old_cls < len(class_names) else f"cls_{old_cls}"
                    change_subtype = f"class_changed_from_{old_cls_name}_to_leaf_blight"

                row = {
                    "split": split,
                    "image": str(img_path),
                    "old_label": str(old_label_path),
                    "new_label": str(new_label_path),
                    "new_line_id": nb["line_id"],
                    "change_type": "added",
                    "change_subtype": change_subtype,
                    "class_id": focus_cls,
                    "class_name": args.focus_class,
                    "area_ratio": nb["area_ratio"],
                    "area_bin": area_bin(nb["area_ratio"]),
                    "best_iou_to_old_leaf_blight": best_iou_old_focus,
                    "best_iou_to_old_other": best_iou_old_other,
                    "x1": float(nb["xyxy"][0]),
                    "y1": float(nb["xyxy"][1]),
                    "x2": float(nb["xyxy"][2]),
                    "y2": float(nb["xyxy"][3]),
                    "manual_category": "",
                    "manual_action": "",
                    "manual_note": "",
                }
                diff_added.append(row)
                image_added.append(row)

                crop_path = out_dir / "label_diff_crops" / split / "added" / f"{safe_name(img_path.stem)}_new{new_idx}.jpg"
                save_crop_for_box(img, nb["xyxy"], crop_path, "ADDED leaf_blight", (0, 0, 255))

            elif best_iou_old_focus < args.label_modified_iou:
                old_xyxy = (
                    best_old_focus["xyxy"]
                    if best_old_focus is not None
                    else np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32)
                )

                row = {
                    "split": split,
                    "image": str(img_path),
                    "old_label": str(old_label_path),
                    "new_label": str(new_label_path),
                    "new_line_id": nb["line_id"],
                    "old_line_id": "" if best_old_focus is None else best_old_focus["line_id"],
                    "change_type": "modified",
                    "change_subtype": "leaf_blight_boundary_changed",
                    "class_id": focus_cls,
                    "class_name": args.focus_class,
                    "new_area_ratio": nb["area_ratio"],
                    "old_area_ratio": "" if best_old_focus is None else best_old_focus["area_ratio"],
                    "area_bin": area_bin(nb["area_ratio"]),
                    "iou_old_new": best_iou_old_focus,

                    "old_x1": float(old_xyxy[0]),
                    "old_y1": float(old_xyxy[1]),
                    "old_x2": float(old_xyxy[2]),
                    "old_y2": float(old_xyxy[3]),

                    "new_x1": float(nb["xyxy"][0]),
                    "new_y1": float(nb["xyxy"][1]),
                    "new_x2": float(nb["xyxy"][2]),
                    "new_y2": float(nb["xyxy"][3]),

                    "manual_category": "",
                    "manual_action": "",
                    "manual_note": "",
                }
                diff_modified.append(row)
                image_modified.append(row)

                crop_path = out_dir / "label_diff_crops" / split / "modified" / f"{safe_name(img_path.stem)}_new{new_idx}.jpg"
                save_crop_for_box(img, nb["xyxy"], crop_path, "MODIFIED new leaf_blight", (255, 0, 255))

        # Removed old leaf_blight
        for old_idx, ob in enumerate(old_focus):
            best_iou_new_focus, best_new_focus = max_iou_to_rows(ob["xyxy"], new_focus)
            best_iou_new_other, best_new_other = max_iou_to_rows(ob["xyxy"], new_other)

            if best_iou_new_focus < args.label_add_remove_iou:
                change_subtype = "removed_old_leaf_blight"
                if best_iou_new_other >= args.label_class_change_iou:
                    new_cls = best_new_other["cls"]
                    new_cls_name = class_names[new_cls] if 0 <= new_cls < len(class_names) else f"cls_{new_cls}"
                    change_subtype = f"class_changed_from_leaf_blight_to_{new_cls_name}"

                row = {
                    "split": split,
                    "image": str(img_path),
                    "old_label": str(old_label_path),
                    "new_label": str(new_label_path),
                    "old_line_id": ob["line_id"],
                    "change_type": "removed",
                    "change_subtype": change_subtype,
                    "class_id": focus_cls,
                    "class_name": args.focus_class,
                    "area_ratio": ob["area_ratio"],
                    "area_bin": area_bin(ob["area_ratio"]),
                    "best_iou_to_new_leaf_blight": best_iou_new_focus,
                    "best_iou_to_new_other": best_iou_new_other,
                    "x1": float(ob["xyxy"][0]),
                    "y1": float(ob["xyxy"][1]),
                    "x2": float(ob["xyxy"][2]),
                    "y2": float(ob["xyxy"][3]),
                    "manual_category": "",
                    "manual_action": "",
                    "manual_note": "",
                }
                diff_removed.append(row)
                image_removed.append(row)

                crop_path = out_dir / "label_diff_crops" / split / "removed" / f"{safe_name(img_path.stem)}_old{old_idx}.jpg"
                save_crop_for_box(img, ob["xyxy"], crop_path, "REMOVED old leaf_blight", (0, 200, 0))

        if image_added or image_removed or image_modified:
            changed_image_paths.append(img_path)

            save_label_diff_visual(
                img=img,
                image_added=image_added,
                image_removed=image_removed,
                image_modified=image_modified,
                out_path=out_dir / "label_diff_visual" / split / f"{safe_name(img_path.name)}",
            )

            changed_summary.append(
                {
                    "split": split,
                    "image": str(img_path),
                    "old_label": str(old_label_path),
                    "new_label": str(new_label_path),
                    "old_leaf_blight_count": len(old_focus),
                    "new_leaf_blight_count": len(new_focus),
                    "added_count": len(image_added),
                    "removed_count": len(image_removed),
                    "modified_count": len(image_modified),
                    "has_added": int(len(image_added) > 0),
                    "has_removed": int(len(image_removed) > 0),
                    "has_modified": int(len(image_modified) > 0),
                }
            )

    return changed_image_paths, diff_added, diff_removed, diff_modified, changed_summary, new_img_dir, new_label_dir


# ============================================================
# Model disagreement on changed images
# ============================================================

def analyze_model_disagreement_on_images(
    split: str,
    image_paths: List[Path],
    img_dir: Path,
    label_dir: Path,
    class_names: List[str],
    args,
    out_dir: Path,
):
    if not image_paths:
        return [], [], [], [], [], []

    baseline_weight = resolve_weight(args.baseline)
    method_weight = resolve_weight(args.method)

    print(f"\n[MODEL-DIFF] split={split}, changed_images={len(image_paths)}")
    print("[BASELINE]", baseline_weight)
    print("[METHOD]", method_weight)

    baseline_model = YOLO(str(baseline_weight))
    method_model = YOLO(str(method_weight))

    print("[PREDICT] baseline on changed images")
    baseline_preds = predict_model(baseline_model, image_paths, args)

    print("[PREDICT] method on changed images")
    method_preds = predict_model(method_model, image_paths, args)

    gt_missed_rows = []
    gt_match_rows = []
    all_prediction_rows = []
    method_unique_fp_rows = []
    baseline_unique_fp_rows = []
    image_prediction_summary_rows = []

    for img_idx, img_path in enumerate(image_paths, start=1):
        print(f"[MODEL-ANALYZE] {split} {img_idx}/{len(image_paths)} {img_path.name}")

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        label_path = find_label_for_image(img_path, img_dir, label_dir)
        gts = read_yolo_label(label_path, img_w, img_h)

        b_preds = baseline_preds.get(str(img_path), [])
        m_preds = method_preds.get(str(img_path), [])

        baseline_pred_rows = match_predictions_to_gts(
            model_tag="baseline",
            img_path=img_path,
            label_path=label_path,
            preds=b_preds,
            gts=gts,
            class_names=class_names,
            img_w=img_w,
            img_h=img_h,
            args=args,
        )

        method_pred_rows = match_predictions_to_gts(
            model_tag="method",
            img_path=img_path,
            label_path=label_path,
            preds=m_preds,
            gts=gts,
            class_names=class_names,
            img_w=img_w,
            img_h=img_h,
            args=args,
        )

        for r in baseline_pred_rows:
            r["split"] = split
        for r in method_pred_rows:
            r["split"] = split

        all_prediction_rows.extend(baseline_pred_rows)
        all_prediction_rows.extend(method_pred_rows)

        baseline_fp_rows = [r for r in baseline_pred_rows if r["prediction_status"] == "FP"]
        method_fp_rows = [r for r in method_pred_rows if r["prediction_status"] == "FP"]

        method_unique_this_image = []
        baseline_unique_this_image = []

        for fp in method_fp_rows:
            if is_unique_fp(fp, baseline_fp_rows, args):
                fp = dict(fp)
                fp["split"] = split
                fp["unique_fp_source"] = "method_unique_fp"
                method_unique_fp_rows.append(fp)
                method_unique_this_image.append(fp)

        for fp in baseline_fp_rows:
            if is_unique_fp(fp, method_fp_rows, args):
                fp = dict(fp)
                fp["split"] = split
                fp["unique_fp_source"] = "baseline_unique_fp"
                baseline_unique_fp_rows.append(fp)
                baseline_unique_this_image.append(fp)

        candidate_visual = img.copy()
        has_candidate = False
        local_gt_match_rows = []

        for gt_index, gt in enumerate(gts):
            gt_cls = gt["cls"]
            gt_cls_name = class_names[gt_cls] if 0 <= gt_cls < len(class_names) else f"cls_{gt_cls}"
            gt_box = gt["xyxy"]

            b_same, b_any, b_wrong = best_pred_for_gt(gt_box, b_preds, gt_cls)
            m_same, m_any, m_wrong = best_pred_for_gt(gt_box, m_preds, gt_cls)

            baseline_detected = b_same["iou"] >= args.match_iou
            method_detected = m_same["iou"] >= args.match_iou

            gt_match_row = {
                "split": split,
                "image": str(img_path),
                "label": str(label_path),
                "gt_index": gt_index,
                "gt_line_id": gt["line_id"],
                "gt_class_id": gt_cls,
                "gt_class_name": gt_cls_name,
                "gt_area_ratio": gt["area_ratio"],
                "gt_area_bin": area_bin(gt["area_ratio"]),
                "baseline_same_iou": b_same["iou"],
                "baseline_same_conf": b_same["conf"],
                "method_same_iou": m_same["iou"],
                "method_same_conf": m_same["conf"],
                "method_any_iou": m_any["iou"],
                "method_any_conf": m_any["conf"],
                "method_any_class_id": "" if m_any["cls"] is None else m_any["cls"],
                "method_any_class_name": "" if m_any["cls"] is None else class_names[m_any["cls"]],
                "baseline_detected": int(baseline_detected),
                "method_detected": int(method_detected),
            }

            gt_match_rows.append(gt_match_row)
            local_gt_match_rows.append(gt_match_row)

            if baseline_detected and not method_detected:
                if m_wrong["iou"] >= args.overlap_iou:
                    miss_type = "method_predicted_wrong_class"
                elif m_any["iou"] >= args.overlap_iou:
                    miss_type = "method_overlap_low_iou_or_low_conf"
                else:
                    miss_type = "method_no_prediction_near_gt"

                row = {
                    "split": split,
                    "image": str(img_path),
                    "label": str(label_path),
                    "gt_index": gt_index,
                    "gt_line_id": gt["line_id"],
                    "gt_class_id": gt_cls,
                    "gt_class_name": gt_cls_name,
                    "gt_area_ratio": gt["area_ratio"],
                    "gt_area_bin": area_bin(gt["area_ratio"]),
                    "gt_x1": float(gt_box[0]),
                    "gt_y1": float(gt_box[1]),
                    "gt_x2": float(gt_box[2]),
                    "gt_y2": float(gt_box[3]),
                    "baseline_same_iou": b_same["iou"],
                    "baseline_same_conf": b_same["conf"],
                    "method_same_iou": m_same["iou"],
                    "method_same_conf": m_same["conf"],
                    "method_any_iou": m_any["iou"],
                    "method_any_conf": m_any["conf"],
                    "method_any_class_id": "" if m_any["cls"] is None else m_any["cls"],
                    "method_any_class_name": "" if m_any["cls"] is None else class_names[m_any["cls"]],
                    "miss_type": miss_type,
                    "manual_label_action": "",
                    "manual_note": "",
                }
                gt_missed_rows.append(row)

                has_candidate = True
                draw_box(candidate_visual, gt_box, (0, 255, 255), f"GT:{gt_cls_name}", 3)
                if b_same["xyxy"] is not None:
                    draw_box(candidate_visual, b_same["xyxy"], (0, 255, 0), f"BASE {b_same['conf']:.2f}", 2)
                if m_same["xyxy"] is not None:
                    draw_box(candidate_visual, m_same["xyxy"], (0, 0, 255), f"METHOD same {m_same['conf']:.2f}", 2)
                elif m_any["xyxy"] is not None:
                    m_cls_name = class_names[m_any["cls"]] if m_any["cls"] is not None else "unknown"
                    draw_box(candidate_visual, m_any["xyxy"], (0, 0, 255), f"METHOD {m_cls_name} {m_any['conf']:.2f}", 2)

        if has_candidate:
            ensure_dir(out_dir / "changed_images_model_missed_visual" / split)
            cv2.imwrite(
                str(out_dir / "changed_images_model_missed_visual" / split / f"{safe_name(img_path.name)}"),
                candidate_visual,
            )

        baseline_tp_count = sum(1 for r in baseline_pred_rows if r["prediction_status"] == "TP")
        baseline_fp_count = sum(1 for r in baseline_pred_rows if r["prediction_status"] == "FP")
        method_tp_count = sum(1 for r in method_pred_rows if r["prediction_status"] == "TP")
        method_fp_count = sum(1 for r in method_pred_rows if r["prediction_status"] == "FP")

        gt_count = len(local_gt_match_rows)
        baseline_detected_gt_count = sum(1 for r in local_gt_match_rows if int(r["baseline_detected"]) == 1)
        method_detected_gt_count = sum(1 for r in local_gt_match_rows if int(r["method_detected"]) == 1)

        baseline_detected_method_missed_count = sum(
            1
            for r in local_gt_match_rows
            if int(r["baseline_detected"]) == 1 and int(r["method_detected"]) == 0
        )

        method_unique_fp_count = len(method_unique_this_image)
        baseline_unique_fp_count = len(baseline_unique_this_image)

        method_unique_high_conf_fp_005_count = sum(
            1 for r in method_unique_this_image if float(r["pred_conf"]) >= 0.05
        )
        baseline_unique_high_conf_fp_005_count = sum(
            1 for r in baseline_unique_this_image if float(r["pred_conf"]) >= 0.05
        )

        flags = []
        if baseline_detected_method_missed_count > 0:
            flags.append("baseline_detected_method_missed")
        if method_unique_high_conf_fp_005_count > 0:
            flags.append("method_unique_high_conf_fp")
        if baseline_unique_high_conf_fp_005_count > 0:
            flags.append("baseline_unique_high_conf_fp")
        if method_fp_count > baseline_fp_count:
            flags.append("method_more_fp")
        if method_detected_gt_count < baseline_detected_gt_count:
            flags.append("method_lower_gt_detect")

        image_prediction_summary_rows.append(
            {
                "split": split,
                "image": str(img_path),
                "label": str(label_path),

                "gt_count": gt_count,
                "baseline_detected_gt_count": baseline_detected_gt_count,
                "method_detected_gt_count": method_detected_gt_count,
                "baseline_detected_method_missed_count": baseline_detected_method_missed_count,

                "baseline_tp_count": baseline_tp_count,
                "baseline_fp_count": baseline_fp_count,
                "method_tp_count": method_tp_count,
                "method_fp_count": method_fp_count,

                "method_unique_fp_count": method_unique_fp_count,
                "baseline_unique_fp_count": baseline_unique_fp_count,

                "method_unique_high_conf_fp_005_count": method_unique_high_conf_fp_005_count,
                "baseline_unique_high_conf_fp_005_count": baseline_unique_high_conf_fp_005_count,

                "prediction_diff_flag": "|".join(flags),
            }
        )

    return (
        gt_missed_rows,
        gt_match_rows,
        all_prediction_rows,
        method_unique_fp_rows,
        baseline_unique_fp_rows,
        image_prediction_summary_rows,
    )


# ============================================================
# Summaries
# ============================================================

def summarize_label_changes(rows: List[dict], change_name: str):
    summary = {}
    for r in rows:
        key = (r["split"], r.get("area_bin", ""))
        summary.setdefault(
            key,
            {
                "split": r["split"],
                "change_type": change_name,
                "area_bin": r.get("area_bin", ""),
                "count": 0,
            },
        )
        summary[key]["count"] += 1
    return list(summary.values())


def summarize_gt_match(rows: List[dict]):
    summary = {}

    for r in rows:
        key = (r["split"], r["gt_class_name"])
        summary.setdefault(
            key,
            {
                "split": r["split"],
                "class_name": r["gt_class_name"],
                "gt_count": 0,
                "baseline_detected_count": 0,
                "method_detected_count": 0,
                "baseline_detected_method_missed_count": 0,
            },
        )

        s = summary[key]
        s["gt_count"] += 1
        if int(r["baseline_detected"]) == 1:
            s["baseline_detected_count"] += 1
        if int(r["method_detected"]) == 1:
            s["method_detected_count"] += 1
        if int(r["baseline_detected"]) == 1 and int(r["method_detected"]) == 0:
            s["baseline_detected_method_missed_count"] += 1

    out = []
    for s in summary.values():
        s["baseline_recall_iou50_on_changed_images"] = safe_div(s["baseline_detected_count"], s["gt_count"])
        s["method_recall_iou50_on_changed_images"] = safe_div(s["method_detected_count"], s["gt_count"])
        out.append(s)
    return out


def summarize_prediction_rows(rows: List[dict]):
    summary = {}

    for r in rows:
        key = (r["split"], r["model_tag"], r["pred_class_name"], r["prediction_status"], r["fp_type"])
        summary.setdefault(
            key,
            {
                "split": r["split"],
                "model_tag": r["model_tag"],
                "class_name": r["pred_class_name"],
                "prediction_status": r["prediction_status"],
                "fp_type": r["fp_type"],
                "count": 0,
                "high_conf_005_count": 0,
                "high_conf_025_count": 0,
                "mean_conf_sum": 0.0,
            },
        )
        s = summary[key]
        s["count"] += 1
        s["mean_conf_sum"] += float(r["pred_conf"])
        if float(r["pred_conf"]) >= 0.05:
            s["high_conf_005_count"] += 1
        if float(r["pred_conf"]) >= 0.25:
            s["high_conf_025_count"] += 1

    out = []
    for s in summary.values():
        s["mean_conf"] = safe_div(s["mean_conf_sum"], s["count"])
        del s["mean_conf_sum"]
        out.append(s)
    return out


def summarize_unique_fp(rows: List[dict]):
    summary = {}

    for r in rows:
        key = (r["split"], r["unique_fp_source"], r["pred_class_name"], r["fp_type"])
        summary.setdefault(
            key,
            {
                "split": r["split"],
                "unique_fp_source": r["unique_fp_source"],
                "class_name": r["pred_class_name"],
                "fp_type": r["fp_type"],
                "count": 0,
                "high_conf_005_count": 0,
                "high_conf_025_count": 0,
                "mean_conf_sum": 0.0,
            },
        )
        s = summary[key]
        s["count"] += 1
        s["mean_conf_sum"] += float(r["pred_conf"])
        if float(r["pred_conf"]) >= 0.05:
            s["high_conf_005_count"] += 1
        if float(r["pred_conf"]) >= 0.25:
            s["high_conf_025_count"] += 1

    out = []
    for s in summary.values():
        s["mean_conf"] = safe_div(s["mean_conf_sum"], s["count"])
        del s["mean_conf_sum"]
        out.append(s)
    return out


def merge_label_and_model_summary(label_rows: List[dict], model_rows: List[dict]) -> List[dict]:
    model_map = {}

    for r in model_rows:
        key = (r["split"], r["image"])
        model_map[key] = r

    merged = []

    for r in label_rows:
        key = (r["split"], r["image"])
        m = model_map.get(key, {})

        row = dict(r)

        row["gt_count_on_changed_image"] = m.get("gt_count", 0)
        row["baseline_detected_gt_count"] = m.get("baseline_detected_gt_count", 0)
        row["method_detected_gt_count"] = m.get("method_detected_gt_count", 0)
        row["baseline_detected_method_missed_count"] = m.get("baseline_detected_method_missed_count", 0)

        row["baseline_tp_count"] = m.get("baseline_tp_count", 0)
        row["baseline_fp_count"] = m.get("baseline_fp_count", 0)
        row["method_tp_count"] = m.get("method_tp_count", 0)
        row["method_fp_count"] = m.get("method_fp_count", 0)

        row["method_unique_fp_count"] = m.get("method_unique_fp_count", 0)
        row["baseline_unique_fp_count"] = m.get("baseline_unique_fp_count", 0)

        row["method_unique_high_conf_fp_005_count"] = m.get("method_unique_high_conf_fp_005_count", 0)
        row["baseline_unique_high_conf_fp_005_count"] = m.get("baseline_unique_high_conf_fp_005_count", 0)

        row["prediction_diff_flag"] = m.get("prediction_diff_flag", "")

        merged.append(row)

    return merged


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="./data/det_dataset.yaml", help="dataset yaml for split paths and class names")
    parser.add_argument("--old-data-root", default="./data/det_dataset_202605112210", help="old dataset root")
    parser.add_argument("--new-data-root", default="./data/det_dataset", help="new/current dataset root")

    parser.add_argument("--baseline", required=True, help="baseline best.pt or run dir")
    parser.add_argument("--method", required=True, help="LS/method best.pt or run dir")
    parser.add_argument("--method-name", default="ls")

    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--focus-class", default="leaf_blight")
    parser.add_argument("--out", default="runs_audit")

    # label diff thresholds
    parser.add_argument("--label-add-remove-iou", type=float, default=0.50)
    parser.add_argument("--label-modified-iou", type=float, default=0.85)
    parser.add_argument("--label-class-change-iou", type=float, default=0.50)

    # prediction settings
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="0")

    # matching thresholds
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--overlap-iou", type=float, default=0.30)
    parser.add_argument("--unique-fp-iou", type=float, default=0.50)

    args = parser.parse_args()

    data_yaml = Path(args.data)
    old_root = Path(args.old_data_root).resolve()
    new_root = Path(args.new_data_root).resolve()

    data, class_names = parse_dataset_yaml(data_yaml)

    if args.focus_class not in class_names:
        raise ValueError(f"focus class {args.focus_class} not found in class names: {class_names}")

    out_dir = Path(args.out).resolve()
    conf_tag = f"{int(round(args.conf * 1000)):03d}"
    out_dir = out_dir / f"labeldiff_{args.focus_class}_modeldiff_{safe_name(args.method_name)}_conf{conf_tag}"
    ensure_dir(out_dir)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    all_changed_paths_by_split = {}
    all_new_img_dir_by_split = {}
    all_new_label_dir_by_split = {}

    all_added = []
    all_removed = []
    all_modified = []
    all_changed_summary = []

    for split in splits:
        (
            changed_paths,
            added,
            removed,
            modified,
            changed_summary,
            new_img_dir,
            new_label_dir,
        ) = compare_leaf_blight_labels_for_split(
            split=split,
            data=data,
            old_root=old_root,
            new_root=new_root,
            class_names=class_names,
            args=args,
            out_dir=out_dir,
        )

        all_changed_paths_by_split[split] = changed_paths
        all_new_img_dir_by_split[split] = new_img_dir
        all_new_label_dir_by_split[split] = new_label_dir

        all_added.extend(added)
        all_removed.extend(removed)
        all_modified.extend(modified)
        all_changed_summary.extend(changed_summary)

    # Save label diff outputs first
    write_csv(out_dir / "01_changed_images_summary.csv", all_changed_summary)
    write_csv(out_dir / "02_added_leaf_blight_boxes.csv", all_added)
    write_csv(out_dir / "03_removed_leaf_blight_boxes.csv", all_removed)
    write_csv(out_dir / "04_modified_leaf_blight_boxes.csv", all_modified)

    label_summary = []
    label_summary.extend(summarize_label_changes(all_added, "added"))
    label_summary.extend(summarize_label_changes(all_removed, "removed"))
    label_summary.extend(summarize_label_changes(all_modified, "modified"))
    write_csv(out_dir / "05_label_change_summary_by_area.csv", label_summary)

    # Model disagreement only on changed images
    all_gt_missed = []
    all_gt_match = []
    all_prediction_rows = []
    all_method_unique_fp = []
    all_baseline_unique_fp = []
    all_image_prediction_summary = []

    for split in splits:
        changed_paths = all_changed_paths_by_split.get(split, [])
        new_img_dir = all_new_img_dir_by_split[split]
        new_label_dir = all_new_label_dir_by_split[split]

        (
            gt_missed,
            gt_match,
            prediction_rows,
            method_unique_fp,
            baseline_unique_fp,
            image_prediction_summary,
        ) = analyze_model_disagreement_on_images(
            split=split,
            image_paths=changed_paths,
            img_dir=new_img_dir,
            label_dir=new_label_dir,
            class_names=class_names,
            args=args,
            out_dir=out_dir,
        )

        all_gt_missed.extend(gt_missed)
        all_gt_match.extend(gt_match)
        all_prediction_rows.extend(prediction_rows)
        all_method_unique_fp.extend(method_unique_fp)
        all_baseline_unique_fp.extend(baseline_unique_fp)
        all_image_prediction_summary.extend(image_prediction_summary)

    write_csv(out_dir / "06_changed_images_gt_match_summary.csv", all_gt_match)
    write_csv(out_dir / "07_changed_images_baseline_detected_method_missed.csv", all_gt_missed)
    write_csv(out_dir / "08_changed_images_prediction_tp_fp.csv", public_rows(all_prediction_rows))
    write_csv(out_dir / "09_changed_images_method_unique_fp.csv", public_rows(all_method_unique_fp))
    write_csv(out_dir / "10_changed_images_baseline_unique_fp.csv", public_rows(all_baseline_unique_fp))

    write_csv(out_dir / "11_summary_gt_recall_on_changed_images.csv", summarize_gt_match(all_gt_match))
    write_csv(out_dir / "12_summary_predictions_tp_fp_on_changed_images.csv", summarize_prediction_rows(all_prediction_rows))
    write_csv(out_dir / "13_summary_method_unique_fp_on_changed_images.csv", summarize_unique_fp(all_method_unique_fp))
    write_csv(out_dir / "14_summary_baseline_unique_fp_on_changed_images.csv", summarize_unique_fp(all_baseline_unique_fp))

    write_csv(
        out_dir / "15_image_prediction_diff_summary.csv",
        all_image_prediction_summary,
    )

    write_csv(
        out_dir / "00_changed_files_label_and_prediction_diff.csv",
        merge_label_and_model_summary(
            label_rows=all_changed_summary,
            model_rows=all_image_prediction_summary,
        ),
    )

    manifest = {
        "data_yaml": str(data_yaml),
        "old_data_root": str(old_root),
        "new_data_root": str(new_root),
        "baseline": str(resolve_weight(args.baseline)),
        "method": str(resolve_weight(args.method)),
        "method_name": args.method_name,
        "focus_class": args.focus_class,
        "splits": splits,
        "class_names": class_names,
        "args": vars(args),
        "outputs": {
            "changed_files_label_and_prediction_diff": str(out_dir / "00_changed_files_label_and_prediction_diff.csv"),
            "changed_images_summary": str(out_dir / "01_changed_images_summary.csv"),
            "added_leaf_blight_boxes": str(out_dir / "02_added_leaf_blight_boxes.csv"),
            "removed_leaf_blight_boxes": str(out_dir / "03_removed_leaf_blight_boxes.csv"),
            "modified_leaf_blight_boxes": str(out_dir / "04_modified_leaf_blight_boxes.csv"),
            "label_diff_visual": str(out_dir / "label_diff_visual"),
            "label_diff_crops": str(out_dir / "label_diff_crops"),
            "changed_images_baseline_detected_method_missed": str(out_dir / "07_changed_images_baseline_detected_method_missed.csv"),
            "changed_images_method_unique_fp": str(out_dir / "09_changed_images_method_unique_fp.csv"),
            "image_prediction_diff_summary": str(out_dir / "15_image_prediction_diff_summary.csv"),
        },
    }

    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = f"""# Label diff + model disagreement audit

This report compares old and new/current dataset roots.

Focus class: `{args.focus_class}`

## Stage 1: label diff

The script finds:
- added `{args.focus_class}` boxes
- removed `{args.focus_class}` boxes
- modified `{args.focus_class}` boxes
- class changes into/from `{args.focus_class}`

Outputs:
- 00_changed_files_label_and_prediction_diff.csv
- 01_changed_images_summary.csv
- 02_added_leaf_blight_boxes.csv
- 03_removed_leaf_blight_boxes.csv
- 04_modified_leaf_blight_boxes.csv
- label_diff_visual/
- label_diff_crops/

Important:
- label_diff_visual only draws changed boxes.
- unchanged boxes are bypassed.

Visual colors:
- red: added new leaf_blight
- green: removed old leaf_blight
- orange: modified old boundary
- magenta: modified new boundary

## Stage 2: model disagreement only on changed images

The script only runs baseline/method analysis on images where `{args.focus_class}` labels changed.

Outputs:
- 06_changed_images_gt_match_summary.csv
- 07_changed_images_baseline_detected_method_missed.csv
- 08_changed_images_prediction_tp_fp.csv
- 09_changed_images_method_unique_fp.csv
- 10_changed_images_baseline_unique_fp.csv
- 15_image_prediction_diff_summary.csv

## Main file to review

Open:

00_changed_files_label_and_prediction_diff.csv

This file contains both label changes and prediction differences:
- old/new leaf_blight count
- added/removed/modified count
- baseline detected GT count
- method detected GT count
- baseline detected but method missed count
- baseline FP count
- method FP count
- method unique FP count
- baseline unique FP count
- high confidence unique FP count
- prediction_diff_flag

## Manual review rule

For added leaf_blight:
- keep: clear necrotic spot, clear dry edge, clear dry tip
- delete: shadow, reflection, slight dark tip, leaf_spot transition, algal edge, blurry tiny point

For removed leaf_blight:
- deletion is correct if it was shadow/reflection/slight dark tip/blurry/noise
- deletion is wrong if it was clear necrosis/dry edge/dry tip

For modified leaf_blight:
- correct if new box better covers complete necrotic area or removes normal leaf/background
- wrong if new box becomes too small or includes too much healthy leaf
"""

    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print("\n[DONE]")
    print("[OUT]", out_dir)
    print("[KEY FILES]")
    print("  00_changed_files_label_and_prediction_diff.csv")
    print("  01_changed_images_summary.csv")
    print("  02_added_leaf_blight_boxes.csv")
    print("  03_removed_leaf_blight_boxes.csv")
    print("  04_modified_leaf_blight_boxes.csv")
    print("  07_changed_images_baseline_detected_method_missed.csv")
    print("  09_changed_images_method_unique_fp.csv")
    print("  15_image_prediction_diff_summary.csv")
    print("  label_diff_visual/")
    print("  label_diff_crops/")


if __name__ == "__main__":
    main()