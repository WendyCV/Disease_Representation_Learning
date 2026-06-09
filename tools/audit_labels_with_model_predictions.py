import sys
import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import yaml

# ============================================================
# Fix project import path
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# print(f"[DEBUG] PROJECT_ROOT = {PROJECT_ROOT}")
# print(f"[DEBUG] sys.path[0] = {sys.path[0]}")

from models import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ============================================================
# Basic utils
# ============================================================

def safe_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\-.]+", "_", name)
    return name[:160]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


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


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base: Path, p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return base / pp


def parse_model_specs(args) -> List[Tuple[str, Path]]:
    """
    Accept:
      --weight name=path/to/best.pt
      --run name=exp_dir  -> exp_dir/weights/best.pt
    """
    specs = []

    for item in args.weight:
        if "=" not in item:
            raise ValueError(f"--weight must use name=path format, got: {item}")
        name, p = item.split("=", 1)
        specs.append((safe_name(name), Path(p)))

    for item in args.run:
        if "=" not in item:
            raise ValueError(f"--run must use name=exp_dir format, got: {item}")
        name, p = item.split("=", 1)
        exp_dir = Path(p)
        specs.append((safe_name(name), exp_dir / "weights" / "best.pt"))

    if not specs:
        raise ValueError("Please provide at least one --weight or --run.")

    checked = []
    for name, w in specs:
        if not w.exists():
            raise FileNotFoundError(f"Weight not found for {name}: {w}")
        checked.append((name, w))

    return checked


# ============================================================
# Dataset parsing
# ============================================================

def parse_dataset_yaml(data_yaml: Path, split: str):
    data = load_yaml(data_yaml)

    if "path" not in data:
        raise ValueError("dataset yaml must contain `path`.")

    dataset_root = Path(data["path"])
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()

    if split not in data or data[split] is None:
        raise ValueError(f"dataset yaml must contain `{split}` path.")

    img_dir = resolve_path(dataset_root, data[split])

    names = data.get("names")
    if names is None:
        raise ValueError("dataset yaml must contain `names`.")

    if isinstance(names, dict):
        class_names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    elif isinstance(names, list):
        class_names = names
    else:
        raise ValueError("Unsupported names format.")

    parts = list(img_dir.parts)
    if "images" not in parts:
        raise ValueError(f"Cannot infer label dir from image dir: {img_dir}")

    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    label_dir = Path(*parts)

    return dataset_root, img_dir, label_dir, class_names


def collect_images(img_dir: Path) -> List[Path]:
    images = []
    for p in img_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            images.append(p)
    return sorted(images)


def find_label_for_image(img_path: Path, img_root: Path, label_root: Path) -> Path:
    rel = img_path.relative_to(img_root).with_suffix(".txt")
    return label_root / rel


def read_yolo_label(label_path: Path, img_w: int, img_h: int) -> List[dict]:
    """
    Return:
    [
      {
        line_id,
        cls,
        xyxy,
        area_ratio,
        raw_line
      }
    ]
    """
    if not label_path.exists():
        return []

    lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    gts = []

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

        gts.append(
            {
                "line_id": line_id,
                "cls": cls,
                "xyxy": np.array([x1, y1, x2, y2], dtype=np.float32),
                "area_ratio": area_ratio,
                "raw_line": raw,
            }
        )

    return gts


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

    area1 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area2 = (
        np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    )

    union = area1 + area2 - inter
    return inter / np.maximum(union, 1e-9)


def box_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def union_box(boxes: List[np.ndarray]) -> Optional[np.ndarray]:
    if not boxes:
        return None

    arr = np.stack(boxes, axis=0)
    return np.array(
        [
            float(np.min(arr[:, 0])),
            float(np.min(arr[:, 1])),
            float(np.max(arr[:, 2])),
            float(np.max(arr[:, 3])),
        ],
        dtype=np.float32,
    )


# ============================================================
# Prediction
# ============================================================

def predict_all_models(
    model_specs: List[Tuple[str, Path]],
    image_paths: List[Path],
    args,
) -> Dict[str, Dict[str, List[dict]]]:
    """
    Return:
    predictions[image_key][model_name] = [
      {cls, conf, xyxy}
    ]

    image_key = str(image_path)
    """
    all_predictions: Dict[str, Dict[str, List[dict]]] = {
        str(p): {} for p in image_paths
    }

    for model_name, weight_path in model_specs:
        print("\n" + "=" * 80)
        print(f"[PREDICT] {model_name}")
        print(f"[WEIGHT] {weight_path}")

        model = YOLO(str(weight_path))

        results_iter = model.predict(
            source=[str(p) for p in image_paths],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.nms_iou,
            max_det=args.max_det,
            device=args.device,
            verbose=True,
            stream=True,
        )

        results = list(results_iter)

        if len(results) != len(image_paths):
            print(
                f"[WARN] results count mismatch for {model_name}: "
                f"results={len(results)}, images={len(image_paths)}"
            )

        for img_path, result in zip(image_paths, results):
            preds = []

            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
                cls = result.boxes.cls.cpu().numpy().astype(np.int64)
                conf = result.boxes.conf.cpu().numpy().astype(np.float32)

                for b, c, cf in zip(xyxy, cls, conf):
                    preds.append(
                        {
                            "cls": int(c),
                            "conf": float(cf),
                            "xyxy": b,
                            "model": model_name,
                        }
                    )

            all_predictions[str(img_path)][model_name] = preds

    return all_predictions


# ============================================================
# Clustering unmatched predictions
# ============================================================

def cluster_predictions(preds: List[dict], iou_thr: float):
    """
    Simple class-aware greedy clustering.
    Preds should already be unmatched to GT.
    """
    remaining = sorted(preds, key=lambda x: x["conf"], reverse=True)
    clusters = []

    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]

        keep = []
        for p in remaining:
            if p["cls"] != seed["cls"]:
                keep.append(p)
                continue

            iou = float(box_iou_one_to_many(seed["xyxy"], np.array([p["xyxy"]]))[0])
            if iou >= iou_thr:
                cluster.append(p)
            else:
                keep.append(p)

        remaining = keep

        clusters.append(cluster)

    return clusters


def summarize_cluster(cluster: List[dict], class_names: List[str], img_area: float):
    boxes = [p["xyxy"] for p in cluster]
    ubox = union_box(boxes)
    cls_id = cluster[0]["cls"]

    confs = [p["conf"] for p in cluster]
    models = sorted(set(p["model"] for p in cluster))

    return {
        "cls": cls_id,
        "class_name": class_names[cls_id] if 0 <= cls_id < len(class_names) else f"cls_{cls_id}",
        "support_model_count": len(models),
        "support_models": "|".join(models),
        "mean_conf": float(np.mean(confs)) if confs else float("nan"),
        "max_conf": float(np.max(confs)) if confs else float("nan"),
        "xyxy": ubox,
        "area_ratio": safe_div(box_area(ubox), img_area) if ubox is not None else float("nan"),
    }


# ============================================================
# Core audit
# ============================================================

def analyze_image(
    img_path: Path,
    img_root: Path,
    label_root: Path,
    class_names: List[str],
    preds_by_model: Dict[str, List[dict]],
    model_names: List[str],
    args,
):
    img = cv2.imread(str(img_path))
    if img is None:
        return [], [], [], [], []

    img_h, img_w = img.shape[:2]
    img_area = float(img_h * img_w)

    label_path = find_label_for_image(img_path, img_root, label_root)
    gts = read_yolo_label(label_path, img_w, img_h)

    gt_summary_rows = []
    possible_wrong_class_rows = []
    possible_box_refinement_rows = []
    missing_label_candidate_rows = []
    model_gt_rows = []

    # --------------------------------------------------------
    # GT-level analysis
    # --------------------------------------------------------
    for gt_idx, gt in enumerate(gts):
        gt_cls = gt["cls"]
        gt_cls_name = class_names[gt_cls] if 0 <= gt_cls < len(class_names) else f"cls_{gt_cls}"
        gt_box = gt["xyxy"]

        same_class_hits = []
        any_class_hits = []
        wrong_class_hits = []

        per_model_best = {}

        for model_name in model_names:
            preds = preds_by_model.get(model_name, [])

            if preds:
                pred_boxes = np.array([p["xyxy"] for p in preds], dtype=np.float32)
                ious = box_iou_one_to_many(gt_box, pred_boxes)
            else:
                ious = np.zeros((0,), dtype=np.float32)

            best_same_iou = 0.0
            best_same_conf = float("nan")
            best_same_box = None

            best_any_iou = 0.0
            best_any_conf = float("nan")
            best_any_cls = None
            best_any_box = None

            best_wrong_iou = 0.0
            best_wrong_conf = float("nan")
            best_wrong_cls = None
            best_wrong_box = None

            for p_i, p in enumerate(preds):
                iou = float(ious[p_i]) if len(ious) else 0.0

                if iou > best_any_iou:
                    best_any_iou = iou
                    best_any_conf = p["conf"]
                    best_any_cls = p["cls"]
                    best_any_box = p["xyxy"]

                if p["cls"] == gt_cls and iou > best_same_iou:
                    best_same_iou = iou
                    best_same_conf = p["conf"]
                    best_same_box = p["xyxy"]

                if p["cls"] != gt_cls and iou > best_wrong_iou:
                    best_wrong_iou = iou
                    best_wrong_conf = p["conf"]
                    best_wrong_cls = p["cls"]
                    best_wrong_box = p["xyxy"]

            same_matched = best_same_iou >= args.match_iou
            any_overlapped = best_any_iou >= args.overlap_iou
            wrong_overlapped = best_wrong_iou >= args.overlap_iou

            if same_matched:
                same_class_hits.append(model_name)

            if any_overlapped:
                any_class_hits.append(model_name)

            if wrong_overlapped:
                wrong_class_hits.append(
                    {
                        "model": model_name,
                        "pred_cls": best_wrong_cls,
                        "pred_conf": best_wrong_conf,
                        "iou": best_wrong_iou,
                        "box": best_wrong_box,
                    }
                )

            per_model_best[model_name] = {
                "same_iou": best_same_iou,
                "same_conf": best_same_conf,
                "any_iou": best_any_iou,
                "any_conf": best_any_conf,
                "any_cls": best_any_cls,
                "wrong_iou": best_wrong_iou,
                "wrong_conf": best_wrong_conf,
                "wrong_cls": best_wrong_cls,
            }

            model_gt_rows.append(
                {
                    "image": str(img_path),
                    "label": str(label_path),
                    "gt_index": gt_idx,
                    "gt_line_id": gt["line_id"],
                    "gt_class_id": gt_cls,
                    "gt_class_name": gt_cls_name,
                    "gt_area_ratio": gt["area_ratio"],
                    "model": model_name,
                    "best_same_class_iou": best_same_iou,
                    "best_same_class_conf": best_same_conf,
                    "best_any_class_iou": best_any_iou,
                    "best_any_class_conf": best_any_conf,
                    "best_any_class_id": best_any_cls,
                    "best_any_class_name": (
                        class_names[best_any_cls]
                        if best_any_cls is not None and 0 <= best_any_cls < len(class_names)
                        else ""
                    ),
                    "best_wrong_class_iou": best_wrong_iou,
                    "best_wrong_class_conf": best_wrong_conf,
                    "best_wrong_class_id": best_wrong_cls,
                    "best_wrong_class_name": (
                        class_names[best_wrong_cls]
                        if best_wrong_cls is not None and 0 <= best_wrong_cls < len(class_names)
                        else ""
                    ),
                }
            )

        same_support = len(same_class_hits)
        any_support = len(any_class_hits)
        wrong_support = len(wrong_class_hits)

        best_same_iou_all = max([v["same_iou"] for v in per_model_best.values()], default=0.0)
        best_any_iou_all = max([v["any_iou"] for v in per_model_best.values()], default=0.0)

        # Wrong class consensus
        wrong_class_votes = {}
        for h in wrong_class_hits:
            c = h["pred_cls"]
            wrong_class_votes[c] = wrong_class_votes.get(c, 0) + 1

        top_wrong_cls = None
        top_wrong_votes = 0
        if wrong_class_votes:
            top_wrong_cls = max(wrong_class_votes.keys(), key=lambda k: wrong_class_votes[k])
            top_wrong_votes = wrong_class_votes[top_wrong_cls]

        # GT issue level
        if same_support == 0 and any_support == 0:
            gt_issue = "GT_NOT_SUPPORTED_BY_MODELS"
            suggestion = "Check whether this GT is too ambiguous, too small, wrong, or not learnable."
        elif same_support == 0 and wrong_support >= args.min_model_support:
            gt_issue = "POSSIBLE_WRONG_CLASS"
            suggestion = "Several models detect overlapping region as another class. Check class label."
        elif same_support <= args.low_support_threshold:
            gt_issue = "LOW_MODEL_SUPPORT"
            suggestion = "GT is rarely detected. Check bbox granularity, class ambiguity, or small/occluded target."
        elif best_same_iou_all < args.box_refine_iou and same_support >= args.min_model_support:
            gt_issue = "POSSIBLE_BOX_GRANULARITY_MISMATCH"
            suggestion = "Same-class predictions overlap but do not align tightly. Check whether GT box is too small/large/split."
        else:
            gt_issue = "OK_OR_MODEL_DEPENDENT"
            suggestion = "No strong label issue signal."

        gt_summary_rows.append(
            {
                "image": str(img_path),
                "label": str(label_path),
                "gt_index": gt_idx,
                "gt_line_id": gt["line_id"],
                "gt_class_id": gt_cls,
                "gt_class_name": gt_cls_name,
                "gt_area_ratio": gt["area_ratio"],
                "gt_x1": float(gt_box[0]),
                "gt_y1": float(gt_box[1]),
                "gt_x2": float(gt_box[2]),
                "gt_y2": float(gt_box[3]),
                "model_count": len(model_names),
                "same_class_support": same_support,
                "same_class_support_models": "|".join(same_class_hits),
                "any_class_support": any_support,
                "wrong_class_support": wrong_support,
                "top_wrong_class_id": top_wrong_cls if top_wrong_cls is not None else "",
                "top_wrong_class_name": (
                    class_names[top_wrong_cls]
                    if top_wrong_cls is not None and 0 <= top_wrong_cls < len(class_names)
                    else ""
                ),
                "top_wrong_votes": top_wrong_votes,
                "best_same_iou_all_models": best_same_iou_all,
                "best_any_iou_all_models": best_any_iou_all,
                "issue_type": gt_issue,
                "suggestion": suggestion,
            }
        )

        if gt_issue == "POSSIBLE_WRONG_CLASS":
            possible_wrong_class_rows.append(gt_summary_rows[-1])

        if gt_issue == "POSSIBLE_BOX_GRANULARITY_MISMATCH":
            possible_box_refinement_rows.append(gt_summary_rows[-1])

    # --------------------------------------------------------
    # Missing label candidate analysis
    # Predictions not overlapping any GT sufficiently
    # --------------------------------------------------------
    all_unmatched_preds = []

    gt_boxes = np.array([g["xyxy"] for g in gts], dtype=np.float32) if gts else np.zeros((0, 4), dtype=np.float32)

    for model_name in model_names:
        preds = preds_by_model.get(model_name, [])

        for p in preds:
            if len(gt_boxes) > 0:
                ious = box_iou_one_to_many(p["xyxy"], gt_boxes)
                best_gt_iou = float(np.max(ious))
                best_gt_idx = int(np.argmax(ious))
            else:
                best_gt_iou = 0.0
                best_gt_idx = -1

            if best_gt_iou < args.missing_gt_iou:
                all_unmatched_preds.append(p)

    clusters = cluster_predictions(all_unmatched_preds, args.cluster_iou)

    for cluster in clusters:
        if len(cluster) < args.min_model_support:
            continue

        summary = summarize_cluster(cluster, class_names, img_area)

        if summary["support_model_count"] < args.min_model_support:
            continue

        xyxy = summary["xyxy"]
        if xyxy is None:
            continue

        missing_label_candidate_rows.append(
            {
                "image": str(img_path),
                "label": str(label_path),
                "candidate_class_id": summary["cls"],
                "candidate_class_name": summary["class_name"],
                "support_model_count": summary["support_model_count"],
                "support_models": summary["support_models"],
                "mean_conf": summary["mean_conf"],
                "max_conf": summary["max_conf"],
                "area_ratio": summary["area_ratio"],
                "x1": float(xyxy[0]),
                "y1": float(xyxy[1]),
                "x2": float(xyxy[2]),
                "y2": float(xyxy[3]),
                "issue_type": "POSSIBLE_MISSING_LABEL",
                "suggestion": "Multiple models predict this region but no GT overlaps. Check whether this is a missed label.",
            }
        )

    return (
        gt_summary_rows,
        model_gt_rows,
        possible_wrong_class_rows,
        possible_box_refinement_rows,
        missing_label_candidate_rows,
    )


# ============================================================
# Visualization
# ============================================================

CLASS_COLORS = {
    0: (0, 255, 255),    # yellow-ish
    1: (0, 255, 0),      # green
    2: (255, 0, 0),      # blue
    3: (0, 128, 255),    # orange
}


def draw_box(img, box, color, text, thickness=2):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))

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


def save_flag_visuals(
    out_dir: Path,
    img_root: Path,
    label_root: Path,
    class_names: List[str],
    gt_summary_rows: List[dict],
    missing_rows: List[dict],
    args,
):
    """
    Save visuals for images that have flagged GT or missing-label candidates.
    """
    ensure_dir(out_dir)

    flagged_by_image = {}

    for r in gt_summary_rows:
        if r["issue_type"] in {
            "GT_NOT_SUPPORTED_BY_MODELS",
            "POSSIBLE_WRONG_CLASS",
            "LOW_MODEL_SUPPORT",
            "POSSIBLE_BOX_GRANULARITY_MISMATCH",
        }:
            flagged_by_image.setdefault(r["image"], {"gt": [], "missing": []})
            flagged_by_image[r["image"]]["gt"].append(r)

    for r in missing_rows:
        flagged_by_image.setdefault(r["image"], {"gt": [], "missing": []})
        flagged_by_image[r["image"]]["missing"].append(r)

    for img_str, group in flagged_by_image.items():
        img_path = Path(img_str)
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        label_path = find_label_for_image(img_path, img_root, label_root)
        gts = read_yolo_label(label_path, img_w, img_h)

        canvas = img.copy()

        # Draw all GT faintly
        for gt in gts:
            cls_id = gt["cls"]
            cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"cls_{cls_id}"
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            draw_box(canvas, gt["xyxy"], color, f"GT:{cls_name}", thickness=1)

        # Draw flagged GT in red
        for r in group["gt"]:
            box = [r["gt_x1"], r["gt_y1"], r["gt_x2"], r["gt_y2"]]
            draw_box(canvas, box, (0, 0, 255), f"{r['issue_type']}:{r['gt_class_name']}", thickness=3)

        # Draw missing label candidates in magenta
        for r in group["missing"]:
            box = [r["x1"], r["y1"], r["x2"], r["y2"]]
            draw_box(
                canvas,
                box,
                (255, 0, 255),
                f"MISS? {r['candidate_class_name']} n={r['support_model_count']}",
                thickness=3,
            )

        rel_name = img_path.name
        out_path = out_dir / f"{safe_name(rel_name)}"
        cv2.imwrite(str(out_path), canvas)


# ============================================================
# Summary
# ============================================================

def build_per_class_summary(gt_rows: List[dict], missing_rows: List[dict], class_names: List[str]):
    summary = {}

    for cls_id, cls_name in enumerate(class_names):
        summary[cls_id] = {
            "class_id": cls_id,
            "class_name": cls_name,
            "gt_count": 0,
            "gt_not_supported": 0,
            "low_model_support": 0,
            "possible_wrong_class": 0,
            "possible_box_mismatch": 0,
            "missing_label_candidates": 0,
        }

    for r in gt_rows:
        cls_id = int(r["gt_class_id"])
        if cls_id not in summary:
            continue

        summary[cls_id]["gt_count"] += 1

        if r["issue_type"] == "GT_NOT_SUPPORTED_BY_MODELS":
            summary[cls_id]["gt_not_supported"] += 1
        elif r["issue_type"] == "LOW_MODEL_SUPPORT":
            summary[cls_id]["low_model_support"] += 1
        elif r["issue_type"] == "POSSIBLE_WRONG_CLASS":
            summary[cls_id]["possible_wrong_class"] += 1
        elif r["issue_type"] == "POSSIBLE_BOX_GRANULARITY_MISMATCH":
            summary[cls_id]["possible_box_mismatch"] += 1

    for r in missing_rows:
        cls_id = int(r["candidate_class_id"])
        if cls_id in summary:
            summary[cls_id]["missing_label_candidates"] += 1

    rows = []
    for cls_id in sorted(summary.keys()):
        s = summary[cls_id]
        gt_count = s["gt_count"]

        s["gt_not_supported_ratio"] = safe_div(s["gt_not_supported"], gt_count)
        s["low_model_support_ratio"] = safe_div(s["low_model_support"], gt_count)
        s["possible_wrong_class_ratio"] = safe_div(s["possible_wrong_class"], gt_count)
        s["possible_box_mismatch_ratio"] = safe_div(s["possible_box_mismatch"], gt_count)

        rows.append(s)

    return rows


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="./data/det_dataset.yaml", help="yolo dataset config")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", default="runs_audit")

    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="name=exp_dir, expects exp_dir/weights/best.pt. Can be repeated.",
    )

    parser.add_argument(
        "--weight",
        action="append",
        default=[
            fr"v8_baseline=.\runs_yolov8n\glcp_stage2_yolo_det\baseline\weights\best.pt",
            fr"v8_sw=.\runs_yolov8n\glcp_stage2_yolo_det\full_no_freeze_use_pos_mask_sw_lesion_sensitive\weights\best.pt",
            fr"v8_leafaux_sw=.\runs_yolov8n\glcp_stage2_yolo_det\leafaux_best_use_pos_mask_sw_lesion_sensitive\weights\best.pt",
            fr"v8_rpd=.\runs_yolov8n\glcp_stage2_yolo_det\full_no_freeze_use_pos_mask_rpd_hybrid_w020\weights\best.pt",
            fr"v8_leafaux_rpd=.\runs_yolov8n\glcp_stage2_yolo_det\leafaux_best_use_pos_mask_rpd_hybrid_w020\weights\best.pt",
            fr"v9_baseline=.\runs_yolov9t\glcp_stage2_yolo_det\baseline\weights\best.pt",
            fr"v9_sw=.\runs_yolov9t\glcp_stage2_yolo_det\full_no_freeze_use_pos_mask_sw_lesion_sensitive\weights\best.pt",
            fr"v9_leafaux_sw=.\runs_yolov9t\glcp_stage2_yolo_det\leafaux_best_use_pos_mask_sw_lesion_sensitive\weights\best.pt",
            fr"v9_rpd=.\runs_yolov9t\glcp_stage2_yolo_det\full_no_freeze_use_pos_mask_rpd_hybrid_w020\weights\best.pt",
            fr"v9_leafaux_rpd=.\runs_yolov9t\glcp_stage2_yolo_det\leafaux_best_use_pos_mask_rpd_hybrid_w020\weights\best.pt",
            fr"v10_baseline=.\runs_yolov10n\glcp_stage2_yolo_det\baseline\weights\best.pt",
            fr"v10_sw=.\runs_yolov10n\glcp_stage2_yolo_det\full_no_freeze_use_pos_mask_sw_lesion_sensitive\weights\best.pt",
            fr"v10_leafaux_sw=.\runs_yolov10n\glcp_stage2_yolo_det\leafaux_best_use_pos_mask_sw_lesion_sensitive\weights\best.pt",
            fr"v10_rpd=.\runs_yolov10n\glcp_stage2_yolo_det\full_no_freeze_use_pos_mask_rpd_hybrid_w020\weights\best.pt",
            fr"v10_leafaux_rpd=.\runs_yolov10n\glcp_stage2_yolo_det\leafaux_best_use_pos_mask_rpd_hybrid_w020\weights\best.pt",
            fr"v11_baseline=.\runs_yolov11n\glcp_stage2_yolo_det\baseline\weights\best.pt",
            fr"v11_sw=.\runs_yolov11n\glcp_stage2_yolo_det\full_no_freeze_use_pos_mask_sw_lesion_sensitive\weights\best.pt",
            fr"v11_leafaux_sw=.\runs_yolov11n\glcp_stage2_yolo_det\leafaux_best_use_pos_mask_sw_lesion_sensitive\weights\best.pt",
            fr"v11_rpd=.\runs_yolov11n\glcp_stage2_yolo_det\full_no_freeze_use_pos_mask_rpd_hybrid_w020\weights\best.pt",
            fr"v11_leafaux_rpd=.\runs_yolov11n\glcp_stage2_yolo_det\leafaux_best_use_pos_mask_rpd_hybrid_w020\weights\best.pt",
        ],
        help="name=path/to/best.pt. Can be repeated.",
    )

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)

    # Matching thresholds
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--overlap-iou", type=float, default=0.30)
    parser.add_argument("--missing-gt-iou", type=float, default=0.30)
    parser.add_argument("--cluster-iou", type=float, default=0.50)
    parser.add_argument("--box-refine-iou", type=float, default=0.65)

    # Consensus thresholds
    parser.add_argument("--min-model-support", type=int, default=2)
    parser.add_argument("--low-support-threshold", type=int, default=1)

    parser.add_argument("--save-visuals", action="store_true")

    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    out_dir = Path(args.out).resolve()
    conf_tag = f"conf{int(round(args.conf * 1000)):03d}"
    seg_path_name = f"{args.split}_label_prediction_audit_{conf_tag}"
    out_dir = out_dir.joinpath(seg_path_name)
    ensure_dir(out_dir)

    dataset_root, img_dir, label_dir, class_names = parse_dataset_yaml(data_yaml, args.split)
    image_paths = collect_images(img_dir)
    model_specs = parse_model_specs(args)
    model_names = [name for name, _ in model_specs]

    print("[INFO] dataset_root:", dataset_root)
    print("[INFO] split:", args.split)
    print("[INFO] img_dir:", img_dir)
    print("[INFO] label_dir:", label_dir)
    print("[INFO] class_names:", class_names)
    print("[INFO] num_images:", len(image_paths))
    print("[INFO] num_models:", len(model_specs))

    print("[INFO] first 5 images and labels:")
    for p in image_paths[:5]:
        lp = find_label_for_image(p, img_dir, label_dir)
        print(f"  IMG: {p}")
        print(f"  LAB: {lp} | exists={lp.exists()}")

    predictions = predict_all_models(model_specs, image_paths, args)

    all_gt_rows = []
    all_model_gt_rows = []
    all_wrong_class_rows = []
    all_box_refinement_rows = []
    all_missing_rows = []

    for idx, img_path in enumerate(image_paths, start=1):
        print(f"[AUDIT] {idx}/{len(image_paths)} {img_path.name}")

        preds_by_model = predictions.get(str(img_path), {})
        (
            gt_rows,
            model_gt_rows,
            wrong_class_rows,
            box_refinement_rows,
            missing_rows,
        ) = analyze_image(
            img_path=img_path,
            img_root=img_dir,
            label_root=label_dir,
            class_names=class_names,
            preds_by_model=preds_by_model,
            model_names=model_names,
            args=args,
        )

        all_gt_rows.extend(gt_rows)
        all_model_gt_rows.extend(model_gt_rows)
        all_wrong_class_rows.extend(wrong_class_rows)
        all_box_refinement_rows.extend(box_refinement_rows)
        all_missing_rows.extend(missing_rows)

    per_class_summary = build_per_class_summary(all_gt_rows, all_missing_rows, class_names)

    # Sort important issue files
    all_gt_rows_sorted = sorted(
        all_gt_rows,
        key=lambda r: (
            r["issue_type"] == "OK_OR_MODEL_DEPENDENT",
            -int(r["wrong_class_support"]) if r["wrong_class_support"] != "" else 0,
            int(r["same_class_support"]),
            r["gt_class_name"],
        ),
    )

    all_missing_rows_sorted = sorted(
        all_missing_rows,
        key=lambda r: (
            -int(r["support_model_count"]),
            -float(r["mean_conf"]),
            r["candidate_class_name"],
        ),
    )

    # Write outputs
    write_csv(out_dir / "gt_level_audit.csv", all_gt_rows_sorted)
    write_csv(out_dir / "model_gt_match_details.csv", all_model_gt_rows)
    write_csv(out_dir / "possible_wrong_class.csv", all_wrong_class_rows)
    write_csv(out_dir / "possible_box_granularity_mismatch.csv", all_box_refinement_rows)
    write_csv(out_dir / "possible_missing_labels.csv", all_missing_rows_sorted)
    write_csv(out_dir / "per_class_label_audit_summary.csv", per_class_summary)

    if args.save_visuals:
        save_flag_visuals(
            out_dir=out_dir / "visual_flagged_cases",
            img_root=img_dir,
            label_root=label_dir,
            class_names=class_names,
            gt_summary_rows=all_gt_rows,
            missing_rows=all_missing_rows,
            args=args,
        )

    manifest = {
        "data_yaml": str(data_yaml),
        "split": args.split,
        "dataset_root": str(dataset_root),
        "img_dir": str(img_dir),
        "label_dir": str(label_dir),
        "class_names": class_names,
        "num_images": len(image_paths),
        "models": [{"name": n, "weight": str(w)} for n, w in model_specs],
        "args": vars(args),
        "outputs": {
            "gt_level_audit": str(out_dir / "gt_level_audit.csv"),
            "model_gt_match_details": str(out_dir / "model_gt_match_details.csv"),
            "possible_wrong_class": str(out_dir / "possible_wrong_class.csv"),
            "possible_box_granularity_mismatch": str(out_dir / "possible_box_granularity_mismatch.csv"),
            "possible_missing_labels": str(out_dir / "possible_missing_labels.csv"),
            "per_class_label_audit_summary": str(out_dir / "per_class_label_audit_summary.csv"),
            "visual_flagged_cases": str(out_dir / "visual_flagged_cases"),
        },
    }

    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (out_dir / "README_label_audit.md").open("w", encoding="utf-8") as f:
        f.write("# Label Audit with Model Predictions\n\n")
        f.write("## Purpose\n\n")
        f.write(
            "This analysis compares GT labels with predictions from multiple trained models. "
            "It is designed to find possible label noise, missing labels, class confusion, "
            "and box granularity mismatch.\n\n"
        )

        f.write("## Key output files\n\n")
        f.write("- `gt_level_audit.csv`: one row per GT box, with model support and issue type.\n")
        f.write("- `model_gt_match_details.csv`: one row per GT box per model.\n")
        f.write("- `possible_wrong_class.csv`: GT boxes where models consistently predict another class.\n")
        f.write("- `possible_box_granularity_mismatch.csv`: GT boxes with same-class predictions but low IoU alignment.\n")
        f.write("- `possible_missing_labels.csv`: regions predicted by multiple models but not covered by GT.\n")
        f.write("- `per_class_label_audit_summary.csv`: class-level summary.\n")
        f.write("- `visual_flagged_cases/`: visual examples if `--save-visuals` is enabled.\n\n")

        f.write("## Interpretation\n\n")
        f.write("- `POSSIBLE_MISSING_LABEL`: check if the predicted region is a missed annotation.\n")
        f.write("- `POSSIBLE_WRONG_CLASS`: check if the GT class should be changed.\n")
        f.write("- `POSSIBLE_BOX_GRANULARITY_MISMATCH`: check if the box is too small, too large, or split inconsistently.\n")
        f.write("- `GT_NOT_SUPPORTED_BY_MODELS`: check whether the GT is ambiguous, tiny, occluded, or wrong.\n")
        f.write("- `LOW_MODEL_SUPPORT`: not necessarily wrong, but should be reviewed for weak or ambiguous labels.\n\n")

    print("\n[DONE]")
    print(f"[OUT] {out_dir}")
    print("[FILES]")
    print("  gt_level_audit.csv")
    print("  model_gt_match_details.csv")
    print("  possible_wrong_class.csv")
    print("  possible_box_granularity_mismatch.csv")
    print("  possible_missing_labels.csv")
    print("  per_class_label_audit_summary.csv")
    if args.save_visuals:
        print("  visual_flagged_cases/")


if __name__ == "__main__":
    main()