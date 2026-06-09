# -*- coding: utf-8 -*-
"""
Evaluate best YOLO models on a selected split and add FP/FN error attribution.

This script is based on eval_best_models_on_split.py and keeps the original outputs:
- overall_metrics.csv
- per_class_metrics.csv
- custom_iou50_per_class.csv
- size_stratified_recall_iou50.csv
- missed_small_targets.csv
- small_target_summary.csv
- image_match_summary.csv
- run_manifest.json

New outputs:
- fp_detail.csv
- fp_summary.csv
- fp_by_class.csv
- fn_detail.csv
- fn_summary.csv
- fn_by_class.csv
- prediction_outcome_detail.csv
- confidence_summary.csv
- confidence_bin_summary.csv
- high_conf_fp_detail.csv
- fn_candidate_conf_summary.csv
- model_comparison_summary.csv
- visual_fp_cases/  if --save-error-visuals is used
- visual_fn_cases/  if --save-error-visuals is used
"""

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

from models import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ============================================================
# Basic utils
# ============================================================

def safe_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^\w\-.]+", "_", name)
    return name[:160] if name else "unknown"


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else float("nan")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[dict]):
    ensure_dir(path.parent)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

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


def xyxy_to_str(box: Optional[np.ndarray]) -> str:
    if box is None:
        return ""
    return ",".join([f"{float(x):.2f}" for x in box])


def get_class_name(class_names: List[str], cls_id) -> str:
    try:
        cls_id = int(cls_id)
        if 0 <= cls_id < len(class_names):
            return class_names[cls_id]
    except Exception:
        pass
    return ""


# ============================================================
# Dataset parsing
# ============================================================

def parse_dataset_yaml(data_yaml: Path, split: str):
    data = load_yaml(data_yaml)

    if "path" not in data:
        raise ValueError("det_dataset.yaml must contain `path`.")

    dataset_root = Path(data["path"])
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()

    if split not in data or data[split] is None:
        raise ValueError(f"dataset yaml must contain `{split}` path.")

    split_value = data[split]
    if isinstance(split_value, list):
        if len(split_value) != 1:
            raise ValueError(
                f"This script currently expects one image directory for split `{split}`, "
                f"but got {len(split_value)} paths: {split_value}"
            )
        split_value = split_value[0]

    test_img_dir = resolve_path(dataset_root, split_value)

    names = data.get("names")
    if names is None:
        raise ValueError("det_dataset.yaml must contain `names`.")

    if isinstance(names, dict):
        class_names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    elif isinstance(names, list):
        class_names = names
    else:
        raise ValueError("Unsupported names format in det_dataset.yaml.")

    # Infer label path by replacing the LAST "images" with "labels"
    parts = list(test_img_dir.parts)

    if "images" not in parts:
        raise ValueError(f"Cannot infer labels dir from test image dir: {test_img_dir}")

    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    test_label_dir = Path(*parts)

    return dataset_root, test_img_dir, test_label_dir, class_names


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
    Return GT boxes:
    {
        "line_id": int,
        "cls": int,
        "xyxy": np.ndarray,
        "area_ratio": float,
        "label_path": str
    }
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
                "label_path": str(label_path),
            }
        )

    return gts


# ============================================================
# IoU and matching
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


def best_iou_to_gts(
    box: np.ndarray,
    gts: List[dict],
    class_filter: Optional[int] = None,
    only_indices: Optional[set] = None,
) -> Tuple[float, int]:
    indices = []

    for i, gt in enumerate(gts):
        if class_filter is not None and gt["cls"] != class_filter:
            continue
        if only_indices is not None and i not in only_indices:
            continue
        indices.append(i)

    if not indices:
        return 0.0, -1

    gt_boxes = np.array([gts[i]["xyxy"] for i in indices], dtype=np.float32)
    ious = box_iou_one_to_many(box, gt_boxes)
    best_local = int(np.argmax(ious))
    best_iou = float(ious[best_local])
    best_gt_idx = indices[best_local]

    return best_iou, best_gt_idx


def best_iou_to_preds(
    box: np.ndarray,
    preds: List[dict],
    class_filter: Optional[int] = None,
    exclude_pred_ids: Optional[set] = None,
) -> Tuple[float, int, Optional[dict]]:
    candidates = []

    for pred in preds:
        if class_filter is not None and pred["cls"] != class_filter:
            continue
        if exclude_pred_ids is not None and pred["pred_id"] in exclude_pred_ids:
            continue
        candidates.append(pred)

    if not candidates:
        return 0.0, -1, None

    boxes = np.array([p["xyxy"] for p in candidates], dtype=np.float32)
    ious = box_iou_one_to_many(box, boxes)
    best_local = int(np.argmax(ious))
    best_iou = float(ious[best_local])
    best_pred = candidates[best_local]

    return best_iou, int(best_pred["pred_id"]), best_pred


# ============================================================
# Ultralytics metric extraction
# ============================================================

def get_attr(obj, name: str, default=None):
    return getattr(obj, name, default)


def extract_ultralytics_metrics(metrics, class_names: List[str], model_name: str):
    box = metrics.box
    nc = len(class_names)

    overall = {
        "model": model_name,
        "precision_overall": float(get_attr(box, "mp", float("nan"))),
        "recall_overall": float(get_attr(box, "mr", float("nan"))),
        "map50_overall": float(get_attr(box, "map50", float("nan"))),
        "map50_90_overall": float("nan"),
        "map50_95_overall": float(get_attr(box, "map", float("nan"))),
        "map75_overall": float(get_attr(box, "map75", float("nan"))),
    }

    all_ap = get_attr(box, "all_ap", None)
    ap_class_index = get_attr(box, "ap_class_index", None)

    if all_ap is not None:
        all_ap_np = np.array(all_ap, dtype=np.float32)
        if all_ap_np.ndim == 2 and all_ap_np.shape[1] >= 9:
            overall["map50_90_overall"] = float(np.nanmean(all_ap_np[:, :9]))

    p_arr = np.array(get_attr(box, "p", []), dtype=np.float32)
    r_arr = np.array(get_attr(box, "r", []), dtype=np.float32)
    ap50_arr = np.array(get_attr(box, "ap50", []), dtype=np.float32)
    ap_arr = np.array(get_attr(box, "ap", []), dtype=np.float32)

    if ap_class_index is None:
        ap_class_index = list(range(len(ap50_arr)))
    else:
        ap_class_index = [int(x) for x in list(ap_class_index)]

    per_class_map = {}

    for local_i, cls_id in enumerate(ap_class_index):
        if cls_id < 0 or cls_id >= nc:
            continue

        map50_90 = float("nan")

        if all_ap is not None:
            all_ap_np = np.array(all_ap, dtype=np.float32)
            if (
                all_ap_np.ndim == 2
                and local_i < all_ap_np.shape[0]
                and all_ap_np.shape[1] >= 9
            ):
                map50_90 = float(np.nanmean(all_ap_np[local_i, :9]))

        per_class_map[cls_id] = {
            "precision": float(p_arr[local_i]) if local_i < len(p_arr) else float("nan"),
            "recall": float(r_arr[local_i]) if local_i < len(r_arr) else float("nan"),
            "ap50": float(ap50_arr[local_i]) if local_i < len(ap50_arr) else float("nan"),
            "ap50_90": map50_90,
            "ap50_95": float(ap_arr[local_i]) if local_i < len(ap_arr) else float("nan"),
        }

    per_class_rows = []

    for cls_id, cls_name in enumerate(class_names):
        row = {
            "model": model_name,
            "class_id": cls_id,
            "class_name": cls_name,
            "precision": float("nan"),
            "recall": float("nan"),
            "ap50": float("nan"),
            "ap50_90": float("nan"),
            "ap50_95": float("nan"),
        }

        if cls_id in per_class_map:
            row.update(per_class_map[cls_id])

        per_class_rows.append(row)

    return overall, per_class_rows


# ============================================================
# Error attribution helpers
# ============================================================

def result_to_predictions(result, nc: int) -> List[dict]:
    preds = []

    if result.boxes is None or len(result.boxes) == 0:
        return preds

    xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    cls = result.boxes.cls.cpu().numpy().astype(np.int64)
    conf = result.boxes.conf.cpu().numpy().astype(np.float32)

    for b, c, cf in zip(xyxy, cls, conf):
        c = int(c)
        if 0 <= c < nc:
            preds.append(
                {
                    "pred_id": len(preds),
                    "cls": c,
                    "xyxy": b,
                    "conf": float(cf),
                }
            )

    preds = sorted(preds, key=lambda x: x["conf"], reverse=True)
    for i, p in enumerate(preds):
        p["pred_id"] = i

    return preds


def fp_diagnostic_hint(fp_type: str) -> str:
    hints = {
        "duplicate_fp": "重复预测框；优先检查同一GT是否被多个框重复预测，或NMS是否过松。",
        "class_error_fp": "位置基本对上GT，但类别预测错误；优先检查类别混淆、类别定义或GT类别是否一致。",
        "localization_fp_low_iou": "预测类别正确，但IoU不足；优先检查框偏移、框大小、病斑边界标注方式。",
        "partial_overlap_fp": "预测框与GT有弱重叠但不足以匹配；可能是定位偏移、邻近目标、背景纹理干扰。",
        "background_or_possible_missing_label_fp": "预测框几乎不重叠任何GT；如果红框确实覆盖病斑，优先检查漏标；否则是真背景误检。",
    }
    return hints.get(fp_type, "")


def fn_diagnostic_hint(fn_type: str) -> str:
    hints = {
        "fn_low_confidence": "有同类候选框且IoU足够，但置信度低于正式阈值；说明模型看到了目标但confidence不够。",
        "fn_class_error_high_conf": "有高置信度候选框覆盖该GT，但类别错误；主要是类别判别问题。",
        "fn_class_error_low_conf": "有低置信度候选框覆盖该GT，但类别错误；类别判别和confidence都不足。",
        "fn_low_iou_localization": "有同类候选框但IoU不够；主要是定位不准或框尺度不合适。",
        "fn_partial_overlap_other_class": "有其它类别候选框与GT弱重叠；可能是类别混淆或邻近纹理干扰。",
        "fn_no_candidate": "没有接近该GT的候选框；说明模型基本没有响应，常见于小目标、遮挡或特征不明显。",
        "fn_matching_conflict": "有同类高置信度候选框但未匹配，可能是贪心匹配冲突或重复框竞争导致。",
    }
    return hints.get(fn_type, "")


def classify_fp(
    pred: dict,
    gts: List[dict],
    matched_gt: set,
    class_names: List[str],
    match_iou: float,
    near_iou: float,
) -> dict:
    pred_cls = int(pred["cls"])

    best_any_iou, best_any_idx = best_iou_to_gts(pred["xyxy"], gts)
    best_same_iou, best_same_idx = best_iou_to_gts(pred["xyxy"], gts, class_filter=pred_cls)

    matched_same_indices = set()
    for i in matched_gt:
        if 0 <= i < len(gts) and int(gts[i]["cls"]) == pred_cls:
            matched_same_indices.add(i)

    best_matched_same_iou, best_matched_same_idx = best_iou_to_gts(
        pred["xyxy"],
        gts,
        class_filter=pred_cls,
        only_indices=matched_same_indices,
    )

    if best_matched_same_idx >= 0 and best_matched_same_iou >= match_iou:
        fp_type = "duplicate_fp"
        reason = "Same-class GT was already matched by a higher-confidence prediction."
    elif best_any_idx >= 0 and best_any_iou >= match_iou and int(gts[best_any_idx]["cls"]) != pred_cls:
        fp_type = "class_error_fp"
        reason = "Prediction overlaps a GT at IoU>=match_iou, but predicted class is different."
    elif best_same_idx >= 0 and near_iou <= best_same_iou < match_iou:
        fp_type = "localization_fp_low_iou"
        reason = "Same-class GT exists nearby, but IoU is below match_iou."
    elif best_any_idx >= 0 and near_iou <= best_any_iou < match_iou:
        fp_type = "partial_overlap_fp"
        reason = "Prediction weakly overlaps a GT but does not satisfy IoU matching."
    else:
        fp_type = "background_or_possible_missing_label_fp"
        reason = "Prediction has little or no overlap with any GT. If it covers a true lesion, it may indicate missing annotation."

    best_any_gt = gts[best_any_idx] if best_any_idx >= 0 else None
    best_same_gt = gts[best_same_idx] if best_same_idx >= 0 else None

    return {
        "fp_type": fp_type,
        "fp_reason": reason,
        "fp_diagnostic_hint": fp_diagnostic_hint(fp_type),
        "best_any_iou": best_any_iou,
        "best_any_gt_index": best_any_idx,
        "best_any_gt_line_id": best_any_gt["line_id"] if best_any_gt is not None else "",
        "best_any_gt_class_id": best_any_gt["cls"] if best_any_gt is not None else "",
        "best_any_gt_class_name": get_class_name(class_names, best_any_gt["cls"]) if best_any_gt is not None else "",
        "best_same_iou": best_same_iou,
        "best_same_gt_index": best_same_idx,
        "best_same_gt_line_id": best_same_gt["line_id"] if best_same_gt is not None else "",
    }


def classify_fn(
    gt: dict,
    all_preds: List[dict],
    matched_pred_ids: set,
    class_names: List[str],
    match_iou: float,
    near_iou: float,
    pred_conf: float,
) -> dict:
    gt_cls = int(gt["cls"])

    best_any_iou, best_any_pred_id, best_any_pred = best_iou_to_preds(
        gt["xyxy"],
        all_preds,
        class_filter=None,
        exclude_pred_ids=matched_pred_ids,
    )

    best_same_iou, best_same_pred_id, best_same_pred = best_iou_to_preds(
        gt["xyxy"],
        all_preds,
        class_filter=gt_cls,
        exclude_pred_ids=matched_pred_ids,
    )

    if best_same_pred is not None and best_same_iou >= match_iou:
        if float(best_same_pred["conf"]) < pred_conf:
            fn_type = "fn_low_confidence"
            reason = "Same-class candidate has enough IoU but confidence is below pred_conf."
        else:
            fn_type = "fn_matching_conflict"
            reason = "Same-class high-confidence candidate exists but was not matched, likely due to greedy matching conflict."
    elif (
        best_any_pred is not None
        and best_any_iou >= match_iou
        and int(best_any_pred["cls"]) != gt_cls
    ):
        if float(best_any_pred["conf"]) >= pred_conf:
            fn_type = "fn_class_error_high_conf"
            reason = "Wrong-class high-confidence prediction overlaps this GT at IoU>=match_iou."
        else:
            fn_type = "fn_class_error_low_conf"
            reason = "Wrong-class low-confidence prediction overlaps this GT at IoU>=match_iou."
    elif best_same_pred is not None and near_iou <= best_same_iou < match_iou:
        fn_type = "fn_low_iou_localization"
        reason = "Same-class candidate exists nearby, but IoU is below match_iou."
    elif best_any_pred is not None and near_iou <= best_any_iou < match_iou:
        fn_type = "fn_partial_overlap_other_class"
        reason = "A candidate overlaps weakly with this GT but not enough for matching."
    else:
        fn_type = "fn_no_candidate"
        reason = "No candidate is close enough to this GT."

    return {
        "fn_type": fn_type,
        "fn_reason": reason,
        "fn_diagnostic_hint": fn_diagnostic_hint(fn_type),
        "best_any_iou": best_any_iou,
        "best_any_pred_id": best_any_pred_id if best_any_pred is not None else "",
        "best_any_pred_class_id": best_any_pred["cls"] if best_any_pred is not None else "",
        "best_any_pred_class_name": get_class_name(class_names, best_any_pred["cls"]) if best_any_pred is not None else "",
        "best_any_pred_conf": best_any_pred["conf"] if best_any_pred is not None else "",
        "best_same_iou": best_same_iou,
        "best_same_pred_id": best_same_pred_id if best_same_pred is not None else "",
        "best_same_pred_conf": best_same_pred["conf"] if best_same_pred is not None else "",
    }


def aggregate_count_rows(rows: List[dict], group_keys: List[str], count_name: str = "count") -> List[dict]:
    counter = {}

    for row in rows:
        key = tuple(row.get(k, "") for k in group_keys)
        counter[key] = counter.get(key, 0) + 1

    out = []
    for key, cnt in sorted(counter.items(), key=lambda x: tuple(str(v) for v in x[0])):
        item = {k: key[i] for i, k in enumerate(group_keys)}
        item[count_name] = cnt
        out.append(item)

    return out


# ============================================================
# Confidence/ranking diagnostic helpers
# ============================================================

def area_to_size_bin(area_ratio: float, args) -> str:
    if area_ratio < args.small_area_ratio:
        return "small"
    if area_ratio < args.medium_area_ratio:
        return "medium"
    return "large"


def as_float_or_nan(v) -> float:
    try:
        if v == "" or v is None:
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def summarize_numeric(values: List[float], prefix: str) -> dict:
    vals = []
    for v in values:
        try:
            fv = float(v)
        except Exception:
            continue
        if not math.isnan(fv):
            vals.append(fv)

    if not vals:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_p75": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_max": float("nan"),
        }

    arr = np.array(vals, dtype=np.float32)
    return {
        f"{prefix}_count": int(len(arr)),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p75": float(np.percentile(arr, 75)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_max": float(np.max(arr)),
    }


def make_confidence_summary(pred_rows: List[dict], topk_values: List[int]) -> List[dict]:
    by_model = {}
    for row in pred_rows:
        by_model.setdefault(row.get("model", ""), []).append(row)

    out = []
    for model_name, rows in sorted(by_model.items()):
        tp_rows = [r for r in rows if r.get("pred_outcome") == "tp"]
        fp_rows = [r for r in rows if r.get("pred_outcome") == "fp"]
        total = len(rows)
        tp_n = len(tp_rows)
        fp_n = len(fp_rows)

        row = {
            "model": model_name,
            "eval_pred_count": total,
            "tp_pred_count": tp_n,
            "fp_pred_count": fp_n,
            "custom_precision_from_pred_rows": safe_div(tp_n, total),
            "fp_rate_from_pred_rows": safe_div(fp_n, total),
        }
        row.update(summarize_numeric([r.get("pred_conf") for r in tp_rows], "tp_conf"))
        row.update(summarize_numeric([r.get("pred_conf") for r in fp_rows], "fp_conf"))
        row["tp_minus_fp_conf_mean"] = row["tp_conf_mean"] - row["fp_conf_mean"]
        row["tp_minus_fp_conf_median"] = row["tp_conf_median"] - row["fp_conf_median"]
        row.update(summarize_numeric([r.get("matched_iou") for r in tp_rows], "tp_iou"))
        row.update(summarize_numeric([r.get("best_same_iou") for r in fp_rows], "fp_best_same_iou"))
        row.update(summarize_numeric([r.get("best_any_iou") for r in fp_rows], "fp_best_any_iou"))

        sorted_rows = sorted(rows, key=lambda x: as_float_or_nan(x.get("pred_conf")), reverse=True)
        for k in topk_values:
            top_rows = sorted_rows[: min(k, len(sorted_rows))]
            top_tp = sum(1 for r in top_rows if r.get("pred_outcome") == "tp")
            top_fp = sum(1 for r in top_rows if r.get("pred_outcome") == "fp")
            row[f"top{k}_available"] = len(top_rows)
            row[f"top{k}_tp"] = top_tp
            row[f"top{k}_fp"] = top_fp
            row[f"top{k}_precision"] = safe_div(top_tp, top_tp + top_fp)
            row[f"top{k}_fp_rate"] = safe_div(top_fp, top_tp + top_fp)

        out.append(row)

    return out


def make_confidence_bin_rows(pred_rows: List[dict], bins: List[float]) -> List[dict]:
    by_key = {}
    for row in pred_rows:
        conf = as_float_or_nan(row.get("pred_conf"))
        if math.isnan(conf):
            continue

        bin_name = None
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            if lo <= conf < hi:
                bin_name = f"[{lo:.2f},{hi:.2f})"
                break
        if bin_name is None and conf >= bins[-1]:
            bin_name = f">={bins[-1]:.2f}"
        if bin_name is None:
            bin_name = f"<{bins[0]:.2f}"

        key = (row.get("model", ""), bin_name)
        if key not in by_key:
            by_key[key] = {"tp": 0, "fp": 0, "conf_values": []}
        if row.get("pred_outcome") == "tp":
            by_key[key]["tp"] += 1
        elif row.get("pred_outcome") == "fp":
            by_key[key]["fp"] += 1
        by_key[key]["conf_values"].append(conf)

    out = []
    for (model_name, bin_name), item in sorted(by_key.items()):
        tp_n = item["tp"]
        fp_n = item["fp"]
        conf_vals = item["conf_values"]
        out.append({
            "model": model_name,
            "confidence_bin": bin_name,
            "tp_count": tp_n,
            "fp_count": fp_n,
            "total_count": tp_n + fp_n,
            "precision_in_bin": safe_div(tp_n, tp_n + fp_n),
            "fp_rate_in_bin": safe_div(fp_n, tp_n + fp_n),
            "mean_conf_in_bin": float(np.mean(conf_vals)) if conf_vals else float("nan"),
        })
    return out


def make_top_fp_rows(fp_rows: List[dict], max_rows_per_model: int) -> List[dict]:
    by_model = {}
    for row in fp_rows:
        by_model.setdefault(row.get("model", ""), []).append(row)

    out = []
    for model_name, rows in sorted(by_model.items()):
        rows_sorted = sorted(rows, key=lambda x: as_float_or_nan(x.get("pred_conf")), reverse=True)
        for rank, row in enumerate(rows_sorted[:max_rows_per_model], start=1):
            new_row = dict(row)
            new_row["rank_by_fp_conf_within_model"] = rank
            out.append(new_row)
    return out


def make_fn_candidate_conf_summary(fn_rows: List[dict]) -> List[dict]:
    by_key = {}
    for row in fn_rows:
        key = (row.get("model", ""), row.get("fn_type", ""))
        if key not in by_key:
            by_key[key] = {"best_any_conf": [], "best_same_conf": [], "best_any_iou": [], "best_same_iou": [], "count": 0}
        by_key[key]["count"] += 1
        by_key[key]["best_any_conf"].append(as_float_or_nan(row.get("best_any_pred_conf")))
        by_key[key]["best_same_conf"].append(as_float_or_nan(row.get("best_same_pred_conf")))
        by_key[key]["best_any_iou"].append(as_float_or_nan(row.get("best_any_iou")))
        by_key[key]["best_same_iou"].append(as_float_or_nan(row.get("best_same_iou")))

    out = []
    for (model_name, fn_type), item in sorted(by_key.items()):
        row = {"model": model_name, "fn_type": fn_type, "count": item["count"]}
        row.update(summarize_numeric(item["best_any_conf"], "best_any_pred_conf"))
        row.update(summarize_numeric(item["best_same_conf"], "best_same_pred_conf"))
        row.update(summarize_numeric(item["best_any_iou"], "best_any_iou"))
        row.update(summarize_numeric(item["best_same_iou"], "best_same_iou"))
        out.append(row)
    return out


def make_model_comparison_summary(
    overall_rows: List[dict],
    custom_rows: List[dict],
    fp_rows: List[dict],
    fn_rows: List[dict],
    size_rows: List[dict],
    confidence_rows: List[dict],
    baseline_name_patterns: List[str],
    reference_name_patterns: List[str],
) -> List[dict]:
    overall_by_model = {r.get("model", ""): r for r in overall_rows}

    custom_totals = {}
    for r in custom_rows:
        m = r.get("model", "")
        item = custom_totals.setdefault(m, {"tp": 0, "fp": 0, "fn": 0})
        item["tp"] += int(r.get("tp_iou50", 0) or 0)
        item["fp"] += int(r.get("fp_iou50", 0) or 0)
        item["fn"] += int(r.get("fn_iou50", 0) or 0)

    fp_type_counts = {}
    for r in fp_rows:
        m = r.get("model", "")
        t = r.get("fp_type", "")
        fp_type_counts.setdefault(m, {})[t] = fp_type_counts.setdefault(m, {}).get(t, 0) + 1

    fn_type_counts = {}
    for r in fn_rows:
        m = r.get("model", "")
        t = r.get("fn_type", "")
        fn_type_counts.setdefault(m, {})[t] = fn_type_counts.setdefault(m, {}).get(t, 0) + 1

    size_totals = {}
    for r in size_rows:
        m = r.get("model", "")
        b = r.get("size_bin", "")
        item = size_totals.setdefault(m, {}).setdefault(b, {"gt": 0, "det": 0})
        item["gt"] += int(r.get("gt_count", 0) or 0)
        item["det"] += int(r.get("detected_count_iou50", 0) or 0)

    conf_by_model = {r.get("model", ""): r for r in confidence_rows}

    def find_model(patterns):
        lower_map = {m.lower(): m for m in overall_by_model.keys()}
        for pat in patterns:
            pat = pat.lower()
            for ml, orig in lower_map.items():
                if pat and pat in ml:
                    return orig
        return None

    baseline_model = find_model(baseline_name_patterns)
    reference_model = find_model(reference_name_patterns)

    def get_metric(m, key):
        if not m or m not in overall_by_model:
            return float("nan")
        return as_float_or_nan(overall_by_model[m].get(key))

    out = []
    for model_name in sorted(overall_by_model.keys()):
        overall = overall_by_model[model_name]
        c = custom_totals.get(model_name, {"tp": 0, "fp": 0, "fn": 0})
        fp_counts = fp_type_counts.get(model_name, {})
        fn_counts = fn_type_counts.get(model_name, {})
        conf = conf_by_model.get(model_name, {})

        row = {
            "model": model_name,
            "baseline_model_used_for_delta": baseline_model or "",
            "reference_model_used_for_delta": reference_model or "",
            "precision_overall": overall.get("precision_overall"),
            "recall_overall": overall.get("recall_overall"),
            "map50_overall": overall.get("map50_overall"),
            "map50_95_overall": overall.get("map50_95_overall"),
            "custom_tp_iou50_total": c["tp"],
            "custom_fp_iou50_total": c["fp"],
            "custom_fn_iou50_total": c["fn"],
            "custom_precision_iou50_total": safe_div(c["tp"], c["tp"] + c["fp"]),
            "custom_recall_iou50_total": safe_div(c["tp"], c["tp"] + c["fn"]),
            "fp_background_or_possible_missing_label_fp": fp_counts.get("background_or_possible_missing_label_fp", 0),
            "fp_localization_fp_low_iou": fp_counts.get("localization_fp_low_iou", 0),
            "fp_partial_overlap_fp": fp_counts.get("partial_overlap_fp", 0),
            "fp_duplicate_fp": fp_counts.get("duplicate_fp", 0),
            "fp_class_error_fp": fp_counts.get("class_error_fp", 0),
            "fn_no_candidate": fn_counts.get("fn_no_candidate", 0),
            "fn_low_iou_localization": fn_counts.get("fn_low_iou_localization", 0),
            "fn_low_confidence": fn_counts.get("fn_low_confidence", 0),
            "fn_matching_conflict": fn_counts.get("fn_matching_conflict", 0),
            "fn_class_error_high_conf": fn_counts.get("fn_class_error_high_conf", 0),
            "fn_class_error_low_conf": fn_counts.get("fn_class_error_low_conf", 0),
            "tp_conf_median": conf.get("tp_conf_median", float("nan")),
            "fp_conf_median": conf.get("fp_conf_median", float("nan")),
            "tp_minus_fp_conf_median": conf.get("tp_minus_fp_conf_median", float("nan")),
            "top100_precision": conf.get("top100_precision", float("nan")),
            "top100_fp": conf.get("top100_fp", float("nan")),
        }

        for bin_name in ["small", "medium", "large"]:
            item = size_totals.get(model_name, {}).get(bin_name, {"gt": 0, "det": 0})
            row[f"{bin_name}_gt_total"] = item["gt"]
            row[f"{bin_name}_detected_iou50_total"] = item["det"]
            row[f"{bin_name}_recall_iou50_total"] = safe_div(item["det"], item["gt"])

        if baseline_model:
            row["delta_precision_vs_baseline"] = as_float_or_nan(row["precision_overall"]) - get_metric(baseline_model, "precision_overall")
            row["delta_recall_vs_baseline"] = as_float_or_nan(row["recall_overall"]) - get_metric(baseline_model, "recall_overall")
            row["delta_map50_vs_baseline"] = as_float_or_nan(row["map50_overall"]) - get_metric(baseline_model, "map50_overall")
            row["delta_map50_95_vs_baseline"] = as_float_or_nan(row["map50_95_overall"]) - get_metric(baseline_model, "map50_95_overall")
        if reference_model:
            row["delta_precision_vs_reference"] = as_float_or_nan(row["precision_overall"]) - get_metric(reference_model, "precision_overall")
            row["delta_recall_vs_reference"] = as_float_or_nan(row["recall_overall"]) - get_metric(reference_model, "recall_overall")
            row["delta_map50_vs_reference"] = as_float_or_nan(row["map50_overall"]) - get_metric(reference_model, "map50_overall")
            row["delta_map50_95_vs_reference"] = as_float_or_nan(row["map50_95_overall"]) - get_metric(reference_model, "map50_95_overall")

        out.append(row)

    return out


# ============================================================
# Visualization helpers
# ============================================================

def draw_box(img, box, color, text=None, thickness=2):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        t = 1
        (tw, th), base = cv2.getTextSize(text, font, scale, t)
        y_text = max(0, y1 - th - base - 3)
        cv2.rectangle(img, (x1, y_text), (x1 + tw + 4, y_text + th + base + 4), color, -1)
        cv2.putText(img, text, (x1 + 2, y_text + th + 1), font, scale, (255, 255, 255), t, cv2.LINE_AA)


def should_save_visual(visual_counter: dict, model_name: str, err_type: str, args) -> bool:
    if not args.save_error_visuals:
        return False

    model_key = (model_name, "__total__")
    type_key = (model_name, err_type)

    total = visual_counter.get(model_key, 0)
    type_total = visual_counter.get(type_key, 0)

    if total >= args.max_error_visuals_per_model:
        return False
    if type_total >= args.max_error_visuals_per_type:
        return False

    visual_counter[model_key] = total + 1
    visual_counter[type_key] = type_total + 1
    return True


def save_fp_visual(
    img_path: Path,
    out_dir: Path,
    model_name: str,
    fp_type: str,
    pred: dict,
    gts: List[dict],
    class_names: List[str],
    fp_info: dict,
    visual_counter: dict,
    args,
):
    if not should_save_visual(visual_counter, model_name, "FP_" + fp_type, args):
        return

    img = cv2.imread(str(img_path))
    if img is None:
        return

    # GT: green
    for i, gt in enumerate(gts):
        label = f"GT#{i} {get_class_name(class_names, gt['cls'])}"
        draw_box(img, gt["xyxy"], (0, 180, 0), label, 2)

    # Best related GT: yellow
    best_gt_idx = fp_info.get("best_any_gt_index", -1)
    try:
        best_gt_idx = int(best_gt_idx)
    except Exception:
        best_gt_idx = -1

    if 0 <= best_gt_idx < len(gts):
        gt = gts[best_gt_idx]
        label = f"BEST_GT IoU={float(fp_info.get('best_any_iou', 0.0)):.2f}"
        draw_box(img, gt["xyxy"], (0, 220, 220), label, 3)

    # Current FP: red
    pred_label = f"FP {get_class_name(class_names, pred['cls'])} {pred['conf']:.2f}"
    draw_box(img, pred["xyxy"], (0, 0, 255), pred_label, 3)

    save_dir = out_dir / "visual_fp_cases" / safe_name(model_name) / safe_name(fp_type)
    ensure_dir(save_dir)

    stem = safe_name(img_path.stem)
    save_path = save_dir / f"{stem}_pred{pred['pred_id']}.jpg"
    cv2.imwrite(str(save_path), img)


def save_fn_visual(
    img_path: Path,
    out_dir: Path,
    model_name: str,
    fn_type: str,
    gt_idx: int,
    gt: dict,
    all_preds: List[dict],
    class_names: List[str],
    fn_info: dict,
    visual_counter: dict,
    args,
):
    if not should_save_visual(visual_counter, model_name, "FN_" + fn_type, args):
        return

    img = cv2.imread(str(img_path))
    if img is None:
        return

    # Candidate predictions: light blue
    for pred in all_preds:
        if pred["conf"] < args.pred_conf:
            continue
        label = f"P {get_class_name(class_names, pred['cls'])} {pred['conf']:.2f}"
        draw_box(img, pred["xyxy"], (255, 180, 0), label, 1)

    # Current FN GT: orange
    gt_label = f"FN_GT#{gt_idx} {get_class_name(class_names, gt['cls'])}"
    draw_box(img, gt["xyxy"], (0, 140, 255), gt_label, 3)

    # Best related pred: purple
    best_pred_id = fn_info.get("best_any_pred_id", "")
    try:
        best_pred_id = int(best_pred_id)
    except Exception:
        best_pred_id = -1

    for pred in all_preds:
        if int(pred["pred_id"]) == best_pred_id:
            label = f"BEST_P IoU={float(fn_info.get('best_any_iou', 0.0)):.2f} {pred['conf']:.2f}"
            draw_box(img, pred["xyxy"], (180, 0, 180), label, 3)
            break

    save_dir = out_dir / "visual_fn_cases" / safe_name(model_name) / safe_name(fn_type)
    ensure_dir(save_dir)

    stem = safe_name(img_path.stem)
    save_path = save_dir / f"{stem}_gt{gt_idx}_line{gt['line_id']}.jpg"
    cv2.imwrite(str(save_path), img)


# ============================================================
# Custom IoU50 matching, small target analysis, and FP/FN attribution
# ============================================================

def parse_small_target_classes(class_names: List[str], args):
    small_target_names = [
        x.strip()
        for x in args.small_target_class_names.split(",")
        if x.strip()
    ]

    small_target_cls_ids = set()

    for i, name in enumerate(class_names):
        if name in small_target_names:
            small_target_cls_ids.add(i)

    missing_names = [
        name for name in small_target_names
        if name not in class_names
    ]

    if missing_names:
        print(f"[WARN] small target class names not found in dataset names: {missing_names}")

    return small_target_names, small_target_cls_ids


def match_predictions_iou50(
    model: YOLO,
    model_name: str,
    image_paths: List[Path],
    img_root: Path,
    label_root: Path,
    class_names: List[str],
    out_dir: Path,
    args,
):
    """
    Custom IoU50 matching + FP/FN attribution.

    Formal TP/FP/FN statistics use predictions with conf >= args.pred_conf.
    Error diagnosis additionally uses low-confidence candidates from args.analysis_conf.
    """
    nc = len(class_names)

    tp = np.zeros(nc, dtype=np.int64)
    fp = np.zeros(nc, dtype=np.int64)
    fn = np.zeros(nc, dtype=np.int64)
    gt_count = np.zeros(nc, dtype=np.int64)

    size_stats = {
        cls_id: {
            "small_gt": 0,
            "small_detected": 0,
            "medium_gt": 0,
            "medium_detected": 0,
            "large_gt": 0,
            "large_detected": 0,
        }
        for cls_id in range(nc)
    }

    _, small_target_cls_ids = parse_small_target_classes(class_names, args)

    small_target_summary = {
        cls_id: {
            "small_gt": 0,
            "small_detected": 0,
            "small_missed": 0,
        }
        for cls_id in small_target_cls_ids
    }

    missed_small_target_rows = []
    image_match_rows = []

    fp_detail_rows = []
    fn_detail_rows = []
    prediction_detail_rows = []
    visual_counter = {}

    pred_save_dir = out_dir / "predict_images" / model_name if args.save_pred_images else None

    # Use analysis_conf to keep low-confidence candidates for FN diagnosis.
    predict_conf = min(args.pred_conf, args.analysis_conf)

    predict_kwargs = dict(
        source=[str(p) for p in image_paths],
        imgsz=args.imgsz,
        conf=predict_conf,
        iou=args.pred_iou,
        device=args.device,
        save=args.save_pred_images,
        exist_ok=True,
        verbose=False,
        stream=True,
    )

    if pred_save_dir is not None:
        predict_kwargs.update(
            dict(
                project=str(pred_save_dir.parent.resolve()),
                name=pred_save_dir.name,
            )
        )

    results_iter = model.predict(**predict_kwargs)

    # Convert to list to guarantee stable zip alignment
    results = list(results_iter)

    if len(results) != len(image_paths):
        print(
            f"[WARN] Prediction result count mismatch for {model_name}: "
            f"results={len(results)}, images={len(image_paths)}"
        )

    for img_path, result in zip(image_paths, results):
        img_path = Path(img_path)

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] cannot read image: {img_path}")
            continue

        img_h, img_w = img.shape[:2]

        label_path = find_label_for_image(img_path, img_root, label_root)
        gts = read_yolo_label(label_path, img_w, img_h)

        if not label_path.exists():
            print(f"[WARN] label not found: {label_path}")

        gt_count_img = len(gts)

        for gt in gts:
            if 0 <= gt["cls"] < nc:
                gt_count[gt["cls"]] += 1

        all_pred_boxes = result_to_predictions(result, nc)
        eval_pred_boxes = [p for p in all_pred_boxes if p["conf"] >= args.pred_conf]

        matched_gt = set()
        matched_pred_ids = set()
        matched_pred_info = {}
        detected_gt_indices = set()
        fp_pred_items = []

        # Greedy class-aware matching at IoU50 using formal eval predictions only
        for pred in eval_pred_boxes:
            pred_cls = pred["cls"]

            candidate_indices = [
                i for i, gt in enumerate(gts)
                if gt["cls"] == pred_cls and i not in matched_gt
            ]

            if not candidate_indices:
                fp[pred_cls] += 1
                fp_pred_items.append(pred)
                continue

            candidate_boxes = np.array(
                [gts[i]["xyxy"] for i in candidate_indices],
                dtype=np.float32,
            )

            ious = box_iou_one_to_many(pred["xyxy"], candidate_boxes)

            best_local = int(np.argmax(ious)) if len(ious) > 0 else -1
            best_iou = float(ious[best_local]) if best_local >= 0 else 0.0

            if best_iou >= args.match_iou:
                gt_idx = candidate_indices[best_local]
                matched_gt.add(gt_idx)
                matched_pred_ids.add(pred["pred_id"])
                matched_pred_info[pred["pred_id"]] = {
                    "matched_gt_index": gt_idx,
                    "matched_gt_line_id": gts[gt_idx]["line_id"],
                    "matched_gt_class_id": gts[gt_idx]["cls"],
                    "matched_gt_class_name": get_class_name(class_names, gts[gt_idx]["cls"]),
                    "matched_gt_area_ratio": gts[gt_idx]["area_ratio"],
                    "matched_gt_size_bin": area_to_size_bin(gts[gt_idx]["area_ratio"], args),
                    "matched_iou": best_iou,
                    "matched_gt_xyxy": xyxy_to_str(gts[gt_idx]["xyxy"]),
                }
                detected_gt_indices.add(gt_idx)
                tp[pred_cls] += 1
            else:
                fp[pred_cls] += 1
                fp_pred_items.append(pred)

        # Prediction-level outcome rows for matched TP predictions.
        for pred in eval_pred_boxes:
            if pred["pred_id"] not in matched_pred_ids:
                continue
            info = matched_pred_info.get(pred["pred_id"], {})
            prediction_detail_rows.append(
                {
                    "model": model_name,
                    "image": str(img_path),
                    "label": str(label_path),
                    "label_exists": int(label_path.exists()),
                    "pred_id": pred["pred_id"],
                    "pred_outcome": "tp",
                    "pred_class_id": pred["cls"],
                    "pred_class_name": get_class_name(class_names, pred["cls"]),
                    "pred_conf": pred["conf"],
                    "pred_xyxy": xyxy_to_str(pred["xyxy"]),
                    "match_iou_threshold": args.match_iou,
                    "near_iou_threshold": args.near_iou,
                    "matched_iou": info.get("matched_iou", ""),
                    "matched_gt_index": info.get("matched_gt_index", ""),
                    "matched_gt_line_id": info.get("matched_gt_line_id", ""),
                    "matched_gt_class_id": info.get("matched_gt_class_id", ""),
                    "matched_gt_class_name": info.get("matched_gt_class_name", ""),
                    "matched_gt_area_ratio": info.get("matched_gt_area_ratio", ""),
                    "matched_gt_size_bin": info.get("matched_gt_size_bin", ""),
                    "matched_gt_xyxy": info.get("matched_gt_xyxy", ""),
                    "fp_type": "",
                    "best_any_iou": "",
                    "best_same_iou": "",
                }
            )

        # FP attribution
        for pred in fp_pred_items:
            info = classify_fp(
                pred=pred,
                gts=gts,
                matched_gt=matched_gt,
                class_names=class_names,
                match_iou=args.match_iou,
                near_iou=args.near_iou,
            )

            fp_row = {
                "model": model_name,
                "image": str(img_path),
                "label": str(label_path),
                "label_exists": int(label_path.exists()),
                "pred_id": pred["pred_id"],
                "pred_class_id": pred["cls"],
                "pred_class_name": get_class_name(class_names, pred["cls"]),
                "pred_conf": pred["conf"],
                "pred_xyxy": xyxy_to_str(pred["xyxy"]),
                "match_iou_threshold": args.match_iou,
                "near_iou_threshold": args.near_iou,
            }
            fp_row.update(info)
            fp_detail_rows.append(fp_row)

            prediction_detail_rows.append(
                {
                    "model": model_name,
                    "image": str(img_path),
                    "label": str(label_path),
                    "label_exists": int(label_path.exists()),
                    "pred_id": pred["pred_id"],
                    "pred_outcome": "fp",
                    "pred_class_id": pred["cls"],
                    "pred_class_name": get_class_name(class_names, pred["cls"]),
                    "pred_conf": pred["conf"],
                    "pred_xyxy": xyxy_to_str(pred["xyxy"]),
                    "match_iou_threshold": args.match_iou,
                    "near_iou_threshold": args.near_iou,
                    "matched_iou": "",
                    "matched_gt_index": "",
                    "matched_gt_line_id": "",
                    "matched_gt_class_id": "",
                    "matched_gt_class_name": "",
                    "matched_gt_area_ratio": "",
                    "matched_gt_size_bin": "",
                    "matched_gt_xyxy": "",
                    "fp_type": info.get("fp_type", ""),
                    "best_any_iou": info.get("best_any_iou", ""),
                    "best_same_iou": info.get("best_same_iou", ""),
                    "best_any_gt_index": info.get("best_any_gt_index", ""),
                    "best_any_gt_line_id": info.get("best_any_gt_line_id", ""),
                    "best_any_gt_class_id": info.get("best_any_gt_class_id", ""),
                    "best_any_gt_class_name": info.get("best_any_gt_class_name", ""),
                }
            )

            save_fp_visual(
                img_path=img_path,
                out_dir=out_dir,
                model_name=model_name,
                fp_type=info["fp_type"],
                pred=pred,
                gts=gts,
                class_names=class_names,
                fp_info=info,
                visual_counter=visual_counter,
                args=args,
            )

        # FN + size-stratified analysis
        for i, gt in enumerate(gts):
            cls_id = gt["cls"]

            if not (0 <= cls_id < nc):
                continue

            is_missed = i not in matched_gt

            if is_missed:
                fn[cls_id] += 1

                fn_info = classify_fn(
                    gt=gt,
                    all_preds=all_pred_boxes,
                    matched_pred_ids=matched_pred_ids,
                    class_names=class_names,
                    match_iou=args.match_iou,
                    near_iou=args.near_iou,
                    pred_conf=args.pred_conf,
                )

                fn_row = {
                    "model": model_name,
                    "image": str(img_path),
                    "label": str(label_path),
                    "label_exists": int(label_path.exists()),
                    "gt_index": i,
                    "gt_line_id": gt["line_id"],
                    "gt_class_id": cls_id,
                    "gt_class_name": get_class_name(class_names, cls_id),
                    "gt_area_ratio": gt["area_ratio"],
                    "gt_xyxy": xyxy_to_str(gt["xyxy"]),
                    "pred_conf_threshold": args.pred_conf,
                    "analysis_conf_threshold": args.analysis_conf,
                    "match_iou_threshold": args.match_iou,
                    "near_iou_threshold": args.near_iou,
                }
                fn_row.update(fn_info)
                fn_detail_rows.append(fn_row)

                save_fn_visual(
                    img_path=img_path,
                    out_dir=out_dir,
                    model_name=model_name,
                    fn_type=fn_info["fn_type"],
                    gt_idx=i,
                    gt=gt,
                    all_preds=all_pred_boxes,
                    class_names=class_names,
                    fn_info=fn_info,
                    visual_counter=visual_counter,
                    args=args,
                )

            area = gt["area_ratio"]

            if area < args.small_area_ratio:
                bin_name = "small"
            elif area < args.medium_area_ratio:
                bin_name = "medium"
            else:
                bin_name = "large"

            size_stats[cls_id][f"{bin_name}_gt"] += 1

            if i in detected_gt_indices:
                size_stats[cls_id][f"{bin_name}_detected"] += 1

            # Small target analysis for selected classes
            if cls_id in small_target_cls_ids and bin_name == "small":
                small_target_summary[cls_id]["small_gt"] += 1

                if i in detected_gt_indices:
                    small_target_summary[cls_id]["small_detected"] += 1
                else:
                    small_target_summary[cls_id]["small_missed"] += 1

                    missed_small_target_rows.append(
                        {
                            "model": model_name,
                            "image": str(img_path),
                            "label": str(label_path),
                            "gt_line_id": gt["line_id"],
                            "class_id": cls_id,
                            "class_name": class_names[cls_id],
                            "area_ratio": gt["area_ratio"],
                            "x1": float(gt["xyxy"][0]),
                            "y1": float(gt["xyxy"][1]),
                            "x2": float(gt["xyxy"][2]),
                            "y2": float(gt["xyxy"][3]),
                        }
                    )

        image_match_rows.append(
            {
                "model": model_name,
                "image": str(img_path),
                "label": str(label_path),
                "label_exists": int(label_path.exists()),
                "gt_count": gt_count_img,
                "pred_count_eval_conf": len(eval_pred_boxes),
                "pred_count_analysis_conf": len(all_pred_boxes),
                "matched_gt_count": len(matched_gt),
                "missed_gt_count": gt_count_img - len(matched_gt),
                "fp_count": len(fp_pred_items),
            }
        )

    per_class_iou50_rows = []

    for cls_id, cls_name in enumerate(class_names):
        precision = safe_div(tp[cls_id], tp[cls_id] + fp[cls_id])
        recall = safe_div(tp[cls_id], tp[cls_id] + fn[cls_id])

        per_class_iou50_rows.append(
            {
                "model": model_name,
                "class_id": cls_id,
                "class_name": cls_name,
                "gt_count": int(gt_count[cls_id]),
                "tp_iou50": int(tp[cls_id]),
                "fp_iou50": int(fp[cls_id]),
                "fn_iou50": int(fn[cls_id]),
                "precision_iou50_custom": precision,
                "recall_iou50_custom": recall,
            }
        )

    size_recall_rows = []

    for cls_id, cls_name in enumerate(class_names):
        s = size_stats[cls_id]

        for bin_name in ["small", "medium", "large"]:
            gt_n = s[f"{bin_name}_gt"]
            det_n = s[f"{bin_name}_detected"]

            size_recall_rows.append(
                {
                    "model": model_name,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "size_bin": bin_name,
                    "gt_count": int(gt_n),
                    "detected_count_iou50": int(det_n),
                    "recall_iou50": safe_div(det_n, gt_n),
                    "small_area_ratio_threshold": args.small_area_ratio,
                    "medium_area_ratio_threshold": args.medium_area_ratio,
                }
            )

    small_target_summary_rows = []

    for cls_id in sorted(small_target_summary.keys()):
        s = small_target_summary[cls_id]
        small_gt = s["small_gt"]
        small_detected = s["small_detected"]
        small_missed = s["small_missed"]

        small_target_summary_rows.append(
            {
                "model": model_name,
                "class_id": cls_id,
                "class_name": class_names[cls_id],
                "small_gt": int(small_gt),
                "small_detected_iou50": int(small_detected),
                "small_missed_iou50": int(small_missed),
                "small_recall_iou50": safe_div(small_detected, small_gt),
                "small_area_ratio_threshold": args.small_area_ratio,
            }
        )

    return (
        per_class_iou50_rows,
        size_recall_rows,
        missed_small_target_rows,
        small_target_summary_rows,
        image_match_rows,
        fp_detail_rows,
        fn_detail_rows,
        prediction_detail_rows,
    )


# ============================================================
# Model input parsing
# ============================================================

def parse_model_specs(args) -> List[Tuple[str, Path]]:
    """
    Accept:
      --weights name=path/to/best.pt
      --runs_root exp_root_dir   -> recursively find exp_root_dir/**/weights/best.pt
    """
    specs = []

    if args.runs_root:
        runs_root = Path(args.runs_root)
        if not runs_root.exists():
            raise FileNotFoundError(f"runs_root does not exist: {runs_root}")
        if not runs_root.is_dir():
            raise NotADirectoryError(f"runs_root is not a directory: {runs_root}")

        for w in sorted(runs_root.rglob("weights/best.pt")):
            exp_dir = w.parent.parent
            name = exp_dir.name
            specs.append((safe_name(name), w.resolve()))
    elif args.weights:
        for item in args.weights:
            if "=" not in item:
                raise ValueError(f"--weights must use name=path format, got: {item}")

            name, p = item.split("=", 1)
            specs.append((safe_name(name), Path(p).resolve()))

    if not specs:
        raise ValueError(
            "Please provide at least one --weights name=best.pt or --runs_root exp_root_dir."
        )

    checked = []
    for name, w in specs:
        if not w.exists():
            raise FileNotFoundError(f"best weight not found for {name}: {w}")
        checked.append((name, w))

    return checked


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="./data/det_dataset.yaml", help="yolo dataset config")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out_dir", default="./runs_audit/", help="output directory")
    parser.add_argument("--out_suffix", default="", help="output folder suffix")

    parser.add_argument(
        "--runs_root",
        type=str,
        default=None,
        help="Root directory. The script recursively finds **/weights/best.pt.",
    )

    parser.add_argument(
        "--weights",
        action="append",
        default=[],
        help="name=path/to/best.pt. Can be repeated.",
    )

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--pred-conf", type=float, default=0.25)
    parser.add_argument("--pred-iou", type=float, default=0.70)
    parser.add_argument("--match-iou", type=float, default=0.50)

    # New: lower threshold used only to collect diagnostic candidates.
    parser.add_argument("--analysis-conf", type=float, default=0.05)
    parser.add_argument("--near-iou", type=float, default=0.10)

    parser.add_argument("--small-area-ratio", type=float, default=0.03)
    parser.add_argument("--medium-area-ratio", type=float, default=0.15)

    parser.add_argument(
        "--small-target-class-names",
        default="leaf_blight,algal_leaf_spot",
        help=(
            "Comma-separated class names for small-target analysis, "
            "e.g. leaf_blight,algal_leaf_spot"
        ),
    )

    parser.add_argument("--save-pred-images", action="store_true")
    parser.add_argument("--plots", action="store_true")

    # New: save selected FP/FN visualizations.
    parser.add_argument("--save-error-visuals", action="store_true")
    parser.add_argument("--max-error-visuals-per-model", type=int, default=15000)
    parser.add_argument("--max-error-visuals-per-type", type=int, default=5000)

    # New: confidence/ranking diagnostics.
    parser.add_argument("--topk-conf-list", default="50,100,200", help="Comma-separated top-K list for confidence ranking diagnostics.")
    parser.add_argument("--max-top-fp-rows-per-model", type=int, default=200, help="Rows per model saved in high_conf_fp_detail.csv.")
    parser.add_argument("--baseline-name-patterns", default="baseline", help="Comma-separated model-name substrings used as baseline in model_comparison_summary.csv.")
    parser.add_argument("--reference-name-patterns", default="wo_pos_wo_mask,wo_pos", help="Comma-separated model-name substrings used as reference/best direct SSL in model_comparison_summary.csv.")

    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    out_dir = Path(args.out_dir).resolve()
    seg_path_name = f"{args.split}_eval_best_models"
    if args.out_suffix:
        seg_path_name = f"{seg_path_name}_{args.out_suffix}"
    out_dir = out_dir.joinpath(seg_path_name)
    ensure_dir(out_dir)

    dataset_root, test_img_dir, test_label_dir, class_names = parse_dataset_yaml(data_yaml, args.split)
    image_paths = collect_images(test_img_dir)

    if not image_paths:
        raise RuntimeError(f"No test images found: {test_img_dir}")

    model_specs = parse_model_specs(args)

    print("=" * 80)
    print("[INFO] dataset_root:", dataset_root)
    print("[INFO] split:", args.split)
    print("[INFO] test_img_dir:", test_img_dir)
    print("[INFO] test_label_dir:", test_label_dir)
    print("[INFO] first 5 images and labels:")
    for p in image_paths[:5]:
        lp = find_label_for_image(p, test_img_dir, test_label_dir)
        print(f"  IMG: {p}")
        print(f"  LAB: {lp} | exists={lp.exists()}")
    print("[INFO] class_names:", class_names)
    print("[INFO] num_test_images:", len(image_paths))
    print("[INFO] num_models:", len(model_specs))
    print("[INFO] pred_conf:", args.pred_conf)
    print("[INFO] analysis_conf:", args.analysis_conf)
    print("[INFO] pred_iou:", args.pred_iou)
    print("[INFO] match_iou:", args.match_iou)
    print("[INFO] near_iou:", args.near_iou)
    print("[INFO] small_target_class_names:", args.small_target_class_names)
    print("[INFO] small_area_ratio:", args.small_area_ratio)
    print("[INFO] medium_area_ratio:", args.medium_area_ratio)
    print("=" * 80)

    overall_rows = []
    per_class_rows = []
    custom_iou50_rows = []
    size_recall_rows = []
    missed_small_rows = []
    small_target_summary_rows = []
    image_match_rows = []

    # New FP/FN rows
    fp_detail_rows = []
    fn_detail_rows = []
    prediction_detail_rows = []

    run_manifest = []

    for model_name, weight_path in model_specs:
        print("\n" + "=" * 80)
        print(f"[EVAL] {model_name}")
        print(f"[WEIGHT] {weight_path}")
        print("=" * 80)

        model = YOLO(str(weight_path))

        val_project = out_dir / "ultralytics_val"
        val_name = model_name

        metrics = model.val(
            data=str(data_yaml),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            plots=args.plots,
            save_json=True,
            project=str(val_project),
            name=val_name,
            exist_ok=True,
            verbose=True,
        )

        overall, per_cls = extract_ultralytics_metrics(metrics, class_names, model_name)
        overall["weight"] = str(weight_path)

        overall_rows.append(overall)
        per_class_rows.extend(per_cls)

        (
            custom_rows,
            size_rows,
            missed_rows,
            small_target_rows,
            image_rows,
            fp_rows,
            fn_rows,
            pred_rows,
        ) = match_predictions_iou50(
            model=model,
            model_name=model_name,
            image_paths=image_paths,
            img_root=test_img_dir,
            label_root=test_label_dir,
            class_names=class_names,
            out_dir=out_dir,
            args=args,
        )

        custom_iou50_rows.extend(custom_rows)
        size_recall_rows.extend(size_rows)
        missed_small_rows.extend(missed_rows)
        small_target_summary_rows.extend(small_target_rows)
        image_match_rows.extend(image_rows)
        fp_detail_rows.extend(fp_rows)
        fn_detail_rows.extend(fn_rows)
        prediction_detail_rows.extend(pred_rows)

        run_manifest.append(
            {
                "model": model_name,
                "weight": str(weight_path),
                "ultralytics_val_dir": str(val_project / val_name),
            }
        )

    # ========================================================
    # Save outputs
    # ========================================================

    write_csv(out_dir / "overall_metrics.csv", overall_rows)
    write_csv(out_dir / "per_class_metrics.csv", per_class_rows)
    write_csv(out_dir / "custom_iou50_per_class.csv", custom_iou50_rows)
    write_csv(out_dir / "size_stratified_recall_iou50.csv", size_recall_rows)
    write_csv(out_dir / "missed_small_targets.csv", missed_small_rows)
    write_csv(out_dir / "small_target_summary.csv", small_target_summary_rows)
    write_csv(out_dir / "image_match_summary.csv", image_match_rows)

    # New FP/FN outputs
    write_csv(out_dir / "fp_detail.csv", fp_detail_rows)
    write_csv(out_dir / "fn_detail.csv", fn_detail_rows)

    fp_summary_rows = aggregate_count_rows(fp_detail_rows, ["model", "fp_type"])
    fp_by_class_rows = aggregate_count_rows(fp_detail_rows, ["model", "pred_class_id", "pred_class_name", "fp_type"])
    fn_summary_rows = aggregate_count_rows(fn_detail_rows, ["model", "fn_type"])
    fn_by_class_rows = aggregate_count_rows(fn_detail_rows, ["model", "gt_class_id", "gt_class_name", "fn_type"])

    topk_values = []
    for item in str(args.topk_conf_list).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            topk_values.append(int(item))
        except Exception:
            print(f"[WARN] ignore invalid top-k value: {item}")
    if not topk_values:
        topk_values = [50, 100, 200]

    confidence_summary_rows = make_confidence_summary(prediction_detail_rows, topk_values)
    confidence_bin_rows = make_confidence_bin_rows(prediction_detail_rows, [args.pred_conf, 0.40, 0.60, 0.80])
    high_conf_fp_rows = make_top_fp_rows(fp_detail_rows, args.max_top_fp_rows_per_model)
    fn_candidate_conf_rows = make_fn_candidate_conf_summary(fn_detail_rows)
    baseline_patterns = [x.strip() for x in str(args.baseline_name_patterns).split(",") if x.strip()]
    reference_patterns = [x.strip() for x in str(args.reference_name_patterns).split(",") if x.strip()]
    model_comparison_rows = make_model_comparison_summary(
        overall_rows=overall_rows,
        custom_rows=custom_iou50_rows,
        fp_rows=fp_detail_rows,
        fn_rows=fn_detail_rows,
        size_rows=size_recall_rows,
        confidence_rows=confidence_summary_rows,
        baseline_name_patterns=baseline_patterns,
        reference_name_patterns=reference_patterns,
    )

    write_csv(out_dir / "fp_summary.csv", fp_summary_rows)
    write_csv(out_dir / "fp_by_class.csv", fp_by_class_rows)
    write_csv(out_dir / "fn_summary.csv", fn_summary_rows)
    write_csv(out_dir / "fn_by_class.csv", fn_by_class_rows)

    # New confidence/ranking diagnostics
    write_csv(out_dir / "prediction_outcome_detail.csv", prediction_detail_rows)
    write_csv(out_dir / "confidence_summary.csv", confidence_summary_rows)
    write_csv(out_dir / "confidence_bin_summary.csv", confidence_bin_rows)
    write_csv(out_dir / "high_conf_fp_detail.csv", high_conf_fp_rows)
    write_csv(out_dir / "fn_candidate_conf_summary.csv", fn_candidate_conf_rows)
    write_csv(out_dir / "model_comparison_summary.csv", model_comparison_rows)

    manifest = {
        "data_yaml": str(data_yaml),
        "dataset_root": str(dataset_root),
        "split": args.split,
        "test_img_dir": str(test_img_dir),
        "test_label_dir": str(test_label_dir),
        "class_names": class_names,
        "num_test_images": len(image_paths),
        "args": vars(args),
        "runs": run_manifest,
        "outputs": {
            "overall_metrics": str(out_dir / "overall_metrics.csv"),
            "per_class_metrics": str(out_dir / "per_class_metrics.csv"),
            "custom_iou50_per_class": str(out_dir / "custom_iou50_per_class.csv"),
            "size_stratified_recall_iou50": str(out_dir / "size_stratified_recall_iou50.csv"),
            "missed_small_targets": str(out_dir / "missed_small_targets.csv"),
            "small_target_summary": str(out_dir / "small_target_summary.csv"),
            "image_match_summary": str(out_dir / "image_match_summary.csv"),
            "fp_detail": str(out_dir / "fp_detail.csv"),
            "fp_summary": str(out_dir / "fp_summary.csv"),
            "fp_by_class": str(out_dir / "fp_by_class.csv"),
            "fn_detail": str(out_dir / "fn_detail.csv"),
            "fn_summary": str(out_dir / "fn_summary.csv"),
            "fn_by_class": str(out_dir / "fn_by_class.csv"),
            "prediction_outcome_detail": str(out_dir / "prediction_outcome_detail.csv"),
            "confidence_summary": str(out_dir / "confidence_summary.csv"),
            "confidence_bin_summary": str(out_dir / "confidence_bin_summary.csv"),
            "high_conf_fp_detail": str(out_dir / "high_conf_fp_detail.csv"),
            "fn_candidate_conf_summary": str(out_dir / "fn_candidate_conf_summary.csv"),
            "model_comparison_summary": str(out_dir / "model_comparison_summary.csv"),
        },
    }

    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (out_dir / "README_eval_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Test Evaluation Summary\n\n")

        f.write("## Main metric priority\n\n")
        f.write("- Primary: mAP50\n")
        f.write("- Auxiliary: Recall, Precision, per-class AP50, small target recall\n")
        f.write("- Strict localization auxiliary: mAP50-90 / mAP50-95\n\n")

        f.write("## Split and thresholds\n\n")
        f.write(f"- split: `{args.split}`\n")
        f.write(f"- pred_conf: `{args.pred_conf}`\n")
        f.write(f"- analysis_conf: `{args.analysis_conf}`\n")
        f.write(f"- pred_iou: `{args.pred_iou}`\n")
        f.write(f"- match_iou: `{args.match_iou}`\n")
        f.write(f"- near_iou: `{args.near_iou}`\n\n")

        f.write("## Small target classes\n\n")
        f.write(f"- `{args.small_target_class_names}`\n\n")

        f.write("## Size definition\n\n")
        f.write(f"- small: area_ratio < {args.small_area_ratio}\n")
        f.write(
            f"- medium: {args.small_area_ratio} <= area_ratio < {args.medium_area_ratio}\n"
        )
        f.write(f"- large: area_ratio >= {args.medium_area_ratio}\n\n")

        f.write("## Output files\n\n")
        f.write("- `overall_metrics.csv`: overall P/R/mAP50/mAP50-90/mAP50-95\n")
        f.write("- `per_class_metrics.csv`: per-class P/R/AP50/AP50-90/AP50-95\n")
        f.write("- `custom_iou50_per_class.csv`: custom IoU50 TP/FP/FN/P/R\n")
        f.write("- `size_stratified_recall_iou50.csv`: small/medium/large recall per class\n")
        f.write("- `small_target_summary.csv`: small target recall for selected classes\n")
        f.write("- `missed_small_targets.csv`: missed small target GT cases\n")
        f.write("- `image_match_summary.csv`: per-image match summary\n")
        f.write("- `fp_detail.csv`: detailed FP attribution\n")
        f.write("- `fp_summary.csv`: FP type summary\n")
        f.write("- `fp_by_class.csv`: FP attribution grouped by predicted class\n")
        f.write("- `fn_detail.csv`: detailed FN attribution\n")
        f.write("- `fn_summary.csv`: FN type summary\n")
        f.write("- `fn_by_class.csv`: FN attribution grouped by GT class\n")
        f.write("- `prediction_outcome_detail.csv`: every eval-threshold prediction marked as TP or FP with confidence and IoU context\n")
        f.write("- `confidence_summary.csv`: TP/FP confidence distribution, TP-FP confidence gap, and top-K precision\n")
        f.write("- `confidence_bin_summary.csv`: TP/FP precision grouped by confidence bins\n")
        f.write("- `high_conf_fp_detail.csv`: highest-confidence FP cases for visual/error review\n")
        f.write("- `fn_candidate_conf_summary.csv`: FN type summary with best candidate confidence and IoU statistics\n")
        f.write("- `model_comparison_summary.csv`: one-row model comparison including deltas vs baseline/reference, error counts, and size recall\n")
        f.write("- `ultralytics_val/`: native Ultralytics val outputs\n")

        f.write("\n## FP type explanation\n\n")
        f.write("- `duplicate_fp`: same GT was already matched by a higher-confidence prediction.\n")
        f.write("- `class_error_fp`: IoU is enough, but predicted class is wrong.\n")
        f.write("- `localization_fp_low_iou`: same-class GT exists nearby, but IoU is below match threshold.\n")
        f.write("- `partial_overlap_fp`: weak overlap with a GT, but not enough for matching.\n")
        f.write("- `background_or_possible_missing_label_fp`: little/no overlap with GT; may be background FP or missing annotation.\n")

        f.write("\n## FN type explanation\n\n")
        f.write("- `fn_low_confidence`: same-class candidate has enough IoU but confidence is too low.\n")
        f.write("- `fn_class_error_high_conf`: high-confidence wrong-class prediction covers the GT.\n")
        f.write("- `fn_class_error_low_conf`: low-confidence wrong-class prediction covers the GT.\n")
        f.write("- `fn_low_iou_localization`: same-class candidate exists but IoU is too low.\n")
        f.write("- `fn_partial_overlap_other_class`: another-class candidate weakly overlaps the GT.\n")
        f.write("- `fn_no_candidate`: no candidate is close to the GT.\n")
        f.write("- `fn_matching_conflict`: same-class high-confidence candidate exists but not matched, often from greedy matching conflict.\n")

    print("\n[DONE]")
    print(f"[OUT] {out_dir}")
    print("[FILES]")
    print("  overall_metrics.csv")
    print("  per_class_metrics.csv")
    print("  custom_iou50_per_class.csv")
    print("  size_stratified_recall_iou50.csv")
    print("  small_target_summary.csv")
    print("  missed_small_targets.csv")
    print("  image_match_summary.csv")
    print("  fp_detail.csv")
    print("  fp_summary.csv")
    print("  fp_by_class.csv")
    print("  fn_detail.csv")
    print("  fn_summary.csv")
    print("  fn_by_class.csv")
    print("  prediction_outcome_detail.csv")
    print("  confidence_summary.csv")
    print("  confidence_bin_summary.csv")
    print("  high_conf_fp_detail.csv")
    print("  fn_candidate_conf_summary.csv")
    print("  model_comparison_summary.csv")
    print("  run_manifest.json")


if __name__ == "__main__":
    main()