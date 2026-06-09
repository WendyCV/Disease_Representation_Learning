import argparse
import csv
import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ============================================================
# Basic utilities
# ============================================================

def safe_name(name: str) -> str:
    name = re.sub(r"[^\w\-.]+", "_", str(name))
    return name[:180]


def norm_path(p: Any) -> str:
    return str(p).replace("\\", "/")


def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_csv_to_string(rows: List[dict]) -> str:
    if not rows:
        return ""

    fieldnames = []
    seen = set()

    for r in rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def write_json_to_string(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def write_jsonl_to_string(rows: List[dict]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)


def add_file_to_zip(zf: zipfile.ZipFile, src: Path, arcname: str):
    if src.exists() and src.is_file():
        zf.write(src, arcname)


def add_text_to_zip(zf: zipfile.ZipFile, text: str, arcname: str):
    zf.writestr(arcname, text.encode("utf-8-sig"))


def infer_yolo_ver(runs_root: Path, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit

    name = runs_root.name

    # Examples:
    # runs_yolov8n  -> yolov8n
    # runs_yolov9t  -> yolov9t
    # runs_yolov10n -> yolov10n
    # runs_yolov11n -> yolov11n
    if name.startswith("runs_"):
        return name.replace("runs_", "", 1)

    return name


# ============================================================
# Dataset parsing and packing
# ============================================================

def parse_dataset_yaml(data_yaml: Optional[Path]) -> Tuple[Optional[dict], List[str]]:
    if data_yaml is None or not data_yaml.exists():
        return None, []

    data = load_yaml(data_yaml)
    names = data.get("names", [])

    if isinstance(names, dict):
        class_names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    elif isinstance(names, list):
        class_names = names
    else:
        class_names = []

    return data, class_names


def collect_dataset_split_files(dataset_root: Path, split: str) -> Tuple[List[Path], List[Path]]:
    img_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split

    image_files = []
    label_files = []

    if img_dir.exists():
        for p in img_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                image_files.append(p)

    if label_dir.exists():
        for p in label_dir.rglob("*.txt"):
            if p.is_file():
                label_files.append(p)

    return sorted(image_files), sorted(label_files)


def pack_dataset_split(
    zf: zipfile.ZipFile,
    dataset_root: Path,
    split: str,
):
    image_files, label_files = collect_dataset_split_files(dataset_root, split)

    for p in image_files:
        rel = p.relative_to(dataset_root / "images" / split)
        arcname = Path("dataset") / split / "images" / rel
        add_file_to_zip(zf, p, norm_path(arcname))

    for p in label_files:
        rel = p.relative_to(dataset_root / "labels" / split)
        arcname = Path("dataset") / split / "labels" / rel
        add_file_to_zip(zf, p, norm_path(arcname))

    return {
        "dataset_root": str(dataset_root),
        "split": split,
        "num_images": len(image_files),
        "num_labels": len(label_files),
    }


# ============================================================
# Box helpers
# ============================================================

def box_area_xyxy(box: List[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = box_area_xyxy(a)
    area_b = box_area_xyxy(b)

    union = area_a + area_b - inter
    if union <= 0:
        return 0.0

    return inter / union


def area_bin(area_ratio: float) -> str:
    if area_ratio < 0.003:
        return "tiny"
    if area_ratio < 0.02:
        return "small"
    if area_ratio < 0.05:
        return "medium_small"
    if area_ratio < 0.15:
        return "medium"
    return "large"


def enrich_box(
    box: dict,
    image_w: Optional[int],
    image_h: Optional[int],
    prefix: str = "",
) -> dict:
    xyxy = box.get("xyxy")
    if xyxy is None:
        return dict(box)

    x1, y1, x2, y2 = [float(v) for v in xyxy]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)

    if image_w and image_h and image_w > 0 and image_h > 0:
        area_ratio = (w * h) / float(image_w * image_h)
    else:
        area_ratio = float("nan")

    out = dict(box)
    out[f"{prefix}x1"] = x1
    out[f"{prefix}y1"] = y1
    out[f"{prefix}x2"] = x2
    out[f"{prefix}y2"] = y2
    out[f"{prefix}w_px"] = w
    out[f"{prefix}h_px"] = h
    out[f"{prefix}area_ratio"] = area_ratio
    out[f"{prefix}area_bin"] = area_bin(area_ratio) if area_ratio == area_ratio else ""
    out[f"{prefix}aspect_w_h"] = w / h if h > 0 else ""

    return out


def yolo_to_xyxy(xc, yc, bw, bh, img_w, img_h):
    x1 = (xc - bw / 2.0) * img_w
    y1 = (yc - bh / 2.0) * img_h
    x2 = (xc + bw / 2.0) * img_w
    y2 = (yc + bh / 2.0) * img_h

    return [
        max(0.0, min(float(img_w), x1)),
        max(0.0, min(float(img_h), y1)),
        max(0.0, min(float(img_w), x2)),
        max(0.0, min(float(img_h), y2)),
    ]


# ============================================================
# Label fallback
# ============================================================

def read_yolo_label_file(
    label_path: Path,
    image_w: int,
    image_h: int,
    class_names: List[str],
) -> List[dict]:
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
            cls_id = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:])
        except Exception:
            continue

        xyxy = yolo_to_xyxy(xc, yc, bw, bh, image_w, image_h)
        cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"cls_{cls_id}"

        rows.append(
            {
                "gt_id": line_id - 1,
                "gt_line_id": line_id,
                "cls_id": cls_id,
                "cls_name": cls_name,
                "xyxy": xyxy,
                "source": "label_txt_fallback",
                "raw_line": raw,
            }
        )

    return rows


# ============================================================
# Meta extraction
# ============================================================

def get_image_hw(sample_meta: Optional[dict]) -> Tuple[Optional[int], Optional[int]]:
    if not sample_meta:
        return None, None

    for key in ["original_hw", "resized_hw"]:
        hw = sample_meta.get(key)
        if isinstance(hw, list) and len(hw) == 2:
            h, w = hw
            return int(w), int(h)

    return None, None


def get_image_name(sample_dir: Path, sample_meta: Optional[dict]) -> str:
    if sample_meta:
        for k in ["image_name", "image_rel_path"]:
            if sample_meta.get(k):
                return Path(str(sample_meta[k])).name

    return sample_dir.name


def get_sample_id(sample_dir: Path, sample_meta: Optional[dict]) -> str:
    if sample_meta and sample_meta.get("sample_id"):
        return str(sample_meta["sample_id"])
    return sample_dir.name


def extract_prediction_boxes(
    prediction_meta: Optional[dict],
    sample_meta: Optional[dict],
    image_w: Optional[int],
    image_h: Optional[int],
) -> List[dict]:
    pred_obj = None

    if prediction_meta:
        if "boxes" in prediction_meta:
            pred_obj = prediction_meta
        elif "prediction" in prediction_meta and isinstance(prediction_meta["prediction"], dict):
            pred_obj = prediction_meta["prediction"]

    if pred_obj is None and sample_meta:
        pred_obj = sample_meta.get("prediction")

    boxes_raw = []
    if isinstance(pred_obj, dict):
        boxes_raw = pred_obj.get("boxes", [])

    rows = []

    for i, b in enumerate(boxes_raw):
        xyxy = b.get("xyxy") or b.get("bbox") or b.get("box")
        if not xyxy or len(xyxy) != 4:
            continue

        cls_id = b.get("cls_id", b.get("class_id", b.get("cls", "")))
        cls_name = b.get("cls_name", b.get("class_name", ""))

        row = {
            "pred_id": i,
            "cls_id": cls_id,
            "cls_name": cls_name,
            "conf": float(b.get("conf", b.get("score", 0.0))),
            "xyxy": [float(v) for v in xyxy],
        }

        row = enrich_box(row, image_w, image_h)
        rows.append(row)

    rows = sorted(rows, key=lambda r: float(r.get("conf", 0.0)), reverse=True)
    for i, r in enumerate(rows):
        r["pred_rank"] = i

    return rows


def _extract_boxes_from_list(
    boxes_raw: List[dict],
    image_w: Optional[int],
    image_h: Optional[int],
    class_names: List[str],
) -> List[dict]:
    rows = []

    for i, b in enumerate(boxes_raw):
        xyxy = (
            b.get("xyxy")
            or b.get("bbox_xyxy")
            or b.get("bbox")
            or b.get("box")
        )

        if xyxy is None and all(k in b for k in ["x1", "y1", "x2", "y2"]):
            xyxy = [b["x1"], b["y1"], b["x2"], b["y2"]]

        if xyxy is None or len(xyxy) != 4:
            continue

        cls_id = b.get("cls_id", b.get("class_id", b.get("cls", b.get("category_id", ""))))
        cls_name = b.get("cls_name", b.get("class_name", b.get("name", "")))

        if cls_name == "" and isinstance(cls_id, int) and 0 <= cls_id < len(class_names):
            cls_name = class_names[cls_id]

        row = {
            "gt_id": i,
            "gt_line_id": b.get("line_id", b.get("gt_line_id", "")),
            "cls_id": cls_id,
            "cls_name": cls_name,
            "xyxy": [float(v) for v in xyxy],
            "source": "annotation_bbox_meta",
        }

        row = enrich_box(row, image_w, image_h)
        rows.append(row)

    return rows


def extract_gt_boxes(
    annotation_meta: Optional[dict],
    sample_meta: Optional[dict],
    sample_dir: Path,
    dataset_root: Optional[Path],
    split: str,
    class_names: List[str],
    image_w: Optional[int],
    image_h: Optional[int],
) -> Tuple[List[dict], str]:
    """
    Try order:
      1. annotation_bbox_meta.json
      2. sample_meta["annotation"] / ["gt"] / ["boxes"]
      3. dataset labels/test/*.txt fallback
    """
    candidate_lists = []

    if annotation_meta:
        for key in ["boxes", "annotations", "gt_boxes", "objects", "labels"]:
            if isinstance(annotation_meta.get(key), list):
                candidate_lists.append(annotation_meta[key])

    if sample_meta:
        for key in ["annotation", "annotations", "gt", "gt_boxes"]:
            obj = sample_meta.get(key)
            if isinstance(obj, dict):
                for kk in ["boxes", "annotations", "gt_boxes", "objects", "labels"]:
                    if isinstance(obj.get(kk), list):
                        candidate_lists.append(obj[kk])
            elif isinstance(obj, list):
                candidate_lists.append(obj)

    for boxes_raw in candidate_lists:
        rows = _extract_boxes_from_list(boxes_raw, image_w, image_h, class_names)
        if rows:
            return rows, "annotation_meta"

    # Fallback to YOLO label txt
    if sample_meta and dataset_root and image_w and image_h:
        image_name = get_image_name(sample_dir, sample_meta)
        label_path = dataset_root / "labels" / split / Path(image_name).with_suffix(".txt")
        rows = read_yolo_label_file(label_path, image_w, image_h, class_names)
        if rows:
            for r in rows:
                r = enrich_box(r, image_w, image_h)
            return rows, "label_txt_fallback"

    return [], "missing_gt"


# ============================================================
# Matching
# ============================================================

def match_predictions_to_gt(
    pred_boxes: List[dict],
    gt_boxes: List[dict],
    match_iou: float,
    overlap_iou: float,
) -> Tuple[List[dict], List[dict]]:
    """
    Return:
      pred_rows, gt_rows
    """
    matched_gt_ids = set()

    pred_rows = []
    gt_rows = []

    # Prediction-level greedy matching by confidence
    for pred in sorted(pred_boxes, key=lambda r: float(r.get("conf", 0.0)), reverse=True):
        pred_xyxy = pred["xyxy"]
        pred_cls = pred.get("cls_id", "")

        best_same = None
        best_same_iou = 0.0
        best_any = None
        best_any_iou = 0.0

        for gt in gt_boxes:
            gt_xyxy = gt["xyxy"]
            iou = box_iou(pred_xyxy, gt_xyxy)

            if iou > best_any_iou:
                best_any_iou = iou
                best_any = gt

            if str(gt.get("cls_id", "")) == str(pred_cls) and iou > best_same_iou:
                best_same_iou = iou
                best_same = gt

        status = "FP"
        fp_type = ""

        if best_same is not None and best_same_iou >= match_iou:
            gt_key = best_same.get("gt_id", best_same.get("gt_line_id", ""))
            if gt_key not in matched_gt_ids:
                status = "TP"
                fp_type = ""
                matched_gt_ids.add(gt_key)
            else:
                status = "FP"
                fp_type = "FP_duplicate_same_class"
        else:
            if best_any is not None and best_any_iou >= overlap_iou:
                if str(best_any.get("cls_id", "")) != str(pred_cls):
                    fp_type = "FP_wrong_class_over_gt"
                else:
                    fp_type = "FP_same_class_low_iou"
            else:
                fp_type = "FP_background_or_missing_gt"

        row = dict(pred)
        row.update(
            {
                "prediction_status": status,
                "fp_type": fp_type,
                "best_same_gt_iou": best_same_iou,
                "best_same_gt_id": "" if best_same is None else best_same.get("gt_id", ""),
                "best_same_gt_line_id": "" if best_same is None else best_same.get("gt_line_id", ""),
                "best_same_gt_class_id": "" if best_same is None else best_same.get("cls_id", ""),
                "best_same_gt_class_name": "" if best_same is None else best_same.get("cls_name", ""),
                "best_any_gt_iou": best_any_iou,
                "best_any_gt_id": "" if best_any is None else best_any.get("gt_id", ""),
                "best_any_gt_line_id": "" if best_any is None else best_any.get("gt_line_id", ""),
                "best_any_gt_class_id": "" if best_any is None else best_any.get("cls_id", ""),
                "best_any_gt_class_name": "" if best_any is None else best_any.get("cls_name", ""),
            }
        )
        pred_rows.append(row)

    # GT-level best prediction
    for gt in gt_boxes:
        gt_xyxy = gt["xyxy"]
        gt_cls = gt.get("cls_id", "")

        best_same = None
        best_same_iou = 0.0
        best_any = None
        best_any_iou = 0.0

        for pred in pred_boxes:
            iou = box_iou(gt_xyxy, pred["xyxy"])

            if iou > best_any_iou:
                best_any_iou = iou
                best_any = pred

            if str(pred.get("cls_id", "")) == str(gt_cls) and iou > best_same_iou:
                best_same_iou = iou
                best_same = pred

        detected = int(best_same_iou >= match_iou)

        row = dict(gt)
        row.update(
            {
                "gt_detected": detected,
                "best_same_pred_iou": best_same_iou,
                "best_same_pred_conf": "" if best_same is None else best_same.get("conf", ""),
                "best_same_pred_class_id": "" if best_same is None else best_same.get("cls_id", ""),
                "best_same_pred_class_name": "" if best_same is None else best_same.get("cls_name", ""),
                "best_any_pred_iou": best_any_iou,
                "best_any_pred_conf": "" if best_any is None else best_any.get("conf", ""),
                "best_any_pred_class_id": "" if best_any is None else best_any.get("cls_id", ""),
                "best_any_pred_class_name": "" if best_any is None else best_any.get("cls_name", ""),
            }
        )
        gt_rows.append(row)

    return pred_rows, gt_rows


# ============================================================
# Experiment processing
# ============================================================

def find_experiment_dirs(runs_root: Path) -> List[Path]:
    stage2_dir = runs_root / "glcp_stage2_yolo_det"

    if not stage2_dir.exists():
        stage2_dir = runs_root

    exp_dirs = []

    for d in sorted(stage2_dir.iterdir()):
        if not d.is_dir():
            continue

        fmap_dir = d / "layer_feature_maps"
        if fmap_dir.exists() and fmap_dir.is_dir():
            exp_dirs.append(d)

    return exp_dirs


def find_sample_dirs(exp_dir: Path) -> List[Path]:
    fmap_dir = exp_dir / "layer_feature_maps"
    if not fmap_dir.exists():
        return []

    sample_dirs = []
    for d in sorted(fmap_dir.iterdir()):
        if d.is_dir() and d.name.startswith("sample"):
            sample_dirs.append(d)

    return sample_dirs


def process_sample_dir(
    sample_dir: Path,
    exp_name: str,
    yolo_ver: str,
    dataset_root: Optional[Path],
    split: str,
    class_names: List[str],
    match_iou: float,
    overlap_iou: float,
) -> Tuple[Optional[dict], List[dict], List[dict], Optional[dict]]:
    sample_meta_path = sample_dir / "sample_meta.json"
    prediction_meta_path = sample_dir / "prediction_meta.json"
    annotation_meta_path = sample_dir / "annotation_bbox_meta.json"

    sample_meta = load_json(sample_meta_path)
    prediction_meta = load_json(prediction_meta_path)
    annotation_meta = load_json(annotation_meta_path)

    if sample_meta is None:
        err = {
            "sample_dir": str(sample_dir),
            "experiment": exp_name,
            "error": "missing_or_invalid_sample_meta_json",
        }
        return None, [], [], err

    image_w, image_h = get_image_hw(sample_meta)
    sample_id = get_sample_id(sample_dir, sample_meta)
    image_name = get_image_name(sample_dir, sample_meta)

    pred_boxes = extract_prediction_boxes(prediction_meta, sample_meta, image_w, image_h)

    gt_boxes, gt_source = extract_gt_boxes(
        annotation_meta=annotation_meta,
        sample_meta=sample_meta,
        sample_dir=sample_dir,
        dataset_root=dataset_root,
        split=split,
        class_names=class_names,
        image_w=image_w,
        image_h=image_h,
    )

    pred_rows, gt_rows = match_predictions_to_gt(
        pred_boxes=pred_boxes,
        gt_boxes=gt_boxes,
        match_iou=match_iou,
        overlap_iou=overlap_iou,
    )

    base_info = {
        "yolo_ver": yolo_ver,
        "experiment": exp_name,
        "sample_id": sample_id,
        "sample_index": sample_meta.get("sample_index", ""),
        "dataset_split": sample_meta.get("dataset_split", split),
        "image_name": image_name,
        "image_rel_path": sample_meta.get("image_rel_path", image_name),
        "dataset_rel_path": sample_meta.get("dataset_rel_path", ""),
        "image_path": sample_meta.get("image_path", ""),
        "label_path": sample_meta.get("label_path", ""),
        "image_w": image_w,
        "image_h": image_h,
        "sample_dir": str(sample_dir),
        "gt_source": gt_source,
    }

    pred_out = []
    for r in pred_rows:
        row = dict(base_info)
        row.update(
            {
                "pred_id": r.get("pred_id", ""),
                "pred_rank": r.get("pred_rank", ""),
                "pred_class_id": r.get("cls_id", ""),
                "pred_class_name": r.get("cls_name", ""),
                "pred_conf": r.get("conf", ""),
                "pred_x1": r.get("x1", ""),
                "pred_y1": r.get("y1", ""),
                "pred_x2": r.get("x2", ""),
                "pred_y2": r.get("y2", ""),
                "pred_w_px": r.get("w_px", ""),
                "pred_h_px": r.get("h_px", ""),
                "pred_area_ratio": r.get("area_ratio", ""),
                "pred_area_bin": r.get("area_bin", ""),
                "prediction_status": r.get("prediction_status", ""),
                "fp_type": r.get("fp_type", ""),
                "best_same_gt_iou": r.get("best_same_gt_iou", ""),
                "best_same_gt_line_id": r.get("best_same_gt_line_id", ""),
                "best_same_gt_class_name": r.get("best_same_gt_class_name", ""),
                "best_any_gt_iou": r.get("best_any_gt_iou", ""),
                "best_any_gt_line_id": r.get("best_any_gt_line_id", ""),
                "best_any_gt_class_name": r.get("best_any_gt_class_name", ""),
            }
        )
        pred_out.append(row)

    gt_out = []
    for r in gt_rows:
        row = dict(base_info)
        row.update(
            {
                "gt_id": r.get("gt_id", ""),
                "gt_line_id": r.get("gt_line_id", ""),
                "gt_class_id": r.get("cls_id", ""),
                "gt_class_name": r.get("cls_name", ""),
                "gt_x1": r.get("x1", ""),
                "gt_y1": r.get("y1", ""),
                "gt_x2": r.get("x2", ""),
                "gt_y2": r.get("y2", ""),
                "gt_w_px": r.get("w_px", ""),
                "gt_h_px": r.get("h_px", ""),
                "gt_area_ratio": r.get("area_ratio", ""),
                "gt_area_bin": r.get("area_bin", ""),
                "gt_detected": r.get("gt_detected", ""),
                "best_same_pred_iou": r.get("best_same_pred_iou", ""),
                "best_same_pred_conf": r.get("best_same_pred_conf", ""),
                "best_same_pred_class_name": r.get("best_same_pred_class_name", ""),
                "best_any_pred_iou": r.get("best_any_pred_iou", ""),
                "best_any_pred_conf": r.get("best_any_pred_conf", ""),
                "best_any_pred_class_name": r.get("best_any_pred_class_name", ""),
            }
        )
        gt_out.append(row)

    num_tp = sum(1 for r in pred_rows if r.get("prediction_status") == "TP")
    num_fp = sum(1 for r in pred_rows if r.get("prediction_status") == "FP")
    num_fn = sum(1 for r in gt_rows if int(r.get("gt_detected", 0)) == 0)

    pred_class_counts = {}
    for r in pred_rows:
        name = r.get("cls_name", "")
        pred_class_counts[name] = pred_class_counts.get(name, 0) + 1

    gt_class_counts = {}
    for r in gt_rows:
        name = r.get("cls_name", "")
        gt_class_counts[name] = gt_class_counts.get(name, 0) + 1

    sample_row = dict(base_info)
    sample_row.update(
        {
            "num_gt": len(gt_rows),
            "num_pred": len(pred_rows),
            "num_tp": num_tp,
            "num_fp": num_fp,
            "num_fn": num_fn,
            "pred_class_counts": json.dumps(pred_class_counts, ensure_ascii=False),
            "gt_class_counts": json.dumps(gt_class_counts, ensure_ascii=False),
            "prediction_meta_exists": prediction_meta_path.exists(),
            "annotation_meta_exists": annotation_meta_path.exists(),
            "sample_meta_exists": sample_meta_path.exists(),
            "prediction_image_path": sample_meta.get("prediction_image_path", ""),
            "annotation_image_path": sample_meta.get("annotation_image_path", ""),
            "prediction_vs_annotation_path": sample_meta.get("prediction_vs_annotation_path", ""),
            "overview_path": sample_meta.get("overview_path", ""),
        }
    )

    audit_item = {
        "sample": sample_row,
        "gt_boxes": gt_out,
        "prediction_boxes": pred_out,
    }

    return sample_row, pred_out, gt_out, audit_item


def process_experiment(
    exp_dir: Path,
    yolo_ver: str,
    dataset_root: Optional[Path],
    split: str,
    class_names: List[str],
    match_iou: float,
    overlap_iou: float,
):
    exp_name = exp_dir.name
    sample_dirs = find_sample_dirs(exp_dir)

    sample_rows = []
    pred_rows = []
    gt_rows = []
    audit_items = []
    error_rows = []

    for sample_dir in sample_dirs:
        sample_row, pred_out, gt_out, err_or_audit = process_sample_dir(
            sample_dir=sample_dir,
            exp_name=exp_name,
            yolo_ver=yolo_ver,
            dataset_root=dataset_root,
            split=split,
            class_names=class_names,
            match_iou=match_iou,
            overlap_iou=overlap_iou,
        )

        if sample_row is None:
            error_rows.append(err_or_audit)
            continue

        sample_rows.append(sample_row)
        pred_rows.extend(pred_out)
        gt_rows.extend(gt_out)
        audit_items.append(err_or_audit)

    manifest = {
        "yolo_ver": yolo_ver,
        "experiment": exp_name,
        "experiment_dir": str(exp_dir),
        "num_samples": len(sample_rows),
        "num_prediction_rows": len(pred_rows),
        "num_gt_rows": len(gt_rows),
        "num_error_rows": len(error_rows),
        "match_iou": match_iou,
        "overlap_iou": overlap_iou,
    }

    return {
        "experiment": exp_name,
        "samples_summary": sample_rows,
        "prediction_boxes": pred_rows,
        "gt_boxes": gt_rows,
        "sample_audit": audit_items,
        "errors": error_rows,
        "manifest": manifest,
    }


# ============================================================
# Optional visual packing
# ============================================================

KEY_VISUAL_KEYS = [
    "input_original_path",
    "prediction_image_path",
    "annotation_image_path",
    "prediction_vs_annotation_path",
    "overview_path",
]


def resolve_sample_visual_path(sample_dir: Path, value: str) -> Optional[Path]:
    if not value:
        return None

    # Most robust: file basename exists in sample_dir
    basename = Path(value).name
    candidate = sample_dir / basename
    if candidate.exists():
        return candidate

    p = Path(value)
    if p.exists():
        return p

    return None


def pack_key_visuals_for_experiment(
    zf: zipfile.ZipFile,
    exp_dir: Path,
    yolo_ver: str,
    exp_name: str,
):
    for sample_dir in find_sample_dirs(exp_dir):
        sample_meta = load_json(sample_dir / "sample_meta.json")
        if not sample_meta:
            continue

        sample_id = get_sample_id(sample_dir, sample_meta)

        for key in KEY_VISUAL_KEYS:
            value = sample_meta.get(key, "")
            src = resolve_sample_visual_path(sample_dir, value)
            if src is None:
                continue

            arcname = Path(yolo_ver) / exp_name / "visuals" / sample_id / src.name
            add_file_to_zip(zf, src, norm_path(arcname))


# ============================================================
# Main
# ============================================================

def parse_yolo_versions(yolo_ver_arg: str) -> List[str]:
    """
    Support:
      --yolo-ver yolov8n,yolov9t,yolov10n,yolov11n
      --yolo-ver yolov8n yolov9t yolov10n yolov11n
    """
    if yolo_ver_arg is None:
        return []

    if isinstance(yolo_ver_arg, list):
        raw_items = yolo_ver_arg
    else:
        raw_items = [yolo_ver_arg]

    out = []

    for item in raw_items:
        if item is None:
            continue

        for part in str(item).split(","):
            part = part.strip()
            if not part:
                continue

            # Allow users to input v8n / v9t / v10n / v11n if desired
            if part.startswith("v") and not part.startswith("yolo"):
                part = "yolo" + part

            out.append(part)

    # de-duplicate while keeping order
    seen = set()
    final = []
    for v in out:
        if v not in seen:
            final.append(v)
            seen.add(v)

    return final


def resolve_runs_root_from_yolo_ver(runs_parent: Path, yolo_ver: str) -> Path:
    """
    Given:
      yolo_ver = yolov8n
    return:
      runs_parent / runs_yolov8n
    """
    return runs_parent / f"runs_{yolo_ver}"


def default_multi_out_zip(out_arg: str, split: str, yolo_versions: List[str]) -> Path:
    out_path = Path(out_arg).resolve()

    # If user gives a zip path, use it directly.
    if out_path.suffix.lower() == ".zip":
        return out_path

    # Otherwise treat --out as output directory.
    # tag = "_".join(yolo_versions)
    return out_path / f"packed_{split}_prediction_audit.zip"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        default="./data/det_dataset/",
        help="Dataset root, e.g. data/det_dataset",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to pack, e.g. test/val/train",
    )
    parser.add_argument(
        "--data-yaml",
        default="./data/det_dataset.yaml",
        help="Optional det_dataset.yaml for class names",
    )

    # NEW: yolo-ver is now an array-like argument.
    # Examples:
    #   --yolo-ver yolov8n,yolov9t,yolov10n,yolov11n
    #   --yolo-ver yolov8n yolov9t yolov10n yolov11n
    parser.add_argument(
        "--yolo-ver",
        nargs="+",
        required=True,
        help=(
            "YOLO version list. Examples: "
            "--yolo-ver yolov8n,yolov9t,yolov10n,yolov11n "
            "or --yolo-ver yolov8n yolov9t yolov10n yolov11n"
        ),
    )

    # NEW: runs-root is no longer required.
    # The script will automatically use:
    #   runs_<yolo_ver>
    # under this parent directory.
    parser.add_argument(
        "--runs-parent",
        default=".",
        help="Parent directory containing runs_<yolo_ver> folders. Default: current project root.",
    )

    parser.add_argument(
        "--out",
        default="./runs_audit/",
        help=(
            "Output directory or output zip path. "
            "If a directory is given, one multi-version zip will be created inside it."
        ),
    )

    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--overlap-iou", type=float, default=0.30)

    parser.add_argument(
        "--include-key-visuals",
        action="store_true",
        help=(
            "Also pack input_original/prediction/annotation/"
            "prediction_vs_annotation/overview images from each sample dir."
        ),
    )

    parser.add_argument(
        "--experiment-filter",
        default="",
        help=(
            "Optional comma-separated substrings. "
            "Only experiments whose names contain any substring will be packed."
        ),
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    data_yaml = Path(args.data_yaml).resolve() if args.data_yaml else dataset_root / "det_dataset.yaml"
    runs_parent = Path(args.runs_parent).resolve()

    yolo_versions = parse_yolo_versions(args.yolo_ver)

    if not yolo_versions:
        raise ValueError("No valid --yolo-ver values were provided.")

    out_zip = default_multi_out_zip(
        out_arg=args.out,
        split=args.split,
        yolo_versions=yolo_versions,
    )
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    _, class_names = parse_dataset_yaml(data_yaml if data_yaml.exists() else None)

    exp_filters = [s.strip() for s in args.experiment_filter.split(",") if s.strip()]

    print("[PACK MULTI-YOLO]")
    print("  dataset_root:", dataset_root)
    print("  split:", args.split)
    print("  data_yaml:", data_yaml if data_yaml.exists() else "NOT_FOUND")
    print("  runs_parent:", runs_parent)
    print("  yolo_versions:", yolo_versions)
    print("  out:", out_zip)

    global_manifest = {
        "dataset_root": str(dataset_root),
        "split": args.split,
        "runs_parent": str(runs_parent),
        "yolo_versions": yolo_versions,
        "data_yaml": str(data_yaml) if data_yaml.exists() else "",
        "class_names": class_names,
        "match_iou": args.match_iou,
        "overlap_iou": args.overlap_iou,
        "include_key_visuals": bool(args.include_key_visuals),
        "experiment_filter": exp_filters,
        "yolo_manifests": [],
    }

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # --------------------------------------------------------
        # 1) Pack dataset split only once
        # --------------------------------------------------------
        dataset_manifest = pack_dataset_split(
            zf=zf,
            dataset_root=dataset_root,
            split=args.split,
        )

        add_text_to_zip(
            zf,
            write_json_to_string(dataset_manifest),
            "dataset/manifest.json",
        )

        if data_yaml.exists():
            add_file_to_zip(zf, data_yaml, "dataset/det_dataset.yaml")

        # --------------------------------------------------------
        # 2) Traverse each YOLO version
        # --------------------------------------------------------
        for yolo_idx, yolo_ver in enumerate(yolo_versions, start=1):
            runs_root = resolve_runs_root_from_yolo_ver(runs_parent, yolo_ver)

            print("\n" + "=" * 80)
            print(f"[YOLO] {yolo_idx}/{len(yolo_versions)} {yolo_ver}")
            print("  runs_root:", runs_root)

            yolo_manifest = {
                "yolo_ver": yolo_ver,
                "runs_root": str(runs_root),
                "exists": runs_root.exists(),
                "num_experiments": 0,
                "experiments": [],
                "experiment_manifests": [],
                "errors": [],
            }

            if not runs_root.exists():
                msg = f"runs_root does not exist: {runs_root}"
                print("[WARN]", msg)
                yolo_manifest["errors"].append(msg)
                global_manifest["yolo_manifests"].append(yolo_manifest)
                continue

            exp_dirs = find_experiment_dirs(runs_root)

            if exp_filters:
                exp_dirs = [
                    d for d in exp_dirs
                    if any(f in d.name for f in exp_filters)
                ]

            yolo_manifest["num_experiments"] = len(exp_dirs)
            yolo_manifest["experiments"] = [d.name for d in exp_dirs]

            print("  experiments:", len(exp_dirs))

            for exp_idx, exp_dir in enumerate(exp_dirs, start=1):
                print(f"[EXP] {yolo_ver} {exp_idx}/{len(exp_dirs)} {exp_dir.name}")

                result = process_experiment(
                    exp_dir=exp_dir,
                    yolo_ver=yolo_ver,
                    dataset_root=dataset_root,
                    split=args.split,
                    class_names=class_names,
                    match_iou=args.match_iou,
                    overlap_iou=args.overlap_iou,
                )

                exp_name = result["experiment"]
                arc_base = Path(yolo_ver) / exp_name

                add_text_to_zip(
                    zf,
                    write_csv_to_string(result["samples_summary"]),
                    norm_path(arc_base / "samples_summary.csv"),
                )

                add_text_to_zip(
                    zf,
                    write_csv_to_string(result["gt_boxes"]),
                    norm_path(arc_base / "gt_boxes.csv"),
                )

                add_text_to_zip(
                    zf,
                    write_csv_to_string(result["prediction_boxes"]),
                    norm_path(arc_base / "prediction_boxes.csv"),
                )

                add_text_to_zip(
                    zf,
                    write_jsonl_to_string(result["sample_audit"]),
                    norm_path(arc_base / "sample_audit.jsonl"),
                )

                add_text_to_zip(
                    zf,
                    write_csv_to_string(result["errors"]),
                    norm_path(arc_base / "missing_or_error_samples.csv"),
                )

                add_text_to_zip(
                    zf,
                    write_json_to_string(result["manifest"]),
                    norm_path(arc_base / "manifest.json"),
                )

                yolo_manifest["experiment_manifests"].append(result["manifest"])

                if args.include_key_visuals:
                    pack_key_visuals_for_experiment(
                        zf=zf,
                        exp_dir=exp_dir,
                        yolo_ver=yolo_ver,
                        exp_name=exp_name,
                    )

            global_manifest["yolo_manifests"].append(yolo_manifest)

        # --------------------------------------------------------
        # 3) Global manifest + README
        # --------------------------------------------------------
        add_text_to_zip(
            zf,
            write_json_to_string(global_manifest),
            "pack_manifest.json",
        )

        readme = f"""# Packed multi-YOLO prediction audit

This zip contains:

1. Dataset split:
- dataset/{args.split}/images/
- dataset/{args.split}/labels/
- dataset/det_dataset.yaml
- dataset/manifest.json

2. Per-YOLO and per-experiment prediction-vs-GT meta:
- <yolo_ver>/<experiment>/samples_summary.csv
- <yolo_ver>/<experiment>/gt_boxes.csv
- <yolo_ver>/<experiment>/prediction_boxes.csv
- <yolo_ver>/<experiment>/sample_audit.jsonl
- <yolo_ver>/<experiment>/missing_or_error_samples.csv
- <yolo_ver>/<experiment>/manifest.json

YOLO versions included:
{", ".join(yolo_versions)}

Runs root rule:
- For each yolo_ver, runs root is inferred as:
  runs_parent / runs_<yolo_ver>

Example:
- yolo_ver = yolov8n
- runs root = runs_yolov8n

Matching rule:
- TP if same-class prediction IoU >= {args.match_iou}
- FP_wrong_class_over_gt if prediction overlaps a GT of another class at IoU >= {args.overlap_iou}
- FP_same_class_low_iou if same-class IoU is between overlap and match threshold
- FP_background_or_missing_gt if no sufficient GT overlap

Use:
- gt_boxes.csv to inspect missed GTs
- prediction_boxes.csv to inspect FP / wrong class / low IoU predictions
- sample_audit.jsonl for full per-image structured data
- pack_manifest.json for global package summary
"""

        add_text_to_zip(zf, readme, "README.txt")

    print("\n[DONE]")
    print(out_zip)


if __name__ == "__main__":
    main()