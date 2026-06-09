import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =========================================================
# YAML parsing
# =========================================================

def load_yolo_dataset_yaml(yaml_path: Path):
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "path" not in data:
        raise ValueError("YAML must contain `path` field.")

    dataset_root = Path(data["path"])

    names = data.get("names", None)
    if names is None:
        raise ValueError("YAML must contain `names` field.")

    if isinstance(names, dict):
        class_names = [names[i] for i in sorted(names.keys())]
    elif isinstance(names, list):
        class_names = names
    else:
        raise ValueError("Unsupported `names` format in YAML.")

    split_image_dirs = {}

    for split in ["train", "val", "test"]:
        if split in data and data[split] is not None:
            split_path = Path(data[split])

            if not split_path.is_absolute():
                split_path = dataset_root / split_path

            split_image_dirs[split] = split_path

    if len(split_image_dirs) == 0:
        raise ValueError("YAML must contain at least one of train/val/test.")

    split_label_dirs = {}

    for split, img_dir in split_image_dirs.items():
        parts = list(img_dir.parts)

        if "images" in parts:
            idx = parts.index("images")
            parts[idx] = "labels"
            label_dir = Path(*parts)
        else:
            raise ValueError(
                f"Cannot infer label path from image path: {img_dir}. "
                f"Expected path containing `images`."
            )

        split_label_dirs[split] = label_dir

    return dataset_root, split_image_dirs, split_label_dirs, class_names


def collect_images_by_split(split_image_dirs: Dict[str, Path]) -> Dict[str, dict]:
    images = {}

    for split, img_dir in split_image_dirs.items():
        if not img_dir.exists():
            print(f"[WARN] image dir not found: {img_dir}")
            continue

        for p in img_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                rel = p.relative_to(img_dir)
                rel_key = str(rel.with_suffix("")).replace("\\", "/")
                key = f"{split}/{rel_key}"
                images[key] = {
                    "split": split,
                    "path": p,
                }

    return images


def collect_labels_by_split(split_label_dirs: Dict[str, Path]) -> Dict[str, dict]:
    labels = {}

    for split, label_dir in split_label_dirs.items():
        if not label_dir.exists():
            print(f"[WARN] label dir not found: {label_dir}")
            continue

        for p in label_dir.rglob("*.txt"):
            rel = p.relative_to(label_dir)
            rel_key = str(rel.with_suffix("")).replace("\\", "/")
            key = f"{split}/{rel_key}"
            labels[key] = {
                "split": split,
                "path": p,
            }

    return labels


# =========================================================
# Label parsing and bbox tools
# =========================================================

def read_yolo_label(label_path: Path) -> Tuple[List[dict], List[str]]:
    boxes = []
    errors = []

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = label_path.read_text(encoding="gbk", errors="ignore").splitlines()

    for line_id, line in enumerate(lines, start=1):
        raw = line.strip()

        if not raw:
            continue

        parts = raw.split()

        if len(parts) != 5:
            errors.append(f"line_{line_id}: expected 5 fields, got {len(parts)} | {raw}")
            continue

        try:
            cls = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:])
        except Exception:
            errors.append(f"line_{line_id}: cannot parse numbers | {raw}")
            continue

        boxes.append(
            {
                "line_id": line_id,
                "cls": cls,
                "xc": xc,
                "yc": yc,
                "bw": bw,
                "bh": bh,
            }
        )

    return boxes, errors


def yolo_to_xyxy(box: dict, img_w: int, img_h: int):
    xc_px = box["xc"] * img_w
    yc_px = box["yc"] * img_h
    bw_px = box["bw"] * img_w
    bh_px = box["bh"] * img_h

    x1 = xc_px - bw_px / 2
    y1 = yc_px - bh_px / 2
    x2 = xc_px + bw_px / 2
    y2 = yc_px + bh_px / 2

    return x1, y1, x2, y2


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else float("nan")


def add_reason(reasons: List[str], condition: bool, reason: str):
    if condition:
        reasons.append(reason)


def summarize_numeric(values: List[float]) -> dict:
    vals = [
        v for v in values
        if isinstance(v, (int, float)) and not math.isnan(v)
    ]

    if not vals:
        return {
            "count": 0,
            "min": None,
            "p5": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
            "mean": None,
        }

    arr = np.array(vals, dtype=np.float64)

    return {
        "count": int(len(arr)),
        "min": float(np.min(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


# =========================================================
# Visualization
# =========================================================

def draw_boxes_with_issues(
    image_path: Path,
    boxes: List[dict],
    class_names: List[str],
    out_path: Path,
    issue_map: Dict[int, List[str]],
):
    img = cv2.imread(str(image_path))

    if img is None:
        return

    img_h, img_w = img.shape[:2]

    for box in boxes:
        cls = box["cls"]
        line_id = box["line_id"]

        x1, y1, x2, y2 = yolo_to_xyxy(box, img_w, img_h)

        x1i = max(0, min(img_w - 1, int(round(x1))))
        y1i = max(0, min(img_h - 1, int(round(y1))))
        x2i = max(0, min(img_w - 1, int(round(x2))))
        y2i = max(0, min(img_h - 1, int(round(y2))))

        is_issue = line_id in issue_map

        color = (0, 0, 255) if is_issue else (0, 255, 0)
        thickness = 3 if is_issue else 2

        cv2.rectangle(img, (x1i, y1i), (x2i, y2i), color, thickness)

        if 0 <= cls < len(class_names):
            cls_name = class_names[cls]
        else:
            cls_name = f"OUT_OF_RANGE_{cls}"

        text = f"{line_id}:{cls_name}"
        if is_issue:
            text += " *"

        cv2.putText(
            img,
            text,
            (x1i, max(20, y1i - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    if issue_map:
        issue_lines = []

        for line_id, reasons in issue_map.items():
            issue_lines.append(f"line {line_id}: {', '.join(reasons)}")

        pad_h = max(40, 24 * min(len(issue_lines), 12) + 20)
        canvas = np.ones((img_h + pad_h, img_w, 3), dtype=np.uint8) * 255
        canvas[:img_h, :, :] = img

        y = img_h + 25
        for text in issue_lines[:12]:
            cv2.putText(
                canvas,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            y += 24

        img = canvas

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


# =========================================================
# Audit logic
# =========================================================

def audit_single_box(
    box: dict,
    cls_name: str,
    num_classes: int,
    img_w: int,
    img_h: int,
    args,
) -> dict:
    cls = box["cls"]

    x1, y1, x2, y2 = yolo_to_xyxy(box, img_w, img_h)

    bw_px = box["bw"] * img_w
    bh_px = box["bh"] * img_h
    area_px = bw_px * bh_px
    image_area = img_w * img_h
    area_ratio = safe_div(area_px, image_area)
    aspect_ratio = safe_div(bw_px, bh_px)

    reasons = []

    # 基础格式问题
    add_reason(reasons, cls < 0 or cls >= num_classes, "class_id_out_of_range")
    add_reason(reasons, not (0 <= box["xc"] <= 1), "xc_out_of_range")
    add_reason(reasons, not (0 <= box["yc"] <= 1), "yc_out_of_range")
    add_reason(reasons, not (0 < box["bw"] <= 1), "bbox_width_norm_invalid")
    add_reason(reasons, not (0 < box["bh"] <= 1), "bbox_height_norm_invalid")
    eps = 1e-3
    # add_reason(reasons, x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h, "bbox_out_of_image")
    add_reason(reasons, x1 < -eps or y1 < -eps or x2 > img_w + eps or y2 > img_h + eps, "bbox_out_of_image")
    add_reason(reasons, bw_px <= 0 or bh_px <= 0, "bbox_zero_or_negative_size")

    # 贴边
    edge_x = img_w * args.edge_touch_ratio
    edge_y = img_h * args.edge_touch_ratio

    touches_edge = (
        x1 <= edge_x
        or y1 <= edge_y
        or x2 >= img_w - edge_x
        or y2 >= img_h - edge_y
    )

    add_reason(reasons, touches_edge, "bbox_touching_image_edge")

    # 极端宽高比
    add_reason(reasons, aspect_ratio > args.max_aspect, "bbox_too_wide")
    add_reason(reasons, aspect_ratio < 1.0 / args.max_aspect, "bbox_too_tall")

    # 混合标注规则：
    # no_disease = 整片健康叶
    # disease = 病斑区域
    if cls_name == args.healthy_class_name:
        add_reason(
            reasons,
            area_ratio < args.healthy_min_area_ratio,
            "healthy_leaf_box_too_small",
        )

        add_reason(
            reasons,
            area_ratio > args.healthy_max_area_ratio,
            "healthy_leaf_box_too_large",
        )

        add_reason(
            reasons,
            bw_px < args.healthy_min_width_px,
            "healthy_leaf_box_width_too_small",
        )

        add_reason(
            reasons,
            bh_px < args.healthy_min_height_px,
            "healthy_leaf_box_height_too_small",
        )

    else:
        add_reason(
            reasons,
            area_ratio < args.disease_min_area_ratio,
            "lesion_box_too_small",
        )

        add_reason(
            reasons,
            area_ratio > args.disease_max_area_ratio,
            "lesion_box_too_large_maybe_whole_leaf",
        )

        add_reason(
            reasons,
            bw_px < args.disease_min_width_px,
            "lesion_box_width_too_small",
        )

        add_reason(
            reasons,
            bh_px < args.disease_min_height_px,
            "lesion_box_height_too_small",
        )

    return {
        "x1_px": x1,
        "y1_px": y1,
        "x2_px": x2,
        "y2_px": y2,
        "box_w_px": bw_px,
        "box_h_px": bh_px,
        "area_px": area_px,
        "area_ratio": area_ratio,
        "aspect_ratio": aspect_ratio,
        "is_suspicious": int(len(reasons) > 0),
        "suspicious_reasons": "|".join(reasons),
        "reasons_list": reasons,
    }


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="YOLO dataset yaml, e.g. data/det_dataset.yaml")
    parser.add_argument("--out", default="runs_audit/label_audit", help="output directory")

    parser.add_argument("--healthy-class-name", default="no_disease", help="healthy class name, default: no_disease")
    # no_disease 是整叶框
    parser.add_argument("--healthy-min-area-ratio", type=float, default=0.05)
    parser.add_argument("--healthy-max-area-ratio", type=float, default=0.95)
    parser.add_argument("--healthy-min-width-px", type=float, default=20)
    parser.add_argument("--healthy-min-height-px", type=float, default=20)

    # disease 是病斑框
    parser.add_argument("--disease-min-area-ratio", type=float, default=0.0005)
    parser.add_argument("--disease-max-area-ratio", type=float, default=0.60)
    parser.add_argument("--disease-min-width-px", type=float, default=6)
    parser.add_argument("--disease-min-height-px", type=float, default=6)

    # 通用规则
    parser.add_argument("--max-aspect", type=float, default=6.0)
    parser.add_argument("--edge-touch-ratio", type=float, default=0.01)
    parser.add_argument("--draw-all-normal-boxes", action="store_true", help="visualize normal boxes in green together with suspicious boxes")

    args = parser.parse_args()

    yaml_path = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_root, split_image_dirs, split_label_dirs, class_names = load_yolo_dataset_yaml(yaml_path)

    num_classes = len(class_names)

    if args.healthy_class_name not in class_names:
        raise ValueError(
            f"healthy class name '{args.healthy_class_name}' not found in class_names: {class_names}"
        )

    print("[INFO] Dataset root:", dataset_root)
    print("[INFO] Class names:", class_names)
    print("[INFO] Image dirs:")
    for k, v in split_image_dirs.items():
        print(f"  {k}: {v}")
    print("[INFO] Label dirs:")
    for k, v in split_label_dirs.items():
        print(f"  {k}: {v}")

    images = collect_images_by_split(split_image_dirs)
    labels = collect_labels_by_split(split_label_dirs)

    image_keys = set(images.keys())
    label_keys = set(labels.keys())

    missing_label_keys = sorted(image_keys - label_keys)
    orphan_label_keys = sorted(label_keys - image_keys)

    bbox_rows = []
    image_rows = []
    issue_rows = []

    class_box_count = {name: 0 for name in class_names}
    class_suspicious_count = {name: 0 for name in class_names}
    class_area_ratios = {name: [] for name in class_names}
    class_aspect_ratios = {name: [] for name in class_names}
    split_class_count = {}

    for key, item in sorted(images.items()):
        split = item["split"]
        image_path = item["path"]

        img = cv2.imread(str(image_path))
        if img is None:
            issue_rows.append(
                {
                    "image_key": key,
                    "split": split,
                    "issue_type": "image_read_failed",
                    "detail": str(image_path),
                }
            )
            continue

        img_h, img_w = img.shape[:2]

        label_item = labels.get(key)
        label_path = label_item["path"] if label_item else None

        if label_path is None:
            image_rows.append(
                {
                    "image_key": key,
                    "split": split,
                    "image_path": str(image_path),
                    "label_path": "",
                    "img_w": img_w,
                    "img_h": img_h,
                    "box_count": 0,
                    "suspicious_box_count": 0,
                    "status": "missing_label",
                }
            )

            issue_rows.append(
                {
                    "image_key": key,
                    "split": split,
                    "issue_type": "missing_label",
                    "detail": str(image_path),
                }
            )
            continue

        boxes, parse_errors = read_yolo_label(label_path)

        for err in parse_errors:
            issue_rows.append(
                {
                    "image_key": key,
                    "split": split,
                    "issue_type": "label_parse_error",
                    "detail": err,
                }
            )

        if len(boxes) == 0:
            image_rows.append(
                {
                    "image_key": key,
                    "split": split,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "img_w": img_w,
                    "img_h": img_h,
                    "box_count": 0,
                    "suspicious_box_count": 0,
                    "status": "empty_label",
                }
            )

            issue_rows.append(
                {
                    "image_key": key,
                    "split": split,
                    "issue_type": "empty_label",
                    "detail": str(label_path),
                }
            )
            continue

        suspicious_count = 0
        issue_map = {}

        for box in boxes:
            cls = box["cls"]

            if 0 <= cls < num_classes:
                cls_name = class_names[cls]
            else:
                cls_name = "OUT_OF_RANGE"

            result = audit_single_box(
                box=box,
                cls_name=cls_name,
                num_classes=num_classes,
                img_w=img_w,
                img_h=img_h,
                args=args,
            )

            reasons = result["reasons_list"]

            if reasons:
                suspicious_count += 1
                issue_map[box["line_id"]] = reasons

                for reason in reasons:
                    issue_rows.append(
                        {
                            "image_key": key,
                            "split": split,
                            "issue_type": reason,
                            "detail": (
                                f"line_id={box['line_id']}, "
                                f"class={cls_name}, "
                                f"area_ratio={result['area_ratio']:.6f}, "
                                f"aspect_ratio={result['aspect_ratio']:.3f}"
                            ),
                        }
                    )

            if cls_name in class_box_count:
                class_box_count[cls_name] += 1
                class_area_ratios[cls_name].append(result["area_ratio"])
                class_aspect_ratios[cls_name].append(result["aspect_ratio"])

                if reasons:
                    class_suspicious_count[cls_name] += 1

                split_class_count.setdefault(split, {name: 0 for name in class_names})
                split_class_count[split][cls_name] += 1

            bbox_rows.append(
                {
                    "image_key": key,
                    "split": split,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "line_id": box["line_id"],
                    "class_id": cls,
                    "class_name": cls_name,
                    "annotation_unit": (
                        "leaf_level" if cls_name == args.healthy_class_name else "lesion_level"
                    ),
                    "img_w": img_w,
                    "img_h": img_h,
                    "xc": box["xc"],
                    "yc": box["yc"],
                    "w_norm": box["bw"],
                    "h_norm": box["bh"],
                    "x1_px": result["x1_px"],
                    "y1_px": result["y1_px"],
                    "x2_px": result["x2_px"],
                    "y2_px": result["y2_px"],
                    "box_w_px": result["box_w_px"],
                    "box_h_px": result["box_h_px"],
                    "area_px": result["area_px"],
                    "area_ratio": result["area_ratio"],
                    "aspect_ratio": result["aspect_ratio"],
                    "is_suspicious": result["is_suspicious"],
                    "suspicious_reasons": result["suspicious_reasons"],
                }
            )

        status = "ok" if suspicious_count == 0 else "has_suspicious_boxes"

        image_rows.append(
            {
                "image_key": key,
                "split": split,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "img_w": img_w,
                "img_h": img_h,
                "box_count": len(boxes),
                "suspicious_box_count": suspicious_count,
                "status": status,
            }
        )

        if suspicious_count > 0:
            out_img_path = out_dir / "visual_suspicious" / split / f"{key.replace('/', '__')}.jpg"

            if args.draw_all_normal_boxes:
                draw_boxes = boxes
            else:
                draw_boxes = [b for b in boxes if b["line_id"] in issue_map]

            draw_boxes_with_issues(
                image_path=image_path,
                boxes=draw_boxes,
                class_names=class_names,
                out_path=out_img_path,
                issue_map=issue_map,
            )

    for key in orphan_label_keys:
        label_item = labels[key]
        issue_rows.append(
            {
                "image_key": key,
                "split": label_item["split"],
                "issue_type": "orphan_label",
                "detail": str(label_item["path"]),
            }
        )

    # =====================================================
    # Write outputs
    # =====================================================

    bbox_csv = out_dir / "bbox_audit.csv"
    if bbox_rows:
        with bbox_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(bbox_rows[0].keys()))
            writer.writeheader()
            writer.writerows(bbox_rows)

    image_csv = out_dir / "image_audit.csv"
    if image_rows:
        with image_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(image_rows[0].keys()))
            writer.writeheader()
            writer.writerows(image_rows)

    issue_csv = out_dir / "issues.csv"
    if issue_rows:
        with issue_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(issue_rows[0].keys()))
            writer.writeheader()
            writer.writerows(issue_rows)

    class_summary_rows = []

    for name in class_names:
        total = class_box_count[name]
        suspicious = class_suspicious_count[name]

        ratio = safe_div(suspicious, total)

        class_summary_rows.append(
            {
                "class_name": name,
                "annotation_unit": (
                    "leaf_level" if name == args.healthy_class_name else "lesion_level"
                ),
                "box_count": total,
                "suspicious_box_count": suspicious,
                "suspicious_ratio": ratio,
                "area_ratio_summary": json.dumps(
                    summarize_numeric(class_area_ratios[name]),
                    ensure_ascii=False,
                ),
                "aspect_ratio_summary": json.dumps(
                    summarize_numeric(class_aspect_ratios[name]),
                    ensure_ascii=False,
                ),
            }
        )

    class_summary_csv = out_dir / "class_summary.csv"
    with class_summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(class_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(class_summary_rows)

    summary = {
        "dataset_yaml": str(yaml_path),
        "dataset_root": str(dataset_root),
        "num_images": len(images),
        "num_labels": len(labels),
        "num_missing_labels": len(missing_label_keys),
        "num_orphan_labels": len(orphan_label_keys),
        "num_boxes": len(bbox_rows),
        "num_issues": len(issue_rows),
        "class_names": class_names,
        "healthy_class_name": args.healthy_class_name,
        "annotation_rule": {
            args.healthy_class_name: "leaf_level_bbox",
            "disease_classes": "lesion_level_bbox",
        },
        "split_image_dirs": {k: str(v) for k, v in split_image_dirs.items()},
        "split_label_dirs": {k: str(v) for k, v in split_label_dirs.items()},
        "thresholds": {
            "healthy_min_area_ratio": args.healthy_min_area_ratio,
            "healthy_max_area_ratio": args.healthy_max_area_ratio,
            "healthy_min_width_px": args.healthy_min_width_px,
            "healthy_min_height_px": args.healthy_min_height_px,
            "disease_min_area_ratio": args.disease_min_area_ratio,
            "disease_max_area_ratio": args.disease_max_area_ratio,
            "disease_min_width_px": args.disease_min_width_px,
            "disease_min_height_px": args.disease_min_height_px,
            "max_aspect": args.max_aspect,
            "edge_touch_ratio": args.edge_touch_ratio,
        },
        "class_box_count": class_box_count,
        "class_suspicious_count": class_suspicious_count,
        "split_class_count": split_class_count,
        "outputs": {
            "bbox_audit_csv": str(bbox_csv),
            "image_audit_csv": str(image_csv),
            "issues_csv": str(issue_csv),
            "class_summary_csv": str(class_summary_csv),
            "visual_suspicious_dir": str(out_dir / "visual_suspicious"),
        },
    }

    summary_json = out_dir / "summary.json"
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report_md = out_dir / "audit_report.md"

    with report_md.open("w", encoding="utf-8") as f:
        f.write("# YOLO Label Audit Report\n\n")

        f.write("## Dataset YAML\n\n")
        f.write(f"- `{yaml_path}`\n\n")

        f.write("## Annotation Rule\n\n")
        f.write(f"- `{args.healthy_class_name}`: leaf-level bounding box\n")
        f.write("- disease classes: lesion-level bounding box\n\n")

        f.write("## Basic Summary\n\n")
        f.write(f"- Images: {len(images)}\n")
        f.write(f"- Labels: {len(labels)}\n")
        f.write(f"- Missing labels: {len(missing_label_keys)}\n")
        f.write(f"- Orphan labels: {len(orphan_label_keys)}\n")
        f.write(f"- Boxes: {len(bbox_rows)}\n")
        f.write(f"- Issues: {len(issue_rows)}\n\n")

        f.write("## Class Summary\n\n")
        f.write("| Class | Annotation unit | Boxes | Suspicious boxes | Suspicious ratio |\n")
        f.write("|---|---|---:|---:|---:|\n")

        for row in class_summary_rows:
            ratio = row["suspicious_ratio"]
            ratio_str = "" if math.isnan(ratio) else f"{ratio:.4f}"

            f.write(
                f"| {row['class_name']} | {row['annotation_unit']} | "
                f"{row['box_count']} | {row['suspicious_box_count']} | {ratio_str} |\n"
            )

        f.write("\n## Output Files\n\n")
        f.write("- bbox_audit.csv\n")
        f.write("- image_audit.csv\n")
        f.write("- issues.csv\n")
        f.write("- class_summary.csv\n")
        f.write("- summary.json\n")
        f.write("- visual_suspicious/\n")

    print("[OK] Label audit finished.")
    print(f"[OUT] {out_dir}")
    print(f"[REPORT] {report_md}")


if __name__ == "__main__":
    main()