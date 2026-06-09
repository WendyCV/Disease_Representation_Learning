import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clean_class_name(name: str) -> str:
    """
    Convert disease name into YOLO-friendly class name.
    Example:
      "Apple Black Rot" -> "apple_black_rot"
    """
    name = str(name).strip()
    name = name.replace("/", "_").replace("\\", "_")
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^\w\-.]+", "_", name)
    return name.lower()


def normalize_key(name: str) -> str:
    """
    Normalize file basename for matching.
    """
    return Path(str(name).strip()).name.lower()


def normalize_stem(name: str) -> str:
    """
    Normalize file stem for fallback matching.
    """
    return Path(str(name).strip()).stem.lower()


def collect_files(root: Path, exts: set) -> List[Path]:
    if not root.exists():
        return []

    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )


def build_file_lookup(root: Path, exts: set) -> Dict[str, Path]:
    """
    Build exact basename lookup.
    """
    lookup: Dict[str, Path] = {}

    for p in collect_files(root, exts):
        lookup[normalize_key(p.name)] = p

    return lookup


def build_stem_lookup(root: Path, exts: set) -> Dict[str, Path]:
    """
    Build stem lookup for fallback matching.
    """
    lookup: Dict[str, Path] = {}

    for p in collect_files(root, exts):
        key = normalize_stem(p.name)
        if key not in lookup:
            lookup[key] = p

    return lookup


def find_by_name(
    name: str,
    exact_lookup: Dict[str, Path],
    stem_lookup: Dict[str, Path],
) -> Optional[Path]:
    """
    First match by exact basename.
    If failed, match by stem.
    """
    key = normalize_key(name)

    if key in exact_lookup:
        return exact_lookup[key]

    stem = normalize_stem(name)

    if stem in stem_lookup:
        return stem_lookup[stem]

    return None


def read_image_size(path: Path) -> Tuple[int, int]:
    img = Image.open(path)
    return img.size  # W, H


def load_mask(mask_path: Path) -> np.ndarray:
    """
    PlantSeg annotations may look completely black because foreground value may be 1.
    This function treats all non-zero pixels as foreground.

    If annotation is RGB, any non-zero channel is treated as foreground.
    """
    mask = Image.open(mask_path)
    arr = np.array(mask)

    if arr.ndim == 3:
        arr = np.any(arr > 0, axis=2).astype(np.uint8)

    return arr


def make_visible_mask(mask_arr: np.ndarray) -> np.ndarray:
    """
    Convert 0/1 or index mask into visible 0/255 binary mask.
    Saved to masks/.
    """
    out = np.zeros(mask_arr.shape[:2], dtype=np.uint8)
    out[mask_arr > 0] = 255
    return out


def connected_components_to_bboxes(binary: np.ndarray) -> List[dict]:
    """
    Extract all connected components from mask > 0.
    No area filtering is applied.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8),
        connectivity=8,
    )

    boxes = []

    for i in range(1, num_labels):  # 0 is background
        x, y, w, h, area = stats[i]

        if area <= 0 or w <= 0 or h <= 0:
            continue

        boxes.append(
            {
                "x1": int(x),
                "y1": int(y),
                "x2": int(x + w),
                "y2": int(y + h),
                "w": int(w),
                "h": int(h),
                "area_px": int(area),
            }
        )

    return boxes


def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    """
    Convert absolute xyxy bbox into normalized YOLO xywh format.
    """
    xc = ((x1 + x2) / 2.0) / W
    yc = ((y1 + y2) / 2.0) / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H

    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)

    return xc, yc, bw, bh


def write_csv(path: Path, rows: List[dict]):
    ensure_dir(path.parent)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = []
    seen = set()

    for r in rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_classes_txt_from_mapping(labels_root: Path, id_to_name: Dict[int, str]):
    """
    Write labels/classes.txt using continuous class ids.
    """
    ensure_dir(labels_root)

    if not id_to_name:
        (labels_root / "classes.txt").write_text("", encoding="utf-8")
        return

    lines = []
    for i in sorted(id_to_name.keys()):
        lines.append(clean_class_name(id_to_name[i]))

    (labels_root / "classes.txt").write_text("\n".join(lines), encoding="utf-8")


def write_classes_txt(labels_root: Path, index_to_disease: Dict[int, str]):
    """
    classes.txt line number must match YOLO class_id.

    Since class_id = metadata Index, classes.txt is written according to Index.
    If Index is not continuous, unused_class_i is inserted.
    """
    ensure_dir(labels_root)

    if not index_to_disease:
        (labels_root / "classes.txt").write_text("", encoding="utf-8")
        return

    max_idx = max(index_to_disease.keys())

    lines = []

    for i in range(max_idx + 1):
        if i in index_to_disease:
            lines.append(clean_class_name(index_to_disease[i]))
        else:
            lines.append(f"unused_class_{i}")

    (labels_root / "classes.txt").write_text("\n".join(lines), encoding="utf-8")


def write_data_yaml_from_mapping(out_root: Path, id_to_name: Dict[int, str]):
    """
    Write data.yaml using continuous class ids.
    """
    lines = []
    lines.append(f"path: {out_root.as_posix()}")
    lines.append("train: images")
    lines.append("val: images")
    lines.append("test: images")
    lines.append("")
    lines.append("names:")

    for i in sorted(id_to_name.keys()):
        lines.append(f"  {i}: {clean_class_name(id_to_name[i])}")

    (out_root / "data.yaml").write_text("\n".join(lines), encoding="utf-8")


def write_data_yaml(out_root: Path, index_to_disease: Dict[int, str]):
    """
    This dataset has no split at this conversion stage.
    Therefore train/val/test all point to images temporarily.

    You can split the dataset later if needed.
    """
    max_idx = max(index_to_disease.keys()) if index_to_disease else -1

    lines = []
    lines.append(f"path: {out_root.as_posix()}")
    lines.append("train: images")
    lines.append("val: images")
    lines.append("test: images")
    lines.append("")
    lines.append("names:")

    for i in range(max_idx + 1):
        if i in index_to_disease:
            name = clean_class_name(index_to_disease[i])
        else:
            name = f"unused_class_{i}"

        lines.append(f"  {i}: {name}")

    (out_root / "data.yaml").write_text("\n".join(lines), encoding="utf-8")


def build_metadata_maps(
    df: pd.DataFrame,
    name_col: str,
    index_col: str,
    disease_col: str,
    label_file_col: str,
):
    """
    Build:
      image Name -> metadata row
      Index -> Disease
    """
    required_cols = [name_col, index_col, disease_col, label_file_col]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Column not found in metadata.csv: {col}. "
                f"Available columns: {df.columns.tolist()}"
            )

    meta_by_image_name: Dict[str, dict] = {}
    index_to_disease: Dict[int, str] = {}
    metadata_errors = []

    for row_idx, row in df.iterrows():
        image_name = str(row[name_col]).strip()
        label_file = str(row[label_file_col]).strip()
        disease = str(row[disease_col]).strip()

        try:
            class_id = int(row[index_col])
        except Exception:
            metadata_errors.append(
                {
                    "metadata_row": row_idx,
                    "image_name": image_name,
                    "label_file": label_file,
                    "reason": "invalid_index",
                    "raw_index": row[index_col],
                }
            )
            continue

        if class_id < 0:
            metadata_errors.append(
                {
                    "metadata_row": row_idx,
                    "image_name": image_name,
                    "label_file": label_file,
                    "reason": "negative_index",
                    "raw_index": row[index_col],
                }
            )
            continue

        if class_id in index_to_disease:
            old_name = clean_class_name(index_to_disease[class_id])
            new_name = clean_class_name(disease)

            if old_name != new_name:
                metadata_errors.append(
                    {
                        "metadata_row": row_idx,
                        "image_name": image_name,
                        "label_file": label_file,
                        "reason": "same_index_different_disease",
                        "class_id": class_id,
                        "previous_disease": index_to_disease[class_id],
                        "current_disease": disease,
                    }
                )
        else:
            index_to_disease[class_id] = disease

        key = normalize_key(image_name)

        if key in meta_by_image_name:
            metadata_errors.append(
                {
                    "metadata_row": row_idx,
                    "image_name": image_name,
                    "label_file": label_file,
                    "reason": "duplicate_image_name_in_metadata",
                    "key": key,
                }
            )

        meta_by_image_name[key] = {
            "metadata_row": int(row_idx),
            "image_name": image_name,
            "label_file": label_file,
            "disease": disease,
            "class_id": int(class_id),
        }

    return meta_by_image_name, index_to_disease, metadata_errors


def delete_orphan_annotations(
    annotations_root: Path,
    used_mask_paths: set,
) -> List[dict]:
    """
    Delete annotation files that are not used by any existing image.

    orphan annotation means:
      annotation exists in annotations_root,
      but no existing image from images_root uses it through metadata Name + Label file.
    """
    rows = []
    annotation_files = collect_files(annotations_root, MASK_EXTS)
    used_resolved = {Path(p).resolve() for p in used_mask_paths}

    for p in annotation_files:
        if p.resolve() in used_resolved:
            continue

        row = {
            "annotation": str(p),
            "reason": "annotation_has_no_existing_image_match",
            "action": "delete",
        }

        try:
            p.unlink()
            row["status"] = "deleted"
        except Exception as e:
            row["status"] = "delete_failed"
            row["error"] = str(e)

        rows.append(row)

    return rows


# ============================================================
# Visualization helpers
# ============================================================

def color_for_class(class_id: int):
    """
    Deterministic BGR color for each class.
    """
    palette = [
        (0, 0, 255),
        (0, 128, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 128, 0),
        (255, 0, 0),
        (255, 0, 255),
        (128, 0, 255),
        (128, 128, 0),
        (0, 128, 128),
    ]
    return palette[int(class_id) % len(palette)]


def clip_box_to_int(box, img_w: int, img_h: int):
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


def draw_bbox_with_label(
    image: np.ndarray,
    box: dict,
    class_id: int,
    class_name: str,
    box_id: int,
):
    """
    Draw GT bbox and class name on image.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = clip_box_to_int(
        [box["x1"], box["y1"], box["x2"], box["y2"]],
        img_w=w,
        img_h=h,
    )

    color = color_for_class(class_id)
    label = f"{class_name} #{box_id}"

    thickness = max(2, int(round(min(w, h) / 500)))
    font_scale = max(0.5, min(w, h) / 1200)
    font_thickness = max(1, thickness)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    (text_w, text_h), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        font_thickness,
    )

    text_x = x1
    text_y = max(text_h + baseline + 4, y1 - 4)

    bg_x1 = text_x
    bg_y1 = text_y - text_h - baseline - 4
    bg_x2 = min(w, text_x + text_w + 6)
    bg_y2 = min(h, text_y + baseline + 2)

    cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)

    cv2.putText(
        image,
        label,
        (text_x + 3, text_y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA,
    )


def save_gt_visual(
    img_path: Path,
    boxes: List[dict],
    class_id: int,
    disease_name: str,
    out_path: Path,
):
    """
    Save image with GT bbox visualization.
    """
    img = cv2.imread(str(img_path))

    if img is None:
        return False

    class_name = clean_class_name(disease_name)

    for box_id, box in enumerate(boxes):
        draw_bbox_with_label(
            image=img,
            box=box,
            class_id=class_id,
            class_name=class_name,
            box_id=box_id,
        )

    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), img)
    return True


# ============================================================
# Filtering helpers
# ============================================================

def delete_file(path: Optional[Path], reason: str, rows: List[dict], deleted_cache: set):
    if path is None:
        rows.append(
            {
                "file": "",
                "reason": reason,
                "status": "not_found",
            }
        )
        return

    resolved = str(path.resolve()) if path.exists() else str(path)

    if resolved in deleted_cache:
        rows.append(
            {
                "file": str(path),
                "reason": reason,
                "status": "already_deleted",
            }
        )
        return

    if not path.exists():
        rows.append(
            {
                "file": str(path),
                "reason": reason,
                "status": "not_found",
            }
        )
        return

    try:
        path.unlink()
        deleted_cache.add(resolved)
        rows.append(
            {
                "file": str(path),
                "reason": reason,
                "status": "deleted",
            }
        )
    except Exception as e:
        rows.append(
            {
                "file": str(path),
                "reason": reason,
                "status": "delete_failed",
                "error": str(e),
            }
        )


def rewrite_label_file(label_path: Path, old_to_new: Dict[int, int], rows: List[dict]):
    if not label_path.exists():
        rows.append(
            {
                "label": str(label_path),
                "status": "not_found",
            }
        )
        return

    old_lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    new_lines = []
    rewritten = 0
    skipped = 0

    for line in old_lines:
        raw = line.strip()
        if not raw:
            continue

        parts = raw.split()
        if len(parts) < 5:
            skipped += 1
            continue

        try:
            old_cls = int(float(parts[0]))
        except Exception:
            skipped += 1
            continue

        if old_cls not in old_to_new:
            skipped += 1
            continue

        parts[0] = str(old_to_new[old_cls])
        new_lines.append(" ".join(parts))
        rewritten += 1

    label_path.write_text("\n".join(new_lines), encoding="utf-8")

    rows.append(
        {
            "label": str(label_path),
            "status": "rewritten",
            "rewritten_lines": rewritten,
            "skipped_lines": skipped,
        }
    )


def apply_class_filter(
    out_root: Path,
    images_root: Path,
    annotations_root: Path,
    image_records: List[dict],
    class_stats: Dict[int, dict],
    min_image_count: int,
    max_boxes_per_image: float,
):
    labels_root = out_root / "labels"

    kept_classes = {}
    removed_classes = {}

    for class_id, stat in sorted(class_stats.items()):
        image_count = int(stat["image_count"])
        box_count = int(stat["box_count"])
        boxes_per_image = box_count / image_count if image_count > 0 else 0.0

        stat_row = {
            "class_id": class_id,
            "disease": stat["disease"],
            "clean_disease": stat["clean_disease"],
            "image_count": image_count,
            "box_count": box_count,
            "boxes_per_image": boxes_per_image,
        }

        if image_count >= min_image_count and boxes_per_image <= max_boxes_per_image:
            kept_classes[class_id] = stat_row
        else:
            reasons = []
            if image_count < min_image_count:
                reasons.append(f"image_count<{min_image_count}")
            if boxes_per_image > max_boxes_per_image:
                reasons.append(f"boxes_per_image>{max_boxes_per_image}")

            stat_row["remove_reason"] = ";".join(reasons)
            removed_classes[class_id] = stat_row

    old_to_new = {}
    new_to_name = {}

    for new_id, old_id in enumerate(sorted(kept_classes.keys())):
        old_to_new[old_id] = new_id
        new_to_name[new_id] = kept_classes[old_id]["clean_disease"]

    delete_rows = []
    rewrite_rows = []
    kept_image_rows = []
    removed_image_rows = []
    deleted_cache = set()

    for rec in image_records:
        old_class_id = int(rec["class_id"])

        if old_class_id in removed_classes:
            reason = (
                f"remove_class: old_class_id={old_class_id}, "
                f"disease={rec['disease']}, "
                f"{removed_classes[old_class_id]['remove_reason']}"
            )

            delete_file(Path(rec["image_path"]), reason, delete_rows, deleted_cache)
            delete_file(Path(rec["annotation_path"]), reason, delete_rows, deleted_cache)
            delete_file(Path(rec["label_path"]), reason, delete_rows, deleted_cache)
            delete_file(Path(rec["mask_path"]), reason, delete_rows, deleted_cache)
            delete_file(Path(rec["visual_path"]), reason, delete_rows, deleted_cache)

            removed_image_rows.append(
                {
                    "image_name": rec["image_name"],
                    "old_class_id": old_class_id,
                    "disease": rec["disease"],
                    "reason": removed_classes[old_class_id]["remove_reason"],
                }
            )

        elif old_class_id in kept_classes:
            new_class_id = old_to_new[old_class_id]
            rewrite_label_file(Path(rec["label_path"]), old_to_new, rewrite_rows)

            kept_image_rows.append(
                {
                    "image_name": rec["image_name"],
                    "old_class_id": old_class_id,
                    "new_class_id": new_class_id,
                    "disease": rec["disease"],
                    "clean_disease": clean_class_name(rec["disease"]),
                }
            )

    kept_classes_rows = []
    for old_id in sorted(kept_classes.keys()):
        row = dict(kept_classes[old_id])
        row["old_class_id"] = old_id
        row["new_class_id"] = old_to_new[old_id]
        kept_classes_rows.append(row)

    removed_classes_rows = []
    for old_id in sorted(removed_classes.keys()):
        row = dict(removed_classes[old_id])
        row["old_class_id"] = old_id
        removed_classes_rows.append(row)

    filtered_class_stat_rows = []
    for old_id in sorted(kept_classes.keys()):
        row = dict(kept_classes[old_id])
        row["old_class_id"] = old_id
        row["class_id"] = old_to_new[old_id]
        filtered_class_stat_rows.append(row)

    # Rewrite classes.txt and data.yaml using continuous new ids.
    write_classes_txt_from_mapping(labels_root, new_to_name)
    write_data_yaml_from_mapping(out_root, new_to_name)

    # Filter and rewrite conversion_report.csv.
    conversion_report_path = out_root / "conversion_report.csv"
    if conversion_report_path.exists():
        try:
            df_conv = pd.read_csv(conversion_report_path)
            if "class_id" in df_conv.columns:
                df_conv["old_class_id"] = df_conv["class_id"].astype(int)
                df_conv = df_conv[df_conv["old_class_id"].isin(old_to_new.keys())].copy()
                df_conv["class_id"] = df_conv["old_class_id"].map(old_to_new)
                df_conv.to_csv(
                    out_root / "conversion_report_filtered.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
        except Exception as e:
            write_csv(
                out_root / "conversion_report_filter_error.csv",
                [{"error": str(e)}],
            )

    write_csv(out_root / "removed_classes.csv", removed_classes_rows)
    write_csv(out_root / "kept_classes.csv", kept_classes_rows)
    write_csv(out_root / "class_mapping_old_to_new.csv", kept_classes_rows)
    write_csv(out_root / "class_statistics_filtered.csv", filtered_class_stat_rows)
    write_csv(out_root / "filter_delete_report.csv", delete_rows)
    write_csv(out_root / "filter_label_rewrite_report.csv", rewrite_rows)
    write_csv(out_root / "kept_images_after_filter.csv", kept_image_rows)
    write_csv(out_root / "removed_images_after_filter.csv", removed_image_rows)

    filter_summary = {
        "min_image_count": min_image_count,
        "max_boxes_per_image": max_boxes_per_image,
        "total_classes_before_filter": len(class_stats),
        "kept_classes": len(kept_classes),
        "removed_classes": len(removed_classes),
        "kept_images": len(kept_image_rows),
        "removed_images": len(removed_image_rows),
        "deleted_file_records": len(delete_rows),
        "rewritten_label_records": len(rewrite_rows),
        "note": (
            "Removed classes are deleted from images, annotations, labels, masks, and visuals. "
            "Kept classes are remapped to continuous YOLO class ids."
        ),
    }

    (out_root / "filter_summary.json").write_text(
        json.dumps(filter_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return filter_summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata-csv", default="./data/PlantSeg/Metadata.csv")
    parser.add_argument("--images-root", default="./data/PlantSeg/images")
    parser.add_argument("--annotations-root", default="./data/PlantSeg/annotations")
    parser.add_argument("--out-root", required=True)

    parser.add_argument("--name-col", default="Name")
    parser.add_argument("--index-col", default="Index")
    parser.add_argument("--disease-col", default="Disease")
    parser.add_argument("--label-file-col", default="Label file")

    parser.add_argument("--min-image-count", type=int, default=30)
    parser.add_argument("--max-boxes-per-image", type=float, default=20.0)

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Disable class filtering. If not set, classes are filtered after conversion.",
    )

    args = parser.parse_args()

    metadata_csv = Path(args.metadata_csv).resolve()
    images_root = Path(args.images_root).resolve()
    annotations_root = Path(args.annotations_root).resolve()
    out_root = Path(args.out_root).resolve()

    out_masks_dir = out_root / "masks"
    out_labels_dir = out_root / "labels"
    out_visuals_dir = out_root / "visuals"

    ensure_dir(out_masks_dir)
    ensure_dir(out_labels_dir)
    ensure_dir(out_visuals_dir)

    df = pd.read_csv(metadata_csv)

    meta_by_image_name, index_to_disease, metadata_errors = build_metadata_maps(
        df=df,
        name_col=args.name_col,
        index_col=args.index_col,
        disease_col=args.disease_col,
        label_file_col=args.label_file_col,
    )

    image_files = collect_files(images_root, IMG_EXTS)

    mask_lookup = build_file_lookup(annotations_root, MASK_EXTS)
    mask_stem_lookup = build_stem_lookup(annotations_root, MASK_EXTS)

    conversion_rows = []
    missing_rows = []
    empty_rows = []
    used_mask_paths = set()

    # Class-level statistics.
    # image_count = number of images belonging to this class.
    # box_count = number of YOLO boxes generated for this class.
    class_stats: Dict[int, dict] = {}

    # Per-image records used by filtering stage.
    image_records = []

    print("[INFO] metadata:", metadata_csv)
    print("[INFO] images_root:", images_root)
    print("[INFO] annotations_root:", annotations_root)
    print("[INFO] out_root:", out_root)
    print("[INFO] images found:", len(image_files))
    print("[INFO] classes found:", len(index_to_disease))
    print("[INFO] min_image_count:", args.min_image_count)
    print("[INFO] max_boxes_per_image:", args.max_boxes_per_image)
    print("[INFO] class_filter:", not args.dry_run)

    for idx, img_path in enumerate(image_files, start=1):
        image_key = normalize_key(img_path.name)
        meta = meta_by_image_name.get(image_key)

        if meta is None:
            stem = normalize_stem(img_path.name)
            candidates = [
                v for _, v in meta_by_image_name.items()
                if normalize_stem(v["image_name"]) == stem
            ]

            if len(candidates) == 1:
                meta = candidates[0]

        if meta is None:
            missing_rows.append(
                {
                    "image": str(img_path),
                    "reason": "image_exists_but_not_found_in_metadata_name",
                }
            )
            continue

        class_id = meta["class_id"]
        disease = meta["disease"]

        if class_id not in class_stats:
            class_stats[class_id] = {
                "class_id": class_id,
                "disease": disease,
                "clean_disease": clean_class_name(disease),
                "image_count": 0,
                "box_count": 0,
            }

        class_stats[class_id]["image_count"] += 1

        mask_path = find_by_name(
            meta["label_file"],
            mask_lookup,
            mask_stem_lookup,
        )

        if mask_path is None:
            missing_rows.append(
                {
                    "image": str(img_path),
                    "metadata_image_name": meta["image_name"],
                    "label_file": meta["label_file"],
                    "class_id": meta["class_id"],
                    "disease": meta["disease"],
                    "reason": "metadata_label_file_not_found_in_annotations",
                }
            )
            continue

        used_mask_paths.add(mask_path.resolve())

        try:
            W, H = read_image_size(img_path)
            mask_arr = load_mask(mask_path)
        except Exception as e:
            missing_rows.append(
                {
                    "image": str(img_path),
                    "mask": str(mask_path),
                    "class_id": meta["class_id"],
                    "disease": meta["disease"],
                    "reason": "read_error",
                    "error": str(e),
                }
            )
            continue

        if mask_arr.shape[0] != H or mask_arr.shape[1] != W:
            mask_arr = np.array(
                Image.fromarray(mask_arr).resize((W, H), resample=Image.NEAREST)
            )

        binary = (mask_arr > 0).astype(np.uint8)

        out_mask_path = out_masks_dir / Path(img_path.name).with_suffix(".png")
        out_label_path = out_labels_dir / Path(img_path.name).with_suffix(".txt")
        out_visual_path = out_visuals_dir / Path(img_path.name).with_suffix(".jpg")

        Image.fromarray(make_visible_mask(mask_arr)).save(out_mask_path)

        boxes = connected_components_to_bboxes(binary)

        class_stats[class_id]["box_count"] += len(boxes)

        image_records.append(
            {
                "image_name": img_path.name,
                "image_path": str(img_path),
                "annotation_path": str(mask_path),
                "label_path": str(out_label_path),
                "mask_path": str(out_mask_path),
                "visual_path": str(out_visual_path),
                "class_id": class_id,
                "disease": disease,
                "box_count": len(boxes),
            }
        )

        if not boxes:
            out_label_path.write_text("", encoding="utf-8")
            empty_rows.append(
                {
                    "image": str(img_path),
                    "mask": str(mask_path),
                    "class_id": meta["class_id"],
                    "disease": meta["disease"],
                    "nonzero_pixels": int(binary.sum()),
                    "reason": "empty_mask",
                }
            )
            continue

        yolo_lines = []

        for box_id, box in enumerate(boxes):
            xc, yc, bw, bh = xyxy_to_yolo(
                box["x1"], box["y1"], box["x2"], box["y2"], W, H
            )

            line = f"{meta['class_id']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
            yolo_lines.append(line)

            conversion_rows.append(
                {
                    "image": str(img_path),
                    "image_name": img_path.name,
                    "mask": str(mask_path),
                    "label_file": meta["label_file"],
                    "class_id": meta["class_id"],
                    "disease": meta["disease"],
                    "box_id": box_id,
                    "x1": box["x1"],
                    "y1": box["y1"],
                    "x2": box["x2"],
                    "y2": box["y2"],
                    "area_px": box["area_px"],
                    "yolo_line": line,
                    "out_label_path": str(out_label_path),
                    "out_mask_path": str(out_mask_path),
                    "out_visual_path": str(out_visual_path),
                }
            )

        out_label_path.write_text("\n".join(yolo_lines), encoding="utf-8")

        visual_ok = save_gt_visual(
            img_path=img_path,
            boxes=boxes,
            class_id=class_id,
            disease_name=disease,
            out_path=out_visual_path,
        )

        if not visual_ok:
            missing_rows.append(
                {
                    "image": str(img_path),
                    "class_id": class_id,
                    "disease": disease,
                    "reason": "visual_image_read_failed",
                }
            )

        print(
            f"[{idx}/{len(image_files)}] "
            f"{img_path.name} -> {len(boxes)} boxes, visual={visual_ok}"
        )

    write_classes_txt(out_labels_dir, index_to_disease)
    write_data_yaml(out_root, index_to_disease)

    orphan_rows = delete_orphan_annotations(
        annotations_root=annotations_root,
        used_mask_paths=used_mask_paths,
    )

    class_rows = []
    for class_id in sorted(index_to_disease.keys()):
        class_rows.append(
            {
                "class_id": class_id,
                "disease": index_to_disease[class_id],
                "clean_disease": clean_class_name(index_to_disease[class_id]),
            }
        )

    class_stat_rows = [
        class_stats[class_id]
        for class_id in sorted(class_stats.keys())
    ]

    write_csv(out_root / "conversion_report.csv", conversion_rows)
    write_csv(out_root / "missing_files.csv", missing_rows)
    write_csv(out_root / "empty_masks.csv", empty_rows)
    write_csv(out_root / "metadata_errors.csv", metadata_errors)
    write_csv(out_root / "class_mapping_index_disease.csv", class_rows)
    write_csv(out_root / "class_statistics.csv", class_stat_rows)
    write_csv(out_root / "deleted_orphan_annotations.csv", orphan_rows)

    filter_summary = None

    if not args.dry_run:
        filter_summary = apply_class_filter(
            out_root=out_root,
            images_root=images_root,
            annotations_root=annotations_root,
            image_records=image_records,
            class_stats=class_stats,
            min_image_count=args.min_image_count,
            max_boxes_per_image=args.max_boxes_per_image,
        )

    summary = {
        "metadata_csv": str(metadata_csv),
        "images_root": str(images_root),
        "annotations_root": str(annotations_root),
        "out_root": str(out_root),
        "num_images_found": len(image_files),
        "num_classes": len(index_to_disease),
        "num_classes_with_images": len(class_stats),
        "num_converted_boxes": len(conversion_rows),
        "num_missing": len(missing_rows),
        "num_empty_masks": len(empty_rows),
        "num_metadata_errors": len(metadata_errors),
        "num_deleted_orphan_annotations": len(orphan_rows),
        "class_filter_enabled": not args.dry_run,
        "min_image_count": args.min_image_count,
        "max_boxes_per_image": args.max_boxes_per_image,
        "filter_summary": filter_summary,
        "logic": (
            "Use images as primary source. "
            "Match image by metadata Name. "
            "Use metadata Label file to find annotation. "
            "Use metadata Index as YOLO class_id before filtering. "
            "Use metadata Disease to generate labels/classes.txt. "
            "Save visible masks to masks/. "
            "Save YOLO labels to labels/. "
            "Save bbox visualizations with class names to visuals/. "
            "Delete annotations that have no matched existing image. "
            "Output class_statistics.csv with image_count and box_count per class. "
            "If class filtering is enabled, remove classes that violate image_count and boxes_per_image rules, "
            "delete their image/annotation/label/mask/visual files, and remap kept classes to continuous ids."
        ),
    }

    (out_root / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n[DONE]")
    print("out_root:", out_root)
    print("masks:", out_root / "masks")
    print("labels:", out_root / "labels")
    print("visuals:", out_root / "visuals")
    print("classes.txt:", out_root / "labels" / "classes.txt")
    print("data.yaml:", out_root / "data.yaml")
    print("conversion report:", out_root / "conversion_report.csv")
    print("class statistics:", out_root / "class_statistics.csv")
    print("converted boxes before filter:", len(conversion_rows))
    print("missing:", len(missing_rows))
    print("empty masks:", len(empty_rows))
    print("metadata errors:", len(metadata_errors))
    print("deleted orphan annotations:", len(orphan_rows))

    if filter_summary:
        print("filter summary:", out_root / "filter_summary.json")
        print("removed classes:", out_root / "removed_classes.csv")
        print("kept classes:", out_root / "kept_classes.csv")
        print("class statistics filtered:", out_root / "class_statistics_filtered.csv")
        print("class mapping old to new:", out_root / "class_mapping_old_to_new.csv")
        print("filter delete report:", out_root / "filter_delete_report.csv")
        print("filter label rewrite report:", out_root / "filter_label_rewrite_report.csv")


if __name__ == "__main__":
    main()