import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Fixed filtering rule for selected multi-class PlantSeg.
MIN_IMAGE_COUNT = 30
MAX_BOXES_PER_IMAGE = 20.0


def info(msg: str):
    print(f"[INFO] {msg}", flush=True)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def normalize_stem(name: str) -> str:
    return Path(str(name)).stem.lower()


def collect_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        return []

    return sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def read_classes(classes_path: Path) -> Dict[int, str]:
    if not classes_path.exists():
        info(f"classes.txt not found: {classes_path}")
        return {}

    lines = classes_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out = {}

    for i, line in enumerate(lines):
        name = line.strip()
        if name:
            out[i] = name

    info(f"Loaded {len(out)} classes from {classes_path}")
    return out


def read_yolo_label(label_path: Path) -> List[List[str]]:
    if not label_path.exists():
        return []

    rows = []

    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = line.strip()

        if not raw:
            continue

        parts = raw.split()

        if len(parts) < 5:
            continue

        try:
            int(float(parts[0]))
        except Exception:
            continue

        rows.append(parts)

    return rows


def find_sidecar_by_stem(
    root: Path,
    image_stem: str,
    preferred_ext: Optional[str] = None,
) -> Optional[Path]:
    if not root.exists():
        return None

    if preferred_ext:
        p = root / f"{image_stem}{preferred_ext}"
        if p.exists():
            return p

    matches = [
        p for p in root.iterdir()
        if p.is_file() and normalize_stem(p.name) == image_stem.lower()
    ]

    if matches:
        return matches[0]

    return None


def write_csv(path: Path, rows: List[dict]):
    ensure_dir(path.parent)

    if not rows:
        path.write_text("", encoding="utf-8")
        info(f"Wrote empty CSV: {path}")
        return

    fieldnames = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    info(f"Wrote CSV: {path} ({len(rows)} rows)")


def write_classes_txt(labels_root: Path, id_to_name: Dict[int, str]):
    ensure_dir(labels_root)

    lines = []
    for i in sorted(id_to_name.keys()):
        lines.append(id_to_name[i])

    out_path = labels_root / "classes.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    info(f"Wrote classes.txt: {out_path} ({len(lines)} classes)")


def resolve_yaml_path(yaml_arg: str, fallback_parent: Path) -> Path:
    p = Path(yaml_arg)

    if p.parent == Path(".") and not p.is_absolute():
        return (fallback_parent / p).resolve()

    return p.resolve()


def write_data_yaml(
    yaml_path: Path,
    dataset_root: Path,
    id_to_name: Dict[int, str],
):
    ensure_dir(yaml_path.parent)

    lines = []
    lines.append(f"path: {dataset_root.as_posix()}")
    lines.append("train: images/train")
    lines.append("val: images/val")
    lines.append("test: images/test")
    lines.append("")
    lines.append("names:")

    for i in sorted(id_to_name.keys()):
        lines.append(f"  {i}: {id_to_name[i]}")

    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    info(f"Wrote data yaml: {yaml_path}")


def parse_ratio(ratio_text: str) -> Tuple[float, float, float]:
    parts = [float(x.strip()) for x in ratio_text.split(",")]

    if len(parts) != 3:
        raise ValueError("--split-ratio must contain three values, e.g. 0.7,0.2,0.1")

    s = sum(parts)

    if s <= 0:
        raise ValueError("Split ratio sum must be > 0")

    return parts[0] / s, parts[1] / s, parts[2] / s


def split_records(records: List[dict], split_ratio: Tuple[float, float, float], seed: int):
    rng = random.Random(seed)
    records = list(records)
    rng.shuffle(records)

    n = len(records)
    train_ratio, val_ratio, _ = split_ratio

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    if n_train + n_val > n:
        n_val = max(0, n - n_train)

    train = records[:n_train]
    val = records[n_train:n_train + n_val]
    test = records[n_train + n_val:]

    info(
        f"Split records: total={n}, train={len(train)}, val={len(val)}, test={len(test)}, seed={seed}"
    )

    return {
        "train": train,
        "val": val,
        "test": test,
    }


def scan_existing_yolo_dataset(source_root: Path, log_every: int = 200):
    images_dir = source_root / "images"
    labels_dir = source_root / "labels"
    masks_dir = source_root / "masks"
    visuals_dir = source_root / "visuals"
    annotations_dir = source_root / "annotations"

    info(f"Scanning source root: {source_root}")
    info(f"Images dir: {images_dir}")
    info(f"Labels dir: {labels_dir}")
    info(f"Masks dir: {masks_dir}")
    info(f"Visuals dir: {visuals_dir}")
    info(f"Annotations dir: {annotations_dir}")

    classes = read_classes(labels_dir / "classes.txt")
    image_paths = collect_images(images_dir)

    info(f"Found {len(image_paths)} image files")

    records = []
    missing_rows = []
    class_to_images = {}
    class_box_counts = {}

    for idx, img_path in enumerate(image_paths, start=1):
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"

        if not label_path.exists():
            missing_rows.append({
                "image": str(img_path),
                "label": str(label_path),
                "reason": "label_not_found",
            })
            continue

        label_rows = read_yolo_label(label_path)

        if not label_rows:
            missing_rows.append({
                "image": str(img_path),
                "label": str(label_path),
                "reason": "empty_or_invalid_label",
            })
            continue

        class_ids = [int(float(parts[0])) for parts in label_rows]
        unique_classes = sorted(set(class_ids))
        primary_class = unique_classes[0]

        record = {
            "image_path": img_path,
            "label_path": label_path,
            "mask_path": find_sidecar_by_stem(masks_dir, stem, ".png"),
            "visual_path": find_sidecar_by_stem(visuals_dir, stem, ".jpg"),
            "annotation_path": find_sidecar_by_stem(annotations_dir, stem),
            "image_name": img_path.name,
            "stem": stem,
            "label_rows": label_rows,
            "class_ids": class_ids,
            "unique_classes": unique_classes,
            "primary_class": primary_class,
            "num_boxes": len(label_rows),
            "num_boxes_before_exclude": len(label_rows),
            "multi_class_in_one_image": int(len(unique_classes) > 1),
        }

        records.append(record)

        for cid in unique_classes:
            class_to_images.setdefault(cid, set()).add(img_path.name)

        for cid in class_ids:
            class_box_counts[cid] = class_box_counts.get(cid, 0) + 1

        if idx == 1 or idx % log_every == 0 or idx == len(image_paths):
            info(
                f"Scanning labels: {idx}/{len(image_paths)} | valid={len(records)} | missing/empty={len(missing_rows)}"
            )

    class_stats = []
    all_class_ids = sorted(
        set(list(class_to_images.keys()) + list(class_box_counts.keys()))
    )

    for cid in all_class_ids:
        image_count = len(class_to_images.get(cid, set()))
        box_count = class_box_counts.get(cid, 0)
        boxes_per_image = box_count / image_count if image_count > 0 else 0.0

        class_stats.append({
            "class_id": cid,
            "class_name": classes.get(cid, f"class_{cid}"),
            "image_count": image_count,
            "box_count": box_count,
            "boxes_per_image": boxes_per_image,
        })

    info(
        f"Scan complete: valid_images={len(records)}, "
        f"missing_or_empty={len(missing_rows)}, classes={len(class_stats)}"
    )

    if class_stats:
        sorted_stats = sorted(class_stats, key=lambda x: x["image_count"])
        info("Smallest classes by image_count:")
        for row in sorted_stats[:5]:
            info(
                f"  class_id={row['class_id']} name={row['class_name']} "
                f"images={row['image_count']} boxes={row['box_count']} "
                f"boxes/img={row['boxes_per_image']:.2f}"
            )

        sorted_fragmented = sorted(class_stats, key=lambda x: x["boxes_per_image"], reverse=True)
        info("Most fragmented classes by boxes_per_image:")
        for row in sorted_fragmented[:5]:
            info(
                f"  class_id={row['class_id']} name={row['class_name']} "
                f"images={row['image_count']} boxes={row['box_count']} "
                f"boxes/img={row['boxes_per_image']:.2f}"
            )

    return {
        "records": records,
        "classes": classes,
        "class_stats": class_stats,
        "missing_rows": missing_rows,
    }


def compute_class_stats_from_records(
    records: List[dict],
    classes: Dict[int, str],
) -> List[dict]:
    class_to_images = {}
    class_box_counts = {}

    for rec in records:
        image_name = rec["image_name"]
        class_ids = [int(float(parts[0])) for parts in rec["label_rows"]]
        unique_classes = sorted(set(class_ids))

        for cid in unique_classes:
            class_to_images.setdefault(cid, set()).add(image_name)

        for cid in class_ids:
            class_box_counts[cid] = class_box_counts.get(cid, 0) + 1

    class_stats = []
    all_class_ids = sorted(
        set(list(class_to_images.keys()) + list(class_box_counts.keys()))
    )

    for cid in all_class_ids:
        image_count = len(class_to_images.get(cid, set()))
        box_count = class_box_counts.get(cid, 0)
        boxes_per_image = box_count / image_count if image_count > 0 else 0.0

        class_stats.append({
            "class_id": cid,
            "class_name": classes.get(cid, f"class_{cid}"),
            "image_count": image_count,
            "box_count": box_count,
            "boxes_per_image": boxes_per_image,
        })

    return class_stats


def filter_classes(class_stats: List[dict]):
    keep = {}
    remove = {}

    info(
        f"Applying fixed multiclass filter: image_count >= {MIN_IMAGE_COUNT}, "
        f"boxes_per_image <= {MAX_BOXES_PER_IMAGE}"
    )

    for row in class_stats:
        cid = int(row["class_id"])
        image_count = int(row["image_count"])
        boxes_per_image = float(row["boxes_per_image"])

        if image_count >= MIN_IMAGE_COUNT and boxes_per_image <= MAX_BOXES_PER_IMAGE:
            keep[cid] = row
        else:
            reasons = []

            if image_count < MIN_IMAGE_COUNT:
                reasons.append(f"image_count<{MIN_IMAGE_COUNT}")

            if boxes_per_image > MAX_BOXES_PER_IMAGE:
                reasons.append(f"boxes_per_image>{MAX_BOXES_PER_IMAGE}")

            row = dict(row)
            row["remove_reason"] = ";".join(reasons)
            remove[cid] = row

    old_to_new = {}
    new_to_name = {}

    for new_id, old_id in enumerate(sorted(keep.keys())):
        old_to_new[old_id] = new_id
        new_to_name[new_id] = keep[old_id]["class_name"]

    info(f"Filter result: kept_classes={len(keep)}, removed_classes={len(remove)}")

    if remove:
        info("First removed classes:")
        for old_id in sorted(remove.keys())[:10]:
            row = remove[old_id]
            info(
                f"  old_class={old_id} name={row['class_name']} "
                f"images={row['image_count']} boxes/img={row['boxes_per_image']:.2f} "
                f"reason={row['remove_reason']}"
            )

    return keep, remove, old_to_new, new_to_name


def parse_exclude_class_ids(text: str | None) -> set[int]:
    if not text:
        return set()

    out = set()
    for x in text.split(","):
        x = x.strip()
        if not x:
            continue
        out.add(int(x))

    return out


def parse_exclude_class_names(text: str | None, classes: Dict[int, str]) -> set[int]:
    if not text:
        return set()

    exclude_names = {x.strip().lower() for x in text.split(",") if x.strip()}
    exclude_ids = set()

    name_to_ids = {}
    for cid, name in classes.items():
        name_to_ids.setdefault(name.lower(), []).append(cid)

    for name in sorted(exclude_names):
        if name not in name_to_ids:
            info(f"[WARN] exclude class name not found in classes.txt: {name}")
            continue

        for cid in name_to_ids[name]:
            exclude_ids.add(cid)

    return exclude_ids


def make_excluded_class_rows(
    exclude_class_ids: set[int],
    classes: Dict[int, str],
) -> List[dict]:
    rows = []

    for cid in sorted(exclude_class_ids):
        rows.append({
            "class_id": cid,
            "class_name": classes.get(cid, f"class_{cid}"),
        })

    return rows


def filter_records_by_exclude_classes(
    records: List[dict],
    exclude_class_ids: set[int],
) -> Tuple[List[dict], List[dict]]:
    filtered_records = []
    skipped_records = []

    for rec in records:
        filtered_label_rows = []
        removed_label_rows = []

        for parts in rec["label_rows"]:
            cls_id = int(float(parts[0]))

            if cls_id in exclude_class_ids:
                removed_label_rows.append(parts)
                continue

            filtered_label_rows.append(parts)

        if not filtered_label_rows:
            skipped_records.append({
                "image_name": rec["image_name"],
                "stem": rec["stem"],
                "old_classes": ",".join(map(str, rec["unique_classes"])),
                "num_boxes_before": rec["num_boxes"],
                "num_boxes_removed": len(removed_label_rows),
                "num_boxes_after": 0,
                "reason": "all_labels_excluded",
            })
            continue

        new_class_ids = [int(float(parts[0])) for parts in filtered_label_rows]
        new_unique_classes = sorted(set(new_class_ids))

        new_rec = dict(rec)
        new_rec["label_rows"] = filtered_label_rows
        new_rec["class_ids"] = new_class_ids
        new_rec["unique_classes"] = new_unique_classes
        new_rec["primary_class"] = new_unique_classes[0]
        new_rec["num_boxes"] = len(filtered_label_rows)
        new_rec["num_boxes_after_exclude"] = len(filtered_label_rows)
        new_rec["num_boxes_removed_by_exclude"] = len(removed_label_rows)
        new_rec["multi_class_in_one_image"] = int(len(new_unique_classes) > 1)

        filtered_records.append(new_rec)

    return filtered_records, skipped_records


def filter_records_by_allowed_classes(
    records: List[dict],
    allowed_class_ids: set[int],
) -> Tuple[List[dict], List[dict]]:
    filtered_records = []
    skipped_records = []

    for rec in records:
        filtered_label_rows = []
        removed_label_rows = []

        for parts in rec["label_rows"]:
            cls_id = int(float(parts[0]))

            if cls_id not in allowed_class_ids:
                removed_label_rows.append(parts)
                continue

            filtered_label_rows.append(parts)

        if not filtered_label_rows:
            skipped_records.append({
                "image_name": rec["image_name"],
                "stem": rec["stem"],
                "old_classes": ",".join(map(str, rec["unique_classes"])),
                "num_boxes_before": rec["num_boxes"],
                "num_boxes_removed": len(removed_label_rows),
                "num_boxes_after": 0,
                "reason": "all_labels_removed_by_multiclass_filter",
            })
            continue

        new_class_ids = [int(float(parts[0])) for parts in filtered_label_rows]
        new_unique_classes = sorted(set(new_class_ids))

        new_rec = dict(rec)
        new_rec["label_rows"] = filtered_label_rows
        new_rec["class_ids"] = new_class_ids
        new_rec["unique_classes"] = new_unique_classes
        new_rec["primary_class"] = new_unique_classes[0]
        new_rec["num_boxes"] = len(filtered_label_rows)
        new_rec["num_boxes_after_multiclass_filter"] = len(filtered_label_rows)
        new_rec["num_boxes_removed_by_multiclass_filter"] = len(removed_label_rows)
        new_rec["multi_class_in_one_image"] = int(len(new_unique_classes) > 1)

        filtered_records.append(new_rec)

    return filtered_records, skipped_records


def make_single_label_lines(label_rows: List[List[str]]) -> List[str]:
    out = []

    for parts in label_rows:
        new_parts = list(parts)
        new_parts[0] = "0"
        out.append(" ".join(new_parts[:5]))

    return out


def make_multiclass_label_lines(
    label_rows: List[List[str]],
    old_to_new: Dict[int, int],
) -> List[str]:
    out = []

    for parts in label_rows:
        old_cls = int(float(parts[0]))

        if old_cls not in old_to_new:
            continue

        new_parts = list(parts)
        new_parts[0] = str(old_to_new[old_cls])
        out.append(" ".join(new_parts[:5]))

    return out


def copy_optional(src: Optional[Path], dst: Path) -> bool:
    if src is None or not src.exists():
        return False

    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def write_dataset_records(
    records_by_split: Dict[str, List[dict]],
    out_root: Path,
    mode: str,
    id_to_name: Dict[int, str],
    yaml_path: Path,
    old_to_new: Optional[Dict[int, int]] = None,
    copy_masks: bool = True,
    copy_visuals: bool = True,
    copy_annotations: bool = True,
    log_every: int = 200,
):
    summary_rows = []
    skipped_rows = []

    info(f"Writing {mode} dataset to: {out_root}")
    info(f"Writing {mode} yaml to: {yaml_path}")
    info(f"{mode} classes: {len(id_to_name)}")

    for split, records in records_by_split.items():
        info(f"Start writing split={split}, mode={mode}, records={len(records)}")

        out_images_dir = out_root / "images" / split
        out_labels_dir = out_root / "labels" / split
        out_masks_dir = out_root / "masks" / split
        out_visuals_dir = out_root / "visuals" / split
        out_annotations_dir = out_root / "annotations" / split

        ensure_dir(out_images_dir)
        ensure_dir(out_labels_dir)

        written_count = 0
        skipped_count = 0

        for idx, rec in enumerate(records, start=1):
            image_path: Path = rec["image_path"]
            stem = rec["stem"]

            if mode == "single":
                label_lines = make_single_label_lines(rec["label_rows"])
            else:
                if old_to_new is None:
                    raise ValueError("old_to_new is required for multiclass mode")
                label_lines = make_multiclass_label_lines(rec["label_rows"], old_to_new)

            if not label_lines:
                skipped_count += 1
                skipped_rows.append({
                    "image_name": rec["image_name"],
                    "mode": mode,
                    "reason": "no_label_lines_after_mapping",
                    "old_classes": ",".join(map(str, rec["unique_classes"])),
                })
                continue

            out_img = out_images_dir / image_path.name
            out_label = out_labels_dir / f"{stem}.txt"

            shutil.copy2(image_path, out_img)
            out_label.write_text("\n".join(label_lines), encoding="utf-8")

            copied_mask = False
            copied_visual = False
            copied_annotation = False

            if copy_masks and rec["mask_path"] is not None:
                copied_mask = copy_optional(
                    rec["mask_path"],
                    out_masks_dir / f"{stem}.png",
                )

            if copy_visuals and rec["visual_path"] is not None:
                copied_visual = copy_optional(
                    rec["visual_path"],
                    out_visuals_dir / f"{stem}.jpg",
                )

            if copy_annotations and rec["annotation_path"] is not None:
                copied_annotation = copy_optional(
                    rec["annotation_path"],
                    out_annotations_dir / rec["annotation_path"].name,
                )

            num_boxes_before = rec.get("num_boxes_before_exclude", rec.get("num_boxes", 0))
            num_boxes_after_exclude = rec.get("num_boxes_after_exclude", rec.get("num_boxes", 0))
            num_boxes_removed_by_exclude = rec.get("num_boxes_removed_by_exclude", 0)
            num_boxes_after_multiclass_filter = rec.get(
                "num_boxes_after_multiclass_filter",
                len(label_lines),
            )
            num_boxes_removed_by_multiclass_filter = rec.get(
                "num_boxes_removed_by_multiclass_filter",
                0,
            )

            summary_rows.append({
                "split": split,
                "mode": mode,
                "image_name": rec["image_name"],
                "old_classes": ",".join(map(str, rec["unique_classes"])),
                "primary_old_class": rec["primary_class"],
                "num_boxes_before_exclude": num_boxes_before,
                "num_boxes_after_exclude": num_boxes_after_exclude,
                "num_boxes_removed_by_exclude": num_boxes_removed_by_exclude,
                "num_boxes_after_multiclass_filter": num_boxes_after_multiclass_filter,
                "num_boxes_removed_by_multiclass_filter": num_boxes_removed_by_multiclass_filter,
                "num_boxes_after_write": len(label_lines),
                "multi_class_in_one_image": rec["multi_class_in_one_image"],
                "copied_mask": int(copied_mask),
                "copied_visual": int(copied_visual),
                "copied_annotation": int(copied_annotation),
            })

            written_count += 1

            if idx == 1 or idx % log_every == 0 or idx == len(records):
                info(
                    f"Writing {mode}/{split}: {idx}/{len(records)} | "
                    f"written={written_count} | skipped={skipped_count}"
                )

        info(
            f"Finished split={split}, mode={mode}: written={written_count}, skipped={skipped_count}"
        )

    write_classes_txt(out_root / "labels", id_to_name)
    write_data_yaml(
        yaml_path=yaml_path,
        dataset_root=out_root,
        id_to_name=id_to_name,
    )

    write_csv(out_root / "dataset_records.csv", summary_rows)
    write_csv(out_root / "skipped_records.csv", skipped_rows)

    total_images = len(summary_rows)
    total_boxes = sum(int(r["num_boxes_after_write"]) for r in summary_rows)

    split_counts = {}

    for r in summary_rows:
        split = r["split"]
        split_counts.setdefault(split, {"images": 0, "boxes": 0})
        split_counts[split]["images"] += 1
        split_counts[split]["boxes"] += int(r["num_boxes_after_write"])

    dataset_summary = {
        "out_root": str(out_root),
        "yaml_path": str(yaml_path),
        "mode": mode,
        "num_classes": len(id_to_name),
        "total_images": total_images,
        "total_boxes": total_boxes,
        "split_counts": split_counts,
        "classes": id_to_name,
    }

    (out_root / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    info(
        f"Finished {mode} dataset: images={total_images}, boxes={total_boxes}, "
        f"summary={out_root / 'dataset_summary.json'}"
    )

    return dataset_summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-root",
        default="./data/PlantSeg/",
        help="Existing YOLO-style PlantSeg root containing images/labels/masks/visuals.",
    )
    parser.add_argument("--out-single-root", default="./data/PlantSeg_singleclass_leaf")
    parser.add_argument("--out-multiclass-root", default="./data/PlantSeg_multiclass_leaf")

    parser.add_argument(
        "--single-yaml",
        default="./data/plantseg_singleclass_leaf.yaml",
        help="Output YAML path or filename for single-class dataset.",
    )
    parser.add_argument(
        "--multiclass-yaml",
        default="./data/plantseg_multiclass_leaf.yaml",
        help="Output YAML path or filename for selected multi-class dataset.",
    )

    parser.add_argument(
        "--split-ratio",
        default="0.7,0.2,0.1",
        help="train,val,test ratio, e.g. 0.7,0.2,0.1",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=200)

    parser.add_argument("--no-copy-masks", action="store_true")
    parser.add_argument("--no-copy-visuals", action="store_true")
    parser.add_argument("--no-copy-annotations", action="store_true")

    parser.add_argument(
        "--exclude-class-ids",
        default="4,5,8,12,16,27,30,50,52,55",
        help="Comma-separated old class ids to exclude, e.g. '4,5,8'.",
    )

    parser.add_argument(
        "--exclude-class-names",
        default="apple_scab,apple_rust,bell_pepper_blossom_end_rot,citrus_canker,citrus_greening_disease,cucumber_bacterial_wilt,peach_scab,peach_brown_rot,grape_black_rot",
        help="Comma-separated old class names to exclude, exact match, case-insensitive.",
    )

    args = parser.parse_args()

    info("========== PlantSeg dataset builder started ==========")

    source_root = Path(args.source_root).resolve()
    out_single_root = Path(args.out_single_root).resolve()
    out_multiclass_root = Path(args.out_multiclass_root).resolve()

    single_yaml_path = resolve_yaml_path(args.single_yaml, out_single_root)
    multiclass_yaml_path = resolve_yaml_path(args.multiclass_yaml, out_multiclass_root)

    info(f"source_root: {source_root}")
    info(f"out_single_root: {out_single_root}")
    info(f"out_multiclass_root: {out_multiclass_root}")
    info(f"single_yaml_path: {single_yaml_path}")
    info(f"multiclass_yaml_path: {multiclass_yaml_path}")
    info(f"split_ratio raw: {args.split_ratio}")
    info(f"seed: {args.seed}")
    info(f"log_every: {args.log_every}")
    info(f"copy_masks: {not args.no_copy_masks}")
    info(f"copy_visuals: {not args.no_copy_visuals}")
    info(f"copy_annotations: {not args.no_copy_annotations}")
    info(f"exclude_class_ids raw: {args.exclude_class_ids}")
    info(f"exclude_class_names raw: {args.exclude_class_names}")

    split_ratio = parse_ratio(args.split_ratio)

    info(
        f"split_ratio normalized: train={split_ratio[0]:.4f}, "
        f"val={split_ratio[1]:.4f}, test={split_ratio[2]:.4f}"
    )

    scan = scan_existing_yolo_dataset(source_root, log_every=args.log_every)

    records_before_exclude = scan["records"]
    classes = scan["classes"]
    class_stats_before_exclude = scan["class_stats"]

    exclude_ids_from_ids = parse_exclude_class_ids(args.exclude_class_ids)
    exclude_ids_from_names = parse_exclude_class_names(args.exclude_class_names, classes)
    exclude_class_ids = exclude_ids_from_ids | exclude_ids_from_names

    excluded_class_rows = make_excluded_class_rows(exclude_class_ids, classes)

    info(f"Exclude classes: ids={sorted(exclude_class_ids)}")
    if excluded_class_rows:
        for row in excluded_class_rows:
            info(f"  exclude class_id={row['class_id']} name={row['class_name']}")

    records_after_exclude, skipped_after_exclude = filter_records_by_exclude_classes(
        records=records_before_exclude,
        exclude_class_ids=exclude_class_ids,
    )

    info(
        f"After exclude: images_before={len(records_before_exclude)}, "
        f"images_after={len(records_after_exclude)}, "
        f"skipped={len(skipped_after_exclude)}"
    )

    class_stats_after_exclude = compute_class_stats_from_records(
        records=records_after_exclude,
        classes=classes,
    )

    # -----------------------------
    # Single-class dataset
    # -----------------------------
    single_records_by_split = split_records(
        records=records_after_exclude,
        split_ratio=split_ratio,
        seed=args.seed,
    )

    single_id_to_name = {0: "disease_region"}

    single_summary = write_dataset_records(
        records_by_split=single_records_by_split,
        out_root=out_single_root,
        mode="single",
        id_to_name=single_id_to_name,
        yaml_path=single_yaml_path,
        old_to_new=None,
        copy_masks=not args.no_copy_masks,
        copy_visuals=not args.no_copy_visuals,
        copy_annotations=not args.no_copy_annotations,
        log_every=args.log_every,
    )

    write_csv(out_single_root / "excluded_classes.csv", excluded_class_rows)
    write_csv(out_single_root / "skipped_after_exclude.csv", skipped_after_exclude)
    write_csv(out_single_root / "source_missing_labels.csv", scan["missing_rows"])
    write_csv(out_single_root / "source_class_statistics_before_exclude.csv", class_stats_before_exclude)
    write_csv(out_single_root / "source_class_statistics_after_exclude.csv", class_stats_after_exclude)

    # -----------------------------
    # Multiclass dataset
    # -----------------------------
    keep, remove, old_to_new, multi_id_to_name = filter_classes(class_stats_after_exclude)

    selected_records, skipped_after_multiclass_filter = filter_records_by_allowed_classes(
        records=records_after_exclude,
        allowed_class_ids=set(old_to_new.keys()),
    )

    info(
        f"Selected multiclass records after fixed class filter: "
        f"{len(selected_records)}/{len(records_after_exclude)} images kept"
    )

    multiclass_records_by_split = split_records(
        records=selected_records,
        split_ratio=split_ratio,
        seed=args.seed,
    )

    multiclass_summary = write_dataset_records(
        records_by_split=multiclass_records_by_split,
        out_root=out_multiclass_root,
        mode="multiclass",
        id_to_name=multi_id_to_name,
        yaml_path=multiclass_yaml_path,
        old_to_new=old_to_new,
        copy_masks=not args.no_copy_masks,
        copy_visuals=not args.no_copy_visuals,
        copy_annotations=not args.no_copy_annotations,
        log_every=args.log_every,
    )

    write_csv(out_multiclass_root / "excluded_classes.csv", excluded_class_rows)
    write_csv(out_multiclass_root / "skipped_after_exclude.csv", skipped_after_exclude)
    write_csv(out_multiclass_root / "skipped_after_multiclass_filter.csv", skipped_after_multiclass_filter)
    write_csv(out_multiclass_root / "source_missing_labels.csv", scan["missing_rows"])
    write_csv(out_multiclass_root / "source_class_statistics_before_exclude.csv", class_stats_before_exclude)
    write_csv(out_multiclass_root / "source_class_statistics_after_exclude.csv", class_stats_after_exclude)
    write_csv(out_multiclass_root / "removed_classes.csv", list(remove.values()))

    kept_rows = []

    for old_id, new_id in sorted(old_to_new.items(), key=lambda x: x[1]):
        stat = keep[old_id]
        kept_rows.append({
            "old_class_id": old_id,
            "new_class_id": new_id,
            "old_class_name": stat["class_name"],
            "new_class_name": multi_id_to_name[new_id],
            "image_count": stat["image_count"],
            "box_count": stat["box_count"],
            "boxes_per_image": stat["boxes_per_image"],
        })

    write_csv(out_multiclass_root / "kept_classes.csv", kept_rows)
    write_csv(out_multiclass_root / "class_mapping_old_to_new.csv", kept_rows)

    final_summary = {
        "source_root": str(source_root),
        "out_single_root": str(out_single_root),
        "out_multiclass_root": str(out_multiclass_root),
        "single_yaml_path": str(single_yaml_path),
        "multiclass_yaml_path": str(multiclass_yaml_path),
        "split_ratio": {
            "train": split_ratio[0],
            "val": split_ratio[1],
            "test": split_ratio[2],
        },
        "seed": args.seed,
        "log_every": args.log_every,
        "exclude_class_ids": sorted(exclude_class_ids),
        "excluded_classes": excluded_class_rows,
        "num_source_valid_images_before_exclude": len(records_before_exclude),
        "num_source_valid_images_after_exclude": len(records_after_exclude),
        "num_images_skipped_after_exclude": len(skipped_after_exclude),
        "num_source_classes_before_exclude": len(class_stats_before_exclude),
        "num_source_classes_after_exclude": len(class_stats_after_exclude),
        "single": single_summary,
        "multiclass": {
            **multiclass_summary,
            "fixed_min_image_count": MIN_IMAGE_COUNT,
            "fixed_max_boxes_per_image": MAX_BOXES_PER_IMAGE,
            "kept_classes": len(keep),
            "removed_classes": len(remove),
            "selected_images_before_split": len(selected_records),
            "num_images_skipped_after_multiclass_filter": len(skipped_after_multiclass_filter),
        },
        "note": (
            "This script reads existing YOLO labels and does not regenerate bboxes from annotations. "
            "First, it removes label rows whose old class ids/names are specified by "
            "--exclude-class-ids or --exclude-class-names. If all labels in an image are removed, "
            "the image record is discarded before train/val/test splitting. "
            "Single-class labels map all remaining boxes to class 0 disease_region. "
            "For multiclass, class statistics are recomputed after exclude filtering, then classes are "
            "kept only if image_count >= MIN_IMAGE_COUNT and boxes_per_image <= MAX_BOXES_PER_IMAGE. "
            "Multiclass labels are filtered at box level and remapped to continuous class ids."
        ),
    }

    (out_single_root / "build_summary.json").write_text(
        json.dumps(final_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_multiclass_root / "build_summary.json").write_text(
        json.dumps(final_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    info("========== PlantSeg dataset builder finished ==========")

    print("\n[DONE]")
    print("source root:", source_root)
    print("single dataset:", out_single_root)
    print("single yaml:", single_yaml_path)
    print("multiclass dataset:", out_multiclass_root)
    print("multiclass yaml:", multiclass_yaml_path)
    print("source valid images before exclude:", len(records_before_exclude))
    print("source valid images after exclude:", len(records_after_exclude))
    print("images skipped after exclude:", len(skipped_after_exclude))
    print("single images:", single_summary["total_images"])
    print("multiclass images:", multiclass_summary["total_images"])
    print("source classes before exclude:", len(class_stats_before_exclude))
    print("source classes after exclude:", len(class_stats_after_exclude))
    print("kept multiclass classes:", len(keep))
    print("removed multiclass classes:", len(remove))
    print("fixed filter: image_count >=", MIN_IMAGE_COUNT)
    print("fixed filter: boxes_per_image <=", MAX_BOXES_PER_IMAGE)


if __name__ == "__main__":
    main()