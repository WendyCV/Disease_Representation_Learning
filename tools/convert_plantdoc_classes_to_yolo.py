import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def info(msg: str):
    print(f"[INFO] {msg}", flush=True)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


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


def collect_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        return []

    return sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def collect_labels(labels_dir: Path) -> List[Path]:
    if not labels_dir.exists():
        return []

    return sorted(
        p for p in labels_dir.rglob("*.txt")
        if p.is_file() and p.name.lower() != "classes.txt"
    )


def read_image_size(image_path: Path):
    try:
        with Image.open(image_path) as img:
            return img.size  # W, H
    except Exception:
        return None, None


def read_classes_txt(labels_dir: Path) -> Dict[int, str]:
    candidates = [
        labels_dir / "classes.txt",
        labels_dir.parent / "classes.txt",
    ]

    for p in candidates:
        if p.exists():
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
            classes = {}

            for i, line in enumerate(lines):
                name = line.strip()
                if name:
                    classes[i] = name

            info(f"Loaded {len(classes)} classes from {p}")
            return classes

    info("No classes.txt found. Class names will be class_<id>.")
    return {}


def read_yolo_label(label_path: Path) -> List[dict]:
    rows = []

    if not label_path.exists():
        return rows

    lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    for line_idx, line in enumerate(lines):
        raw = line.strip()

        if not raw:
            continue

        parts = raw.split()

        if len(parts) < 5:
            rows.append({
                "valid": False,
                "line_idx": line_idx,
                "raw": raw,
                "reason": "less_than_5_columns",
            })
            continue

        try:
            cls = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except Exception:
            rows.append({
                "valid": False,
                "line_idx": line_idx,
                "raw": raw,
                "reason": "parse_error",
            })
            continue

        valid_range = (
            0 <= xc <= 1 and
            0 <= yc <= 1 and
            0 < w <= 1 and
            0 < h <= 1
        )

        rows.append({
            "valid": bool(valid_range),
            "line_idx": line_idx,
            "raw": raw,
            "class_id": cls,
            "xc": xc,
            "yc": yc,
            "w": w,
            "h": h,
            "area_norm": w * h,
            "reason": "" if valid_range else "out_of_yolo_range",
        })

    return rows


def yolo_to_xyxy(row: dict, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    xc = float(row["xc"]) * img_w
    yc = float(row["yc"]) * img_h
    bw = float(row["w"]) * img_w
    bh = float(row["h"]) * img_h

    x1 = int(round(xc - bw / 2))
    y1 = int(round(yc - bh / 2))
    x2 = int(round(xc + bw / 2))
    y2 = int(round(yc + bh / 2))

    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))

    if x2 <= x1:
        x2 = min(img_w - 1, x1 + 1)

    if y2 <= y1:
        y2 = min(img_h - 1, y1 + 1)

    return x1, y1, x2, y2


def color_for_class(class_id: int):
    palette = [
        (255, 0, 0),
        (255, 128, 0),
        (255, 255, 0),
        (0, 200, 0),
        (0, 180, 255),
        (0, 80, 255),
        (160, 0, 255),
        (255, 0, 255),
        (128, 128, 0),
        (0, 128, 128),
    ]
    return palette[int(class_id) % len(palette)]


def draw_gt_visual(
    image_path: Path,
    valid_boxes: List[dict],
    classes: Dict[int, str],
    out_path: Path,
) -> bool:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return False

    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for box_id, row in enumerate(valid_boxes):
        cls = int(row["class_id"])
        cls_name = classes.get(cls, f"class_{cls}")
        label = f"{cls_name} #{box_id}"

        x1, y1, x2, y2 = yolo_to_xyxy(row, img_w, img_h)
        color = color_for_class(cls)

        thickness = max(2, int(round(min(img_w, img_h) / 500)))

        for t in range(thickness):
            draw.rectangle(
                [x1 - t, y1 - t, x2 + t, y2 + t],
                outline=color,
            )

        if font is not None:
            try:
                bbox = draw.textbbox((x1, y1), label, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except Exception:
                text_w = len(label) * 7
                text_h = 12

            bg_y1 = max(0, y1 - text_h - 6)
            bg_y2 = min(img_h, bg_y1 + text_h + 6)
            bg_x1 = max(0, x1)
            bg_x2 = min(img_w, x1 + text_w + 8)

            draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=color)
            draw.text((bg_x1 + 4, bg_y1 + 3), label, fill=(255, 255, 255), font=font)

    ensure_dir(out_path.parent)
    img.save(out_path)
    return True


def safe_delete(path: Optional[Path], reason: str, rows: List[dict], dry_run: bool):
    if path is None:
        rows.append({
            "file": "",
            "reason": reason,
            "status": "not_found",
            "dry_run": int(dry_run),
        })
        return

    if not path.exists():
        rows.append({
            "file": str(path),
            "reason": reason,
            "status": "not_found",
            "dry_run": int(dry_run),
        })
        return

    if dry_run:
        rows.append({
            "file": str(path),
            "reason": reason,
            "status": "would_delete",
            "dry_run": int(dry_run),
        })
        return

    try:
        path.unlink()
        rows.append({
            "file": str(path),
            "reason": reason,
            "status": "deleted",
            "dry_run": int(dry_run),
        })
    except Exception as e:
        rows.append({
            "file": str(path),
            "reason": reason,
            "status": "delete_failed",
            "error": str(e),
            "dry_run": int(dry_run),
        })


def analyze_and_clean_flat_yolo(
    dataset_root: Path,
    out_dir: Path,
    filter_mode: str,
    small_area_thr: float,
    tiny_area_thr: float,
    high_boxes_per_image_thr: int,
    high_boxes_per_class_thr: float,
    low_image_class_thr: int,
    log_every: int,
    dry_run: bool,
):
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    visuals_dir = dataset_root / "visuals"

    ensure_dir(out_dir)
    ensure_dir(visuals_dir)

    info(f"dataset_root: {dataset_root}")
    info(f"images_dir: {images_dir}")
    info(f"labels_dir: {labels_dir}")
    info(f"visuals_dir: {visuals_dir}")
    info(f"out_dir: {out_dir}")
    info(f"filter_mode: {filter_mode}")
    info(f"dry_run: {dry_run}")

    classes = read_classes_txt(labels_dir)
    image_paths = collect_images(images_dir)

    info(f"Found images: {len(image_paths)}")
    info(f"small_area_thr: {small_area_thr}")
    info(f"tiny_area_thr: {tiny_area_thr}")
    info(f"high_boxes_per_image_thr: {high_boxes_per_image_thr}")
    info(f"high_boxes_per_class_thr: {high_boxes_per_class_thr}")
    info(f"low_image_class_thr: {low_image_class_thr}")

    class_stats = {}
    image_rows = []
    box_rows = []
    invalid_rows = []
    missing_label_rows = []
    empty_label_rows = []

    image_cache = {}

    # First pass: analyze labels and build class statistics.
    for idx, image_path in enumerate(image_paths, start=1):
        stem = image_path.stem
        label_path = labels_dir / f"{stem}.txt"

        if not label_path.exists():
            missing_label_rows.append({
                "image": str(image_path),
                "label": str(label_path),
                "reason": "label_not_found",
            })
            image_cache[str(image_path)] = {
                "image_path": image_path,
                "label_path": label_path,
                "valid_boxes": [],
                "invalid_boxes": [],
                "status": "missing_label",
                "unique_classes": [],
            }
            continue

        label_rows = read_yolo_label(label_path)

        valid_boxes = [r for r in label_rows if r.get("valid")]
        invalid_boxes = [r for r in label_rows if not r.get("valid")]

        for r in invalid_boxes:
            invalid_rows.append({
                "image": str(image_path),
                "label": str(label_path),
                **r,
            })

        if not valid_boxes:
            empty_label_rows.append({
                "image": str(image_path),
                "label": str(label_path),
                "reason": "empty_or_no_valid_boxes",
            })
            image_cache[str(image_path)] = {
                "image_path": image_path,
                "label_path": label_path,
                "valid_boxes": [],
                "invalid_boxes": invalid_boxes,
                "status": "empty_or_invalid_label",
                "unique_classes": [],
            }
            continue

        img_w, img_h = read_image_size(image_path)
        img_area = img_w * img_h if img_w and img_h else None

        class_ids = [int(r["class_id"]) for r in valid_boxes]
        unique_classes = sorted(set(class_ids))

        num_boxes = len(valid_boxes)
        num_small = sum(1 for r in valid_boxes if float(r["area_norm"]) < small_area_thr)
        num_tiny = sum(1 for r in valid_boxes if float(r["area_norm"]) < tiny_area_thr)

        image_rows.append({
            "image": str(image_path),
            "label": str(label_path),
            "image_name": image_path.name,
            "label_name": label_path.name,
            "img_w": img_w if img_w else "",
            "img_h": img_h if img_h else "",
            "num_boxes": num_boxes,
            "num_classes_in_image": len(unique_classes),
            "unique_classes": ",".join(map(str, unique_classes)),
            "num_small_boxes": num_small,
            "num_tiny_boxes": num_tiny,
            "small_box_ratio": num_small / num_boxes if num_boxes else 0,
            "tiny_box_ratio": num_tiny / num_boxes if num_boxes else 0,
            "is_high_box_image": int(num_boxes > high_boxes_per_image_thr),
        })

        image_cache[str(image_path)] = {
            "image_path": image_path,
            "label_path": label_path,
            "valid_boxes": valid_boxes,
            "invalid_boxes": invalid_boxes,
            "status": "valid",
            "unique_classes": unique_classes,
        }

        for box_id, r in enumerate(valid_boxes):
            cls = int(r["class_id"])
            class_name = classes.get(cls, f"class_{cls}")

            if cls not in class_stats:
                class_stats[cls] = {
                    "class_id": cls,
                    "class_name": class_name,
                    "image_set": set(),
                    "box_count": 0,
                    "small_box_count": 0,
                    "tiny_box_count": 0,
                }

            class_stats[cls]["image_set"].add(str(image_path))
            class_stats[cls]["box_count"] += 1

            if float(r["area_norm"]) < small_area_thr:
                class_stats[cls]["small_box_count"] += 1

            if float(r["area_norm"]) < tiny_area_thr:
                class_stats[cls]["tiny_box_count"] += 1

            area_px_est = ""
            if img_area:
                area_px_est = float(r["area_norm"]) * img_area

            box_rows.append({
                "image": str(image_path),
                "label": str(label_path),
                "image_name": image_path.name,
                "class_id": cls,
                "class_name": class_name,
                "box_id": box_id,
                "xc": r["xc"],
                "yc": r["yc"],
                "w": r["w"],
                "h": r["h"],
                "area_norm": r["area_norm"],
                "area_px_est": area_px_est,
                "is_small": int(float(r["area_norm"]) < small_area_thr),
                "is_tiny": int(float(r["area_norm"]) < tiny_area_thr),
            })

        if idx == 1 or idx % log_every == 0 or idx == len(image_paths):
            info(
                f"Processed images: {idx}/{len(image_paths)} | "
                f"valid_images={len(image_rows)} | boxes={len(box_rows)} | "
                f"missing={len(missing_label_rows)} | empty={len(empty_label_rows)}"
            )

    # Build class rows and decide removed classes.
    class_rows = []

    for cls in sorted(class_stats.keys()):
        stat = class_stats[cls]
        image_count = len(stat["image_set"])
        box_count = int(stat["box_count"])
        small_count = int(stat["small_box_count"])
        tiny_count = int(stat["tiny_box_count"])
        boxes_per_image = box_count / image_count if image_count else 0

        class_rows.append({
            "class_id": cls,
            "class_name": stat["class_name"],
            "image_count": image_count,
            "box_count": box_count,
            "boxes_per_image": boxes_per_image,
            "small_box_count": small_count,
            "tiny_box_count": tiny_count,
            "small_box_ratio": small_count / box_count if box_count else 0,
            "tiny_box_ratio": tiny_count / box_count if box_count else 0,
            "is_low_image_class": int(image_count < low_image_class_thr),
            "is_fragmented_class": int(boxes_per_image > high_boxes_per_class_thr),
        })

    fragmented_class_rows = sorted(
        [r for r in class_rows if int(r["is_fragmented_class"]) == 1],
        key=lambda x: x["boxes_per_image"],
        reverse=True,
    )

    low_image_class_rows = sorted(
        [r for r in class_rows if int(r["is_low_image_class"]) == 1],
        key=lambda x: x["image_count"],
    )

    high_box_image_rows = sorted(
        [r for r in image_rows if int(r["is_high_box_image"]) == 1],
        key=lambda x: x["num_boxes"],
        reverse=True,
    )

    small_box_heavy_image_rows = sorted(
        [
            r for r in image_rows
            if float(r["small_box_ratio"]) >= 0.5 and int(r["num_boxes"]) > 0
        ],
        key=lambda x: (x["small_box_ratio"], x["num_boxes"]),
        reverse=True,
    )

    tiny_box_heavy_image_rows = sorted(
        [
            r for r in image_rows
            if float(r["tiny_box_ratio"]) >= 0.3 and int(r["num_boxes"]) > 0
        ],
        key=lambda x: (x["tiny_box_ratio"], x["num_boxes"]),
        reverse=True,
    )

    remove_class_ids = set()

    if filter_mode == "multiclass":
        for r in class_rows:
            if int(r["is_low_image_class"]) == 1 or int(r["is_fragmented_class"]) == 1:
                remove_class_ids.add(int(r["class_id"]))

    info(f"Classes total: {len(class_rows)}")
    info(f"Low-image classes: {len(low_image_class_rows)}")
    info(f"Fragmented classes: {len(fragmented_class_rows)}")
    info(f"Classes to remove under mode={filter_mode}: {len(remove_class_ids)}")

    # Second pass: delete unqualified image/label and draw visuals for kept images.
    delete_rows = []
    kept_rows = []
    visual_rows = []

    for idx, image_path in enumerate(image_paths, start=1):
        stem = image_path.stem
        label_path = labels_dir / f"{stem}.txt"
        visual_path = visuals_dir / f"{stem}.jpg"

        cache = image_cache.get(str(image_path))

        delete_reason = ""

        if cache is None:
            delete_reason = "not_in_cache"
        elif cache["status"] == "missing_label":
            delete_reason = "missing_label"
        elif cache["status"] == "empty_or_invalid_label":
            delete_reason = "empty_or_invalid_label"
        elif filter_mode == "multiclass":
            unique_classes = set(int(x) for x in cache["unique_classes"])
            bad_classes = sorted(unique_classes.intersection(remove_class_ids))
            if bad_classes:
                delete_reason = "contains_removed_class_ids:" + ",".join(map(str, bad_classes))

        if delete_reason:
            safe_delete(image_path, delete_reason, delete_rows, dry_run=dry_run)
            safe_delete(label_path, delete_reason, delete_rows, dry_run=dry_run)
            safe_delete(visual_path, delete_reason + "_old_visual", delete_rows, dry_run=dry_run)
        else:
            ok = draw_gt_visual(
                image_path=image_path,
                valid_boxes=cache["valid_boxes"],
                classes=classes,
                out_path=visual_path,
            )

            kept_rows.append({
                "image": str(image_path),
                "label": str(label_path),
                "visual": str(visual_path),
                "unique_classes": ",".join(map(str, cache["unique_classes"])),
                "num_boxes": len(cache["valid_boxes"]),
                "filter_mode": filter_mode,
            })

            visual_rows.append({
                "image": str(image_path),
                "visual": str(visual_path),
                "status": "visual_saved" if ok else "visual_failed",
            })

        if idx == 1 or idx % log_every == 0 or idx == len(image_paths):
            info(
                f"Cleanup/visuals: {idx}/{len(image_paths)} | "
                f"kept={len(kept_rows)} | delete_records={len(delete_rows)} | visuals={len(visual_rows)}"
            )

    # Delete orphan labels whose image no longer exists or never existed.
    orphan_label_rows = []
    label_files = collect_labels(labels_dir)

    image_stems = {
        p.stem for p in collect_images(images_dir)
    }

    for label_path in label_files:
        if label_path.stem not in image_stems:
            reason = "orphan_label_without_image"
            safe_delete(label_path, reason, orphan_label_rows, dry_run=dry_run)

    # Write reports.
    write_csv(out_dir / "class_statistics.csv", class_rows)
    write_csv(out_dir / "image_box_statistics.csv", image_rows)
    write_csv(out_dir / "box_statistics.csv", box_rows)
    write_csv(out_dir / "fragmented_classes.csv", fragmented_class_rows)
    write_csv(out_dir / "low_image_classes.csv", low_image_class_rows)
    write_csv(out_dir / "high_box_images.csv", high_box_image_rows)
    write_csv(out_dir / "small_box_heavy_images.csv", small_box_heavy_image_rows)
    write_csv(out_dir / "tiny_box_heavy_images.csv", tiny_box_heavy_image_rows)
    write_csv(out_dir / "missing_labels.csv", missing_label_rows)
    write_csv(out_dir / "empty_labels.csv", empty_label_rows)
    write_csv(out_dir / "invalid_label_rows.csv", invalid_rows)
    write_csv(out_dir / "delete_report.csv", delete_rows)
    write_csv(out_dir / "kept_images.csv", kept_rows)
    write_csv(out_dir / "visual_report.csv", visual_rows)
    write_csv(out_dir / "orphan_label_delete_report.csv", orphan_label_rows)

    removed_class_rows = [
        r for r in class_rows
        if int(r["class_id"]) in remove_class_ids
    ]
    kept_class_rows = [
        r for r in class_rows
        if int(r["class_id"]) not in remove_class_ids
    ]

    write_csv(out_dir / "removed_classes.csv", removed_class_rows)
    write_csv(out_dir / "kept_classes.csv", kept_class_rows)

    summary = {
        "dataset_root": str(dataset_root),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "visuals_dir": str(visuals_dir),
        "out_dir": str(out_dir),
        "filter_mode": filter_mode,
        "dry_run": dry_run,
        "num_images_found_before_cleanup": len(image_paths),
        "num_valid_images_with_labels_before_cleanup": len(image_rows),
        "num_boxes_before_cleanup": len(box_rows),
        "num_classes_before_cleanup": len(class_rows),
        "num_fragmented_classes": len(fragmented_class_rows),
        "num_low_image_classes": len(low_image_class_rows),
        "num_removed_classes_under_mode": len(remove_class_ids),
        "num_high_box_images": len(high_box_image_rows),
        "num_small_box_heavy_images": len(small_box_heavy_image_rows),
        "num_tiny_box_heavy_images": len(tiny_box_heavy_image_rows),
        "num_missing_labels": len(missing_label_rows),
        "num_empty_labels": len(empty_label_rows),
        "num_invalid_label_rows": len(invalid_rows),
        "num_kept_images": len(kept_rows),
        "num_delete_records": len(delete_rows),
        "num_orphan_label_delete_records": len(orphan_label_rows),
        "thresholds": {
            "small_area_thr": small_area_thr,
            "tiny_area_thr": tiny_area_thr,
            "high_boxes_per_image_thr": high_boxes_per_image_thr,
            "high_boxes_per_class_thr": high_boxes_per_class_thr,
            "low_image_class_thr": low_image_class_thr,
        },
        "outputs": {
            "class_statistics": str(out_dir / "class_statistics.csv"),
            "image_box_statistics": str(out_dir / "image_box_statistics.csv"),
            "box_statistics": str(out_dir / "box_statistics.csv"),
            "fragmented_classes": str(out_dir / "fragmented_classes.csv"),
            "low_image_classes": str(out_dir / "low_image_classes.csv"),
            "removed_classes": str(out_dir / "removed_classes.csv"),
            "kept_classes": str(out_dir / "kept_classes.csv"),
            "delete_report": str(out_dir / "delete_report.csv"),
            "kept_images": str(out_dir / "kept_images.csv"),
            "visual_report": str(out_dir / "visual_report.csv"),
        },
    }

    (out_dir / "cleanup_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    info("Cleanup and visualization finished.")
    info(f"class_statistics: {out_dir / 'class_statistics.csv'}")
    info(f"low_image_classes: {out_dir / 'low_image_classes.csv'}")
    info(f"fragmented_classes: {out_dir / 'fragmented_classes.csv'}")
    info(f"removed_classes: {out_dir / 'removed_classes.csv'}")
    info(f"delete_report: {out_dir / 'delete_report.csv'}")
    info(f"kept_images: {out_dir / 'kept_images.csv'}")
    info(f"visuals_dir: {visuals_dir}")
    info(f"summary: {out_dir / 'cleanup_summary.json'}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        default="./data/PlantDoc/",
        help="Flat YOLO PlantDoc root containing images/ and labels/.",
    )
    parser.add_argument(
        "--out-dir",
        default="./data/PlantDoc/analysis_cleanup",
        help="Output analysis and cleanup report directory.",
    )

    parser.add_argument(
        "--filter-mode",
        choices=["single", "multiclass"],
        default="multiclass",
        help=(
            "single: delete only missing/empty/invalid label images. "
            "multiclass: additionally delete classes with low image count or fragmented boxes."
        ),
    )

    parser.add_argument("--small-area-thr", type=float, default=0.005)
    parser.add_argument("--tiny-area-thr", type=float, default=0.001)
    parser.add_argument("--high-boxes-per-image-thr", type=int, default=20)
    parser.add_argument("--high-boxes-per-class-thr", type=float, default=20.0)
    parser.add_argument("--low-image-class-thr", type=int, default=30)
    parser.add_argument("--log-every", type=int, default=200)

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be deleted, without deleting files.",
    )

    args = parser.parse_args()

    analyze_and_clean_flat_yolo(
        dataset_root=Path(args.dataset_root).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        filter_mode=args.filter_mode,
        small_area_thr=args.small_area_thr,
        tiny_area_thr=args.tiny_area_thr,
        high_boxes_per_image_thr=args.high_boxes_per_image_thr,
        high_boxes_per_class_thr=args.high_boxes_per_class_thr,
        low_image_class_thr=args.low_image_class_thr,
        log_every=args.log_every,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()