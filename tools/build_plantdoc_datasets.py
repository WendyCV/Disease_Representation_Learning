import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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

            info(f"Loaded classes.txt: {p} ({len(classes)} classes)")
            return classes

    info("No classes.txt found. Class names will be class_<id>.")
    return {}


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
            float(parts[1])
            float(parts[2])
            float(parts[3])
            float(parts[4])
        except Exception:
            continue

        rows.append(parts[:5])

    return rows


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


def parse_ratio(ratio_text: str) -> Tuple[float, float, float]:
    parts = [float(x.strip()) for x in ratio_text.split(",")]

    if len(parts) != 3:
        raise ValueError("--split-ratio must be like 0.7,0.2,0.1")

    s = sum(parts)

    if s <= 0:
        raise ValueError("Split ratio sum must be > 0")

    return parts[0] / s, parts[1] / s, parts[2] / s


def split_records(records: List[dict], ratio: Tuple[float, float, float], seed: int):
    rng = random.Random(seed)
    records = list(records)
    rng.shuffle(records)

    n = len(records)
    train_r, val_r, _ = ratio

    n_train = int(round(n * train_r))
    n_val = int(round(n * val_r))

    if n_train + n_val > n:
        n_val = max(0, n - n_train)

    train = records[:n_train]
    val = records[n_train:n_train + n_val]
    test = records[n_train + n_val:]

    return {
        "train": train,
        "val": val,
        "test": test,
    }


def find_sidecar_by_stem(root: Path, stem: str, preferred_ext: Optional[str] = None) -> Optional[Path]:
    if not root.exists():
        return None

    if preferred_ext:
        p = root / f"{stem}{preferred_ext}"
        if p.exists():
            return p

    for p in root.iterdir():
        if p.is_file() and normalize_stem(p.name) == stem.lower():
            return p

    return None


def resolve_yaml_path(yaml_arg: str, fallback_parent: Path) -> Path:
    p = Path(yaml_arg)

    if p.parent == Path(".") and not p.is_absolute():
        return (fallback_parent / p).resolve()

    return p.resolve()


def build_records(dataset_root: Path):
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    visuals_dir = dataset_root / "visuals"

    image_paths = collect_images(images_dir)

    records = []
    missing_rows = []
    class_image_sets = {}
    class_box_counts = {}

    info(f"Scanning cleaned PlantDoc root: {dataset_root}")
    info(f"images_dir: {images_dir}")
    info(f"labels_dir: {labels_dir}")
    info(f"visuals_dir: {visuals_dir}")
    info(f"Found images: {len(image_paths)}")

    for idx, image_path in enumerate(image_paths, start=1):
        stem = image_path.stem
        label_path = labels_dir / f"{stem}.txt"

        if not label_path.exists():
            missing_rows.append({
                "image": str(image_path),
                "label": str(label_path),
                "reason": "label_not_found",
            })
            continue

        label_rows = read_yolo_label(label_path)

        if not label_rows:
            missing_rows.append({
                "image": str(image_path),
                "label": str(label_path),
                "reason": "empty_or_invalid_label",
            })
            continue

        class_ids = [int(float(row[0])) for row in label_rows]
        unique_classes = sorted(set(class_ids))

        visual_path = find_sidecar_by_stem(visuals_dir, stem, ".jpg")

        rec = {
            "image_path": image_path,
            "label_path": label_path,
            "visual_path": visual_path,
            "image_name": image_path.name,
            "stem": stem,
            "label_rows": label_rows,
            "class_ids": class_ids,
            "unique_classes": unique_classes,
            "num_boxes": len(label_rows),
        }

        records.append(rec)

        for cid in unique_classes:
            class_image_sets.setdefault(cid, set()).add(image_path.name)

        for cid in class_ids:
            class_box_counts[cid] = class_box_counts.get(cid, 0) + 1

        if idx == 1 or idx % 100 == 0 or idx == len(image_paths):
            info(
                f"Scan progress: {idx}/{len(image_paths)} | valid={len(records)} | missing/empty={len(missing_rows)}"
            )

    class_stats = []

    for cid in sorted(set(list(class_image_sets.keys()) + list(class_box_counts.keys()))):
        image_count = len(class_image_sets.get(cid, set()))
        box_count = class_box_counts.get(cid, 0)
        class_stats.append({
            "old_class_id": cid,
            "image_count": image_count,
            "box_count": box_count,
            "boxes_per_image": box_count / image_count if image_count else 0,
        })

    info(f"Valid images with labels: {len(records)}")
    info(f"Classes found in remaining labels: {len(class_stats)}")

    return records, class_stats, missing_rows


def build_multiclass_mapping(records: List[dict], old_classes: Dict[int, str]):
    used_old_ids = sorted({
        int(cid)
        for rec in records
        for cid in rec["unique_classes"]
    })

    old_to_new = {}
    new_to_name = {}
    mapping_rows = []

    for new_id, old_id in enumerate(used_old_ids):
        old_to_new[old_id] = new_id
        class_name = old_classes.get(old_id, f"class_{old_id}")
        new_to_name[new_id] = class_name

        mapping_rows.append({
            "old_class_id": old_id,
            "new_class_id": new_id,
            "class_name": class_name,
        })

    return old_to_new, new_to_name, mapping_rows


def make_single_label_lines(label_rows: List[List[str]]) -> List[str]:
    out = []

    for parts in label_rows:
        new_parts = list(parts)
        new_parts[0] = "0"
        out.append(" ".join(new_parts[:5]))

    return out


def make_multiclass_label_lines(label_rows: List[List[str]], old_to_new: Dict[int, int]) -> List[str]:
    out = []

    for parts in label_rows:
        old_cls = int(float(parts[0]))

        if old_cls not in old_to_new:
            continue

        new_parts = list(parts)
        new_parts[0] = str(old_to_new[old_cls])
        out.append(" ".join(new_parts[:5]))

    return out


def write_classes_txt(labels_root: Path, new_to_name: Dict[int, str]):
    ensure_dir(labels_root)

    lines = []

    for cid in sorted(new_to_name.keys()):
        lines.append(new_to_name[cid])

    out_path = labels_root / "classes.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    info(f"Wrote classes.txt: {out_path} ({len(lines)} classes)")


def write_yaml(yaml_path: Path, dataset_root: Path, new_to_name: Dict[int, str]):
    ensure_dir(yaml_path.parent)

    lines = []
    lines.append(f"path: {dataset_root.as_posix()}")
    lines.append("train: images/train")
    lines.append("val: images/val")
    lines.append("test: images/test")
    lines.append("")
    lines.append("names:")

    for cid in sorted(new_to_name.keys()):
        lines.append(f"  {cid}: {new_to_name[cid]}")

    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    info(f"Wrote data yaml: {yaml_path}")


def copy_dataset(
    split_records_map: Dict[str, List[dict]],
    out_root: Path,
    yaml_path: Path,
    mode: str,
    new_to_name: Dict[int, str],
    old_to_new: Optional[Dict[int, int]],
    copy_visuals: bool,
):
    summary_rows = []
    skipped_rows = []

    info(f"Writing {mode} dataset to: {out_root}")
    info(f"Writing {mode} yaml to: {yaml_path}")

    for split, records in split_records_map.items():
        out_images_dir = out_root / "images" / split
        out_labels_dir = out_root / "labels" / split
        out_visuals_dir = out_root / "visuals" / split

        ensure_dir(out_images_dir)
        ensure_dir(out_labels_dir)

        if copy_visuals:
            ensure_dir(out_visuals_dir)

        info(f"Writing {mode}/{split}: {len(records)} images")

        written = 0
        skipped = 0

        for idx, rec in enumerate(records, start=1):
            image_path: Path = rec["image_path"]
            visual_path = rec["visual_path"]

            if mode == "singleclass":
                label_lines = make_single_label_lines(rec["label_rows"])
            elif mode == "multiclass":
                if old_to_new is None:
                    raise ValueError("old_to_new is required for multiclass mode")
                label_lines = make_multiclass_label_lines(rec["label_rows"], old_to_new)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            if not label_lines:
                skipped += 1
                skipped_rows.append({
                    "split": split,
                    "mode": mode,
                    "image_name": rec["image_name"],
                    "reason": "no_label_after_mapping",
                    "old_classes": ",".join(map(str, rec["unique_classes"])),
                })
                continue

            out_image_path = out_images_dir / image_path.name
            out_label_path = out_labels_dir / f"{rec['stem']}.txt"

            shutil.copy2(image_path, out_image_path)
            out_label_path.write_text("\n".join(label_lines), encoding="utf-8")

            copied_visual = False
            if copy_visuals and visual_path is not None and visual_path.exists():
                out_visual_path = out_visuals_dir / visual_path.name
                shutil.copy2(visual_path, out_visual_path)
                copied_visual = True

            if mode == "singleclass":
                new_classes = "0"
            else:
                new_classes = ",".join(
                    str(old_to_new[c]) for c in rec["unique_classes"] if c in old_to_new
                )

            summary_rows.append({
                "split": split,
                "mode": mode,
                "image_name": rec["image_name"],
                "old_classes": ",".join(map(str, rec["unique_classes"])),
                "new_classes": new_classes,
                "num_boxes": len(label_lines),
                "copied_visual": int(copied_visual),
            })

            written += 1

            if idx == 1 or idx % 100 == 0 or idx == len(records):
                info(
                    f"Writing {mode}/{split}: {idx}/{len(records)} | written={written} | skipped={skipped}"
                )

    write_classes_txt(out_root / "labels", new_to_name)
    write_yaml(yaml_path, out_root, new_to_name)

    write_csv(out_root / "split_records.csv", summary_rows)
    write_csv(out_root / "skipped_records.csv", skipped_rows)

    split_counts = {}
    total_boxes = 0

    for row in summary_rows:
        split = row["split"]
        split_counts.setdefault(split, {"images": 0, "boxes": 0})
        split_counts[split]["images"] += 1
        split_counts[split]["boxes"] += int(row["num_boxes"])
        total_boxes += int(row["num_boxes"])

    summary = {
        "out_root": str(out_root),
        "yaml_path": str(yaml_path),
        "mode": mode,
        "num_classes": len(new_to_name),
        "total_images": len(summary_rows),
        "total_boxes": total_boxes,
        "split_counts": split_counts,
        "classes": new_to_name,
    }

    (out_root / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    info(f"Wrote {mode} split summary: {out_root / 'split_summary.json'}")

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        default="./data/PlantDoc/",
        help="Cleaned flat YOLO PlantDoc root containing images/, labels/, and optional visuals/.",
    )

    parser.add_argument(
        "--out-single-root",
        default="./data/PlantDoc_singleclass",
        help="Output single-class split dataset root.",
    )
    parser.add_argument(
        "--out-multiclass-root",
        default="./data/PlantDoc_multiclass",
        help="Output multi-class split dataset root.",
    )

    parser.add_argument(
        "--single-yaml",
        default="./data/plantdoc_singleclass.yaml",
        help="Output yaml path for single-class dataset.",
    )
    parser.add_argument(
        "--multiclass-yaml",
        default="./data/plantdoc_multiclass.yaml",
        help="Output yaml path for multi-class dataset.",
    )

    parser.add_argument(
        "--split-ratio",
        default="0.7,0.2,0.1",
        help="train,val,test ratio.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-copy-visuals", action="store_true")

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    out_single_root = Path(args.out_single_root).resolve()
    out_multiclass_root = Path(args.out_multiclass_root).resolve()
    single_yaml_path = Path(args.single_yaml).resolve()
    multiclass_yaml_path = Path(args.multiclass_yaml).resolve()

    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    info("========== PlantDoc singleclass + multiclass split started ==========")
    info(f"dataset_root: {dataset_root}")
    info(f"images_dir: {images_dir}")
    info(f"labels_dir: {labels_dir}")
    info(f"out_single_root: {out_single_root}")
    info(f"out_multiclass_root: {out_multiclass_root}")
    info(f"single_yaml_path: {single_yaml_path}")
    info(f"multiclass_yaml_path: {multiclass_yaml_path}")
    info(f"split_ratio: {args.split_ratio}")
    info(f"seed: {args.seed}")
    info(f"copy_visuals: {not args.no_copy_visuals}")

    old_classes = read_classes_txt(labels_dir)

    records, class_stats, missing_rows = build_records(dataset_root)

    old_to_new, multiclass_id_to_name, mapping_rows = build_multiclass_mapping(
        records=records,
        old_classes=old_classes,
    )

    single_id_to_name = {
        0: "leaf_region"
    }

    info(f"Valid source images: {len(records)}")
    info(f"Old classes used: {len(old_to_new)}")
    info(f"Multiclass continuous classes: {len(multiclass_id_to_name)}")
    info("Singleclass class: 0 leaf_region")

    ratio = parse_ratio(args.split_ratio)

    # Important:
    # Use the same image split for both singleclass and multiclass
    # so that the two settings are comparable.
    split_records_map = split_records(records, ratio, args.seed)

    for split, recs in split_records_map.items():
        info(f"{split}: {len(recs)} images")

    single_summary = copy_dataset(
        split_records_map=split_records_map,
        out_root=out_single_root,
        yaml_path=single_yaml_path,
        mode="singleclass",
        new_to_name=single_id_to_name,
        old_to_new=None,
        copy_visuals=not args.no_copy_visuals,
    )

    multiclass_summary = copy_dataset(
        split_records_map=split_records_map,
        out_root=out_multiclass_root,
        yaml_path=multiclass_yaml_path,
        mode="multiclass",
        new_to_name=multiclass_id_to_name,
        old_to_new=old_to_new,
        copy_visuals=not args.no_copy_visuals,
    )

    # Common reports
    write_csv(out_single_root / "source_class_statistics.csv", class_stats)
    write_csv(out_single_root / "missing_or_empty_records.csv", missing_rows)

    write_csv(out_multiclass_root / "source_class_statistics.csv", class_stats)
    write_csv(out_multiclass_root / "missing_or_empty_records.csv", missing_rows)
    write_csv(out_multiclass_root / "class_mapping_old_to_new.csv", mapping_rows)

    final_summary = {
        "dataset_root": str(dataset_root),
        "out_single_root": str(out_single_root),
        "out_multiclass_root": str(out_multiclass_root),
        "single_yaml_path": str(single_yaml_path),
        "multiclass_yaml_path": str(multiclass_yaml_path),
        "split_ratio": {
            "train": ratio[0],
            "val": ratio[1],
            "test": ratio[2],
        },
        "seed": args.seed,
        "source_valid_images": len(records),
        "source_missing_or_empty": len(missing_rows),
        "old_classes_used": len(old_to_new),
        "singleclass": single_summary,
        "multiclass": multiclass_summary,
        "note": (
            "This script builds two split datasets from the cleaned flat PlantDoc dataset. "
            "Singleclass maps all boxes to class 0 leaf_region. "
            "Multiclass remaps remaining old class ids to continuous new class ids. "
            "Both datasets use the same train/val/test image split for fair comparison."
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

    info("========== PlantDoc singleclass + multiclass split finished ==========")

    print("\n[DONE]")
    print("single out_root:", out_single_root)
    print("single yaml:", single_yaml_path)
    print("multiclass out_root:", out_multiclass_root)
    print("multiclass yaml:", multiclass_yaml_path)
    print("valid source images:", len(records))
    print("classes multiclass:", len(multiclass_id_to_name))
    print("single train images:", single_summary["split_counts"].get("train", {}).get("images", 0))
    print("single val images:", single_summary["split_counts"].get("val", {}).get("images", 0))
    print("single test images:", single_summary["split_counts"].get("test", {}).get("images", 0))
    print("multi train images:", multiclass_summary["split_counts"].get("train", {}).get("images", 0))
    print("multi val images:", multiclass_summary["split_counts"].get("val", {}).get("images", 0))
    print("multi test images:", multiclass_summary["split_counts"].get("test", {}).get("images", 0))


if __name__ == "__main__":
    main()