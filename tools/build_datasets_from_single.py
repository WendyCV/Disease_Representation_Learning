from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# =========================================================
# Basic IO
# =========================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: List[dict]) -> None:
    ensure_dir(path.parent)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


# =========================================================
# YAML helpers
# =========================================================
def resolve_yaml_path(data_yaml: Path, value: str) -> Path:
    data = load_yaml(data_yaml)
    yaml_dir = data_yaml.parent

    dataset_root = data.get("path", None)
    if dataset_root is None:
        base = yaml_dir
    else:
        dataset_root = Path(str(dataset_root))
        if dataset_root.is_absolute():
            base = dataset_root
        else:
            base = (yaml_dir / dataset_root).resolve()

    p = Path(value)
    if p.is_absolute():
        return p.resolve()

    return (base / p).resolve()


def get_dataset_root(data_yaml: Path) -> Path:
    data = load_yaml(data_yaml)
    yaml_dir = data_yaml.parent

    dataset_root = data.get("path", None)
    if dataset_root is None:
        return yaml_dir.resolve()

    dataset_root = Path(str(dataset_root))
    if dataset_root.is_absolute():
        return dataset_root.resolve()

    return (yaml_dir / dataset_root).resolve()


def get_split_image_dir(data_yaml: Path, split: str) -> Optional[Path]:
    data = load_yaml(data_yaml)
    if split not in data:
        return None
    return resolve_yaml_path(data_yaml, str(data[split]))


def infer_single_label_path(
    single_data_yaml: Path,
    split: str,
    image_path: Path,
    split_image_dir: Path,
) -> Path:
    """
    Example:
      PlantSeg_singleclass/images/train/a.jpg
      PlantSeg_singleclass/labels/train/a.txt
    """
    dataset_root = get_dataset_root(single_data_yaml)
    rel_img = image_path.relative_to(split_image_dir)
    return (dataset_root / "labels" / split / rel_img).with_suffix(".txt")


def iter_images(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        return []

    return [
        p for p in sorted(image_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    ]


# =========================================================
# Label helpers
# =========================================================
def read_label_rows(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Read YOLO label.

    Returns:
        [(cls_id, x, y, w, h), ...]
    """
    rows: List[Tuple[int, float, float, float, float]] = []

    if not label_path.exists():
        return rows

    for line_idx, line in enumerate(
        label_path.read_text(encoding="utf-8", errors="ignore").splitlines(),
        start=1,
    ):
        raw = line.strip()
        if not raw:
            continue

        parts = raw.split()
        if len(parts) < 5:
            print(f"[WARN] {label_path}: line {line_idx} has fewer than 5 columns, skipped")
            continue

        try:
            cls_id = int(float(parts[0]))
            x, y, w, h = [float(v) for v in parts[1:5]]
        except Exception:
            print(f"[WARN] {label_path}: line {line_idx} parse failed, skipped")
            continue

        rows.append((cls_id, x, y, w, h))

    return rows


def bbox_close(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    tol: float,
) -> bool:
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def bbox_distance(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    return sum(abs(float(x) - float(y)) for x, y in zip(a, b))


def format_yolo_line(cls_id: int, x: float, y: float, w: float, h: float) -> str:
    return f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


# =========================================================
# Source PlantSeg helpers
# =========================================================
def build_source_label_index(source_label_root: Path) -> Dict[str, Path]:
    """
    Build original PlantSeg label index:
        image_stem -> original multiclass label path
    """
    index: Dict[str, Path] = {}

    for p in sorted(source_label_root.rglob("*.txt")):
        if p.name.lower() == "classes.txt":
            continue

        stem = p.stem
        if stem in index:
            print(f"[WARN] duplicate source label stem, keep first: {stem}")
            print(f"       keep: {index[stem]}")
            print(f"       skip: {p}")
            continue

        index[stem] = p

    return index


def read_source_classes_txt(path: Path) -> Dict[int, str]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    out: Dict[int, str] = {}
    for i, line in enumerate(lines):
        name = line.strip()
        if name:
            out[i] = name

    return out


# =========================================================
# Build-summary class mapping
# =========================================================
def load_final_classes_from_build_summary(build_summary_path: Path):
    """
    Read final multiclass category space from build_summary.json.

    Returns:
        final_id_to_name: {0: name0, ..., nc-1: name}
        final_name_to_id: {name: new_id}
    """
    summary = load_json(build_summary_path)

    if "multiclass" not in summary or "classes" not in summary["multiclass"]:
        raise KeyError("build_summary.json must contain ['multiclass']['classes'].")

    classes_raw = summary["multiclass"]["classes"]

    final_id_to_name = {int(k): str(v) for k, v in classes_raw.items()}
    final_id_to_name = dict(sorted(final_id_to_name.items(), key=lambda x: x[0]))

    expected_ids = list(range(len(final_id_to_name)))
    actual_ids = sorted(final_id_to_name.keys())
    if actual_ids != expected_ids:
        raise ValueError(
            f"Final class ids in build_summary are not contiguous. "
            f"expected={expected_ids[:5]}...{expected_ids[-5:] if expected_ids else []}, "
            f"actual={actual_ids[:5]}...{actual_ids[-5:] if actual_ids else []}"
        )

    final_name_to_id = {name: cid for cid, name in final_id_to_name.items()}

    return summary, final_id_to_name, final_name_to_id


def convert_old_cls_to_final_cls(
    old_cls: int,
    source_id_to_name: Dict[int, str],
    final_name_to_id: Dict[str, int],
) -> Tuple[Optional[int], Optional[str]]:
    """
    old class id -> source class name -> final new class id from build_summary.
    """
    old_name = source_id_to_name.get(old_cls, None)
    if old_name is None:
        return None, None

    final_id = final_name_to_id.get(old_name, None)
    if final_id is None:
        return None, old_name

    return int(final_id), old_name


# =========================================================
# Matching single bboxes to source multiclass labels
# =========================================================
def match_single_boxes_to_source_rows(
    single_rows: List[Tuple[int, float, float, float, float]],
    source_rows: List[Tuple[int, float, float, float, float]],
    bbox_tol: float,
) -> Tuple[List[dict], List[dict]]:
    """
    single_rows:
      class id is 0, bbox defines which boxes are retained.

    source_rows:
      old class id + source bbox.

    Return:
      matched boxes with old class id recovered from source_rows,
      but bbox coordinates from single_rows.
    """
    matched: List[dict] = []
    unmatched: List[dict] = []

    used_source_indices = set()

    for s_idx, (_, sx, sy, sw, sh) in enumerate(single_rows):
        s_box = (sx, sy, sw, sh)

        best_idx = None
        best_dist = None

        for m_idx, (old_cls, mx, my, mw, mh) in enumerate(source_rows):
            if m_idx in used_source_indices:
                continue

            m_box = (mx, my, mw, mh)

            if bbox_close(s_box, m_box, bbox_tol):
                dist = bbox_distance(s_box, m_box)
                if best_idx is None or dist < best_dist:
                    best_idx = m_idx
                    best_dist = dist

        if best_idx is None:
            unmatched.append({
                "single_box_index": s_idx,
                "single_box": f"{sx:.6f},{sy:.6f},{sw:.6f},{sh:.6f}",
                "reason": "single_box_not_matched_in_source_label",
            })
            continue

        used_source_indices.add(best_idx)
        old_cls, _, _, _, _ = source_rows[best_idx]

        matched.append({
            "single_box_index": s_idx,
            "source_box_index": best_idx,
            "old_cls": int(old_cls),
            "x": sx,
            "y": sy,
            "w": sw,
            "h": sh,
            "bbox_dist": float(best_dist),
        })

    return matched, unmatched


# =========================================================
# Core collection and writing
# =========================================================
def collect_records_from_single(
    single_data_yaml: Path,
    source_label_root: Path,
    source_id_to_name: Dict[int, str],
    final_name_to_id: Dict[str, int],
    splits: List[str],
    bbox_tol: float,
):
    source_label_index = build_source_label_index(source_label_root)

    candidate_records: List[dict] = []
    missing_rows: List[dict] = []
    unmatched_rows: List[dict] = []
    skipped_box_rows: List[dict] = []

    total_single_images = 0
    total_single_boxes = 0
    total_matched_boxes = 0
    total_final_boxes = 0

    for split in splits:
        split_img_dir = get_split_image_dir(single_data_yaml, split)

        if split_img_dir is None:
            print(f"[WARN] split not found in single data yaml: {split}")
            continue

        if not split_img_dir.exists():
            print(f"[WARN] split image dir not found: {split_img_dir}")
            continue

        images = iter_images(split_img_dir)

        print(f"[INFO] Scan split: {split}")
        print(f"       single image dir: {split_img_dir}")
        print(f"       images found    : {len(images)}")

        split_kept_images = 0
        split_skipped_images = 0
        split_boxes = 0
        split_final_boxes = 0

        for img_path in images:
            total_single_images += 1

            single_label_path = infer_single_label_path(
                single_data_yaml=single_data_yaml,
                split=split,
                image_path=img_path,
                split_image_dir=split_img_dir,
            )

            single_rows = read_label_rows(single_label_path)

            if not single_rows:
                missing_rows.append({
                    "split": split,
                    "image_name": img_path.name,
                    "image_path": str(img_path),
                    "single_label_path": str(single_label_path),
                    "reason": "missing_or_empty_single_label",
                })
                split_skipped_images += 1
                continue

            total_single_boxes += len(single_rows)
            split_boxes += len(single_rows)

            source_label_path = source_label_index.get(img_path.stem, None)

            if source_label_path is None:
                missing_rows.append({
                    "split": split,
                    "image_name": img_path.name,
                    "image_path": str(img_path),
                    "single_label_path": str(single_label_path),
                    "reason": "missing_source_multiclass_label",
                })
                split_skipped_images += 1
                continue

            source_rows = read_label_rows(source_label_path)

            if not source_rows:
                missing_rows.append({
                    "split": split,
                    "image_name": img_path.name,
                    "image_path": str(img_path),
                    "single_label_path": str(single_label_path),
                    "source_label_path": str(source_label_path),
                    "reason": "empty_or_invalid_source_label",
                })
                split_skipped_images += 1
                continue

            matched, unmatched = match_single_boxes_to_source_rows(
                single_rows=single_rows,
                source_rows=source_rows,
                bbox_tol=bbox_tol,
            )

            total_matched_boxes += len(matched)

            for u in unmatched:
                unmatched_rows.append({
                    "split": split,
                    "image_name": img_path.name,
                    "image_path": str(img_path),
                    "single_label_path": str(single_label_path),
                    "source_label_path": str(source_label_path),
                    **u,
                })

            final_rows = []
            skipped_boxes_for_this_image = 0

            for m in matched:
                old_cls = int(m["old_cls"])

                final_cls, old_name = convert_old_cls_to_final_cls(
                    old_cls=old_cls,
                    source_id_to_name=source_id_to_name,
                    final_name_to_id=final_name_to_id,
                )

                if final_cls is None:
                    skipped_boxes_for_this_image += 1
                    skipped_box_rows.append({
                        "split": split,
                        "image_name": img_path.name,
                        "image_path": str(img_path),
                        "single_label_path": str(single_label_path),
                        "source_label_path": str(source_label_path),
                        "old_class_id": old_cls,
                        "old_class_name": old_name if old_name else "",
                        "reason": "class_not_in_build_summary_multiclass_classes",
                        "x": f"{m['x']:.6f}",
                        "y": f"{m['y']:.6f}",
                        "w": f"{m['w']:.6f}",
                        "h": f"{m['h']:.6f}",
                    })
                    continue

                final_rows.append({
                    "final_cls": final_cls,
                    "old_cls": old_cls,
                    "old_name": old_name,
                    "x": float(m["x"]),
                    "y": float(m["y"]),
                    "w": float(m["w"]),
                    "h": float(m["h"]),
                    "single_box_index": m["single_box_index"],
                    "source_box_index": m["source_box_index"],
                    "bbox_dist": m["bbox_dist"],
                })

            if not final_rows:
                missing_rows.append({
                    "split": split,
                    "image_name": img_path.name,
                    "image_path": str(img_path),
                    "single_label_path": str(single_label_path),
                    "source_label_path": str(source_label_path),
                    "reason": "all_boxes_removed_by_summary_class_filter",
                    "num_single_boxes": len(single_rows),
                    "num_matched_boxes": len(matched),
                    "num_skipped_boxes": skipped_boxes_for_this_image,
                })
                split_skipped_images += 1
                continue

            total_final_boxes += len(final_rows)
            split_final_boxes += len(final_rows)
            split_kept_images += 1

            rel_img = img_path.relative_to(split_img_dir)

            candidate_records.append({
                "split": split,
                "image_path": img_path,
                "rel_img": rel_img,
                "single_label_path": single_label_path,
                "source_label_path": source_label_path,
                "image_name": img_path.name,
                "image_stem": img_path.stem,
                "num_single_boxes": len(single_rows),
                "num_source_boxes": len(source_rows),
                "num_matched_boxes": len(matched),
                "num_final_boxes": len(final_rows),
                "num_unmatched_boxes": len(unmatched),
                "num_skipped_boxes_not_in_summary": skipped_boxes_for_this_image,
                "rows": final_rows,
            })

        print(f"       single boxes       : {split_boxes}")
        print(f"       kept images        : {split_kept_images}")
        print(f"       final boxes        : {split_final_boxes}")
        print(f"       skipped images     : {split_skipped_images}")
        print("")

    counters = {
        "total_single_images": total_single_images,
        "total_single_boxes": total_single_boxes,
        "total_matched_boxes": total_matched_boxes,
        "total_final_boxes": total_final_boxes,
        "total_output_images": len(candidate_records),
    }

    return candidate_records, missing_rows, unmatched_rows, skipped_box_rows, counters


def clean_output_dir(
    out_root: Path,
    source_label_root: Path,
    enabled: bool,
) -> None:
    if not enabled:
        return

    out_root = out_root.resolve()
    source_label_root = source_label_root.resolve()

    if not out_root.exists():
        return

    if out_root == source_label_root:
        raise RuntimeError(f"Refuse to clean output because out_root == source_label_root: {out_root}")

    try:
        source_label_root.relative_to(out_root)
        raise RuntimeError(
            "Refuse to clean output because source_label_root is inside out_root. "
            f"out_root={out_root}, source_label_root={source_label_root}"
        )
    except ValueError:
        pass

    print(f"[INFO] Clean output root: {out_root}")
    shutil.rmtree(out_root)


def write_dataset(
    records: List[dict],
    out_root: Path,
):
    dataset_rows = []
    total_images = 0
    total_boxes = 0
    split_counts: Dict[str, dict] = {}

    for rec in records:
        split = rec["split"]
        img_path: Path = rec["image_path"]
        rel_img: Path = rec["rel_img"]

        out_img_path = out_root / "images" / split / rel_img
        out_label_path = (out_root / "labels" / split / rel_img).with_suffix(".txt")

        lines = []
        old_classes = set()
        final_classes = set()

        for row in rec["rows"]:
            final_cls = int(row["final_cls"])
            old_cls = int(row["old_cls"])

            old_classes.add(old_cls)
            final_classes.add(final_cls)

            lines.append(
                format_yolo_line(
                    final_cls,
                    float(row["x"]),
                    float(row["y"]),
                    float(row["w"]),
                    float(row["h"]),
                )
            )

        if not lines:
            continue

        copy_file(img_path, out_img_path)
        ensure_dir(out_label_path.parent)
        out_label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        total_images += 1
        total_boxes += len(lines)

        split_counts.setdefault(split, {"images": 0, "boxes": 0})
        split_counts[split]["images"] += 1
        split_counts[split]["boxes"] += len(lines)

        dataset_rows.append({
            "split": split,
            "image_name": rec["image_name"],
            "image_path": str(img_path),
            "out_image_path": str(out_img_path),
            "out_label_path": str(out_label_path),
            "single_label_path": str(rec["single_label_path"]),
            "source_label_path": str(rec["source_label_path"]),
            "num_single_boxes": rec["num_single_boxes"],
            "num_source_boxes": rec["num_source_boxes"],
            "num_matched_boxes": rec["num_matched_boxes"],
            "num_final_boxes": len(lines),
            "num_unmatched_boxes": rec["num_unmatched_boxes"],
            "num_skipped_boxes_not_in_summary": rec["num_skipped_boxes_not_in_summary"],
            "old_class_ids": ",".join(map(str, sorted(old_classes))),
            "final_class_ids": ",".join(map(str, sorted(final_classes))),
        })

    return {
        "total_images": total_images,
        "total_boxes": total_boxes,
        "split_counts": split_counts,
        "dataset_rows": dataset_rows,
    }


def write_classes_txt(labels_root: Path, id_to_name: Dict[int, str]) -> None:
    ensure_dir(labels_root)
    lines = [id_to_name[i] for i in sorted(id_to_name.keys())]
    (labels_root / "classes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_data_yaml(
    yaml_path: Path,
    out_root: Path,
    id_to_name: Dict[int, str],
) -> None:
    data = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(id_to_name),
        "names": [id_to_name[i] for i in sorted(id_to_name.keys())],
    }
    save_yaml(yaml_path, data)


def audit_output_labels(out_root: Path, nc: int):
    labels_root = out_root / "labels"

    total_label_files = 0
    total_boxes = 0
    max_cls = -1
    bad_rows = []

    for p in sorted(labels_root.rglob("*.txt")):
        if p.name.lower() == "classes.txt":
            continue

        total_label_files += 1
        rows = read_label_rows(p)

        for cls_id, x, y, w, h in rows:
            cid = int(cls_id)
            total_boxes += 1
            max_cls = max(max_cls, cid)

            if cid < 0 or cid >= nc:
                bad_rows.append({
                    "label_path": str(p),
                    "class_id": cid,
                    "reason": "class_id_out_of_range",
                })

            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                bad_rows.append({
                    "label_path": str(p),
                    "class_id": cid,
                    "box": f"{x},{y},{w},{h}",
                    "reason": "bbox_out_of_range",
                })

    return total_label_files, total_boxes, max_cls, bad_rows


def build_from_single_with_summary(
    single_data_yaml: Path,
    source_label_root: Path,
    source_classes_txt: Path,
    build_summary: Path,
    out_root: Path,
    out_yaml: Path,
    splits: List[str],
    bbox_tol: float,
    clean_output: bool,
) -> None:
    single_data_yaml = single_data_yaml.resolve()
    source_label_root = source_label_root.resolve()
    source_classes_txt = source_classes_txt.resolve()
    build_summary = build_summary.resolve()
    out_root = out_root.resolve()
    out_yaml = out_yaml.resolve()

    if not single_data_yaml.exists():
        raise FileNotFoundError(f"single-data-yaml not found: {single_data_yaml}")
    if not source_label_root.exists():
        raise FileNotFoundError(f"source-label-root not found: {source_label_root}")
    if not source_classes_txt.exists():
        raise FileNotFoundError(f"source-classes-txt not found: {source_classes_txt}")
    if not build_summary.exists():
        raise FileNotFoundError(f"build-summary not found: {build_summary}")

    summary, final_id_to_name, final_name_to_id = load_final_classes_from_build_summary(build_summary)
    source_id_to_name = read_source_classes_txt(source_classes_txt)

    nc = len(final_id_to_name)

    print("[INFO] Build multiclass dataset from singleclass using build_summary classes")
    print(f"       single_data_yaml  : {single_data_yaml}")
    print(f"       source_label_root : {source_label_root}")
    print(f"       source_classes_txt: {source_classes_txt}")
    print(f"       build_summary     : {build_summary}")
    print(f"       out_root          : {out_root}")
    print(f"       out_yaml          : {out_yaml}")
    print(f"       final nc          : {nc}")
    print(f"       valid class ids   : 0-{nc - 1}")
    print(f"       bbox_tol          : {bbox_tol}")
    print("")

    records, missing_rows, unmatched_rows, skipped_box_rows, counters = collect_records_from_single(
        single_data_yaml=single_data_yaml,
        source_label_root=source_label_root,
        source_id_to_name=source_id_to_name,
        final_name_to_id=final_name_to_id,
        splits=splits,
        bbox_tol=bbox_tol,
    )

    clean_output_dir(
        out_root=out_root,
        source_label_root=source_label_root,
        enabled=clean_output,
    )

    write_result = write_dataset(records, out_root)

    write_classes_txt(out_root / "labels", final_id_to_name)
    write_data_yaml(out_yaml, out_root, final_id_to_name)

    total_label_files, audit_boxes, max_cls, bad_rows = audit_output_labels(out_root, nc)

    write_csv(out_root / "dataset_records.csv", write_result["dataset_rows"])
    write_csv(out_root / "missing_or_removed_images.csv", missing_rows)
    write_csv(out_root / "unmatched_single_boxes.csv", unmatched_rows)
    write_csv(out_root / "skipped_boxes_not_in_summary_classes.csv", skipped_box_rows)
    write_csv(out_root / "audit_bad_labels.csv", bad_rows)

    expected_single = summary.get("single", {})
    expected_multi = summary.get("multiclass", {})

    report = {
        "inputs": {
            "single_data_yaml": str(single_data_yaml),
            "source_label_root": str(source_label_root),
            "source_classes_txt": str(source_classes_txt),
            "build_summary": str(build_summary),
            "out_root": str(out_root),
            "out_yaml": str(out_yaml),
            "splits": splits,
            "bbox_tol": bbox_tol,
        },
        "summary_expected": {
            "single_total_images": expected_single.get("total_images", None),
            "single_total_boxes": expected_single.get("total_boxes", None),
            "multiclass_total_images": expected_multi.get("total_images", None),
            "multiclass_total_boxes": expected_multi.get("total_boxes", None),
            "multiclass_num_classes": expected_multi.get("num_classes", None),
            "multiclass_split_counts": expected_multi.get("split_counts", None),
        },
        "actual_from_single_and_source": counters,
        "actual_written": {
            "total_images": write_result["total_images"],
            "total_boxes": write_result["total_boxes"],
            "split_counts": write_result["split_counts"],
        },
        "audit": {
            "nc": nc,
            "valid_class_range": f"0-{nc - 1}",
            "label_files": total_label_files,
            "total_boxes": audit_boxes,
            "max_class_id": max_cls,
            "bad_labels": len(bad_rows),
        },
        "checks": {
            "single_images_match_summary": counters["total_single_images"] == expected_single.get("total_images", -1),
            "single_boxes_match_summary": counters["total_single_boxes"] == expected_single.get("total_boxes", -1),
            "written_images_match_summary_multiclass": write_result["total_images"] == expected_multi.get("total_images", -1),
            "written_boxes_match_summary_multiclass": write_result["total_boxes"] == expected_multi.get("total_boxes", -1),
            "nc_match_summary_multiclass": nc == expected_multi.get("num_classes", -1),
            "bad_labels_zero": len(bad_rows) == 0,
        },
        "final_classes": final_id_to_name,
    }

    save_json(out_root / "build_from_single_report.json", report)

    print("[AUDIT]")
    print(f"       nc                    : {nc}")
    print(f"       valid class ids       : 0-{nc - 1}")
    print(f"       max class id          : {max_cls}")
    print(f"       bad labels            : {len(bad_rows)}")
    print(f"       written images        : {write_result['total_images']}")
    print(f"       written boxes         : {write_result['total_boxes']}")
    print(f"       expected multi images : {expected_multi.get('total_images', None)}")
    print(f"       expected multi boxes  : {expected_multi.get('total_boxes', None)}")
    print("")

    if bad_rows:
        print("[ERROR] Invalid labels found. See audit_bad_labels.csv")
        raise RuntimeError(f"Invalid labels found: {len(bad_rows)}")

    print("[CHECKS]")
    for k, v in report["checks"].items():
        print(f"       {k}: {v}")

    print("")
    print("[DONE]")
    print(f"       out_root : {out_root}")
    print(f"       out_yaml : {out_yaml}")
    print(f"       report   : {out_root / 'build_from_single_report.json'}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build PlantSeg multiclass dataset from a generated singleclass dataset. "
            "The singleclass dataset defines images and bbox set. "
            "Original PlantSeg labels recover old class ids. "
            "Final class ids are taken from build_summary.json['multiclass']['classes']."
        )
    )

    parser.add_argument(
        "--single-data-yaml",
        type=str,
        required=True,
        help="Generated singleclass data yaml.",
    )

    parser.add_argument(
        "--source-label-root",
        type=str,
        required=True,
        help="Original PlantSeg multiclass labels root.",
    )

    parser.add_argument(
        "--source-classes-txt",
        type=str,
        required=True,
        help="Original PlantSeg classes.txt.",
    )

    parser.add_argument(
        "--build-summary",
        type=str,
        required=True,
        help="build_summary.json from the single/multiclass construction.",
    )

    parser.add_argument(
        "--out-root",
        type=str,
        required=True,
        help="Output multiclass dataset root.",
    )

    parser.add_argument(
        "--out-yaml",
        type=str,
        required=True,
        help="Output multiclass data yaml.",
    )

    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits. Default: train,val,test.",
    )

    parser.add_argument(
        "--bbox-tol",
        type=float,
        default=1e-6,
        help="BBox coordinate tolerance. Use 1e-4 if labels were rounded differently.",
    )

    parser.add_argument(
        "--no-clean-output",
        action="store_true",
        help="Do not clean output root before writing.",
    )

    args = parser.parse_args()

    splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    build_from_single_with_summary(
        single_data_yaml=Path(args.single_data_yaml),
        source_label_root=Path(args.source_label_root),
        source_classes_txt=Path(args.source_classes_txt),
        build_summary=Path(args.build_summary),
        out_root=Path(args.out_root),
        out_yaml=Path(args.out_yaml),
        splits=splits,
        bbox_tol=float(args.bbox_tol),
        clean_output=not args.no_clean_output,
    )


if __name__ == "__main__":
    main()