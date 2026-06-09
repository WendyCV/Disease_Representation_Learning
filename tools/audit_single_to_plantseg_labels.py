from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: List[dict]) -> None:
    ensure_dir(path.parent)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = []
    seen = set()

    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_yaml_path(data_yaml: Path, value: str) -> Path:
    data = load_yaml(data_yaml)
    yaml_dir = data_yaml.parent

    dataset_root = data.get("path", None)
    if dataset_root is None:
        base = yaml_dir
    else:
        dataset_root = Path(str(dataset_root))
        base = dataset_root if dataset_root.is_absolute() else (yaml_dir / dataset_root).resolve()

    p = Path(value)
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def get_dataset_root(data_yaml: Path) -> Path:
    data = load_yaml(data_yaml)
    yaml_dir = data_yaml.parent

    dataset_root = data.get("path", None)
    if dataset_root is None:
        return yaml_dir.resolve()

    dataset_root = Path(str(dataset_root))
    return dataset_root.resolve() if dataset_root.is_absolute() else (yaml_dir / dataset_root).resolve()


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


def read_label_rows(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    rows: List[Tuple[int, float, float, float, float]] = []

    if not label_path.exists():
        return rows

    for line_idx, line in enumerate(label_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        try:
            cls_id = int(float(parts[0]))
            x, y, w, h = [float(v) for v in parts[1:5]]
        except Exception:
            continue

        rows.append((cls_id, x, y, w, h))

    return rows


def read_source_classes_txt(path: Path) -> Dict[int, str]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    out = {}
    for i, line in enumerate(lines):
        name = line.strip()
        if name:
            out[i] = name

    return out


def build_label_index(source_label_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}

    for p in sorted(source_label_root.rglob("*.txt")):
        if p.name.lower() == "classes.txt":
            continue

        stem = p.stem
        if stem in index:
            print(f"[WARN] duplicate label stem, keep first: {stem}")
            print(f"       keep: {index[stem]}")
            print(f"       skip: {p}")
            continue

        index[stem] = p

    return index


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


def match_single_boxes_to_source(
    single_rows: List[Tuple[int, float, float, float, float]],
    source_rows: List[Tuple[int, float, float, float, float]],
    bbox_tol: float,
):
    """
    single_rows: class is 0, bbox defines kept boxes.
    source_rows: old class id + original bbox.
    """
    matched = []
    unmatched = []
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
            "old_cls": int(old_cls),
            "x": sx,
            "y": sy,
            "w": sw,
            "h": sh,
            "single_box_index": s_idx,
            "source_box_index": best_idx,
            "bbox_dist": float(best_dist),
        })

    return matched, unmatched


def load_summary_expected(summary_path: Path):
    summary = load_json(summary_path)

    single = summary.get("single", {})
    multiclass = summary.get("multiclass", {})

    expected_single = {
        "total_images": int(single.get("total_images", -1)),
        "total_boxes": int(single.get("total_boxes", -1)),
        "split_counts": single.get("split_counts", {}),
    }

    classes_raw = multiclass.get("classes", {})
    final_id_to_name = {int(k): v for k, v in classes_raw.items()}
    final_id_to_name = dict(sorted(final_id_to_name.items(), key=lambda x: x[0]))
    final_name_to_id = {v: k for k, v in final_id_to_name.items()}

    expected_multi = {
        "num_classes": int(multiclass.get("num_classes", len(final_id_to_name))),
        "total_images": int(multiclass.get("total_images", -1)),
        "total_boxes": int(multiclass.get("total_boxes", -1)),
        "classes": final_id_to_name,
        "name_to_id": final_name_to_id,
    }

    return summary, expected_single, expected_multi


def audit(
    single_data_yaml: Path,
    source_label_root: Path,
    source_classes_txt: Path,
    build_summary: Path,
    out_dir: Path,
    splits: List[str],
    bbox_tol: float,
):
    single_data_yaml = single_data_yaml.resolve()
    source_label_root = source_label_root.resolve()
    source_classes_txt = source_classes_txt.resolve()
    build_summary = build_summary.resolve()
    out_dir = out_dir.resolve()

    ensure_dir(out_dir)

    summary, expected_single, expected_multi = load_summary_expected(build_summary)
    source_id_to_name = read_source_classes_txt(source_classes_txt)
    label_index = build_label_index(source_label_root)

    final_name_to_id = expected_multi["name_to_id"]
    final_id_to_name = expected_multi["classes"]

    print("[INFO] Audit singleclass against source PlantSeg labels")
    print(f"       single_data_yaml  : {single_data_yaml}")
    print(f"       source_label_root : {source_label_root}")
    print(f"       source_classes_txt: {source_classes_txt}")
    print(f"       build_summary     : {build_summary}")
    print(f"       out_dir           : {out_dir}")
    print(f"       bbox_tol          : {bbox_tol}")
    print("")
    print("[EXPECTED FROM SUMMARY]")
    print(f"       single images : {expected_single['total_images']}")
    print(f"       single boxes  : {expected_single['total_boxes']}")
    print(f"       multi classes : {expected_multi['num_classes']}")
    print(f"       multi images  : {expected_multi['total_images']}")
    print(f"       multi boxes   : {expected_multi['total_boxes']}")
    print("")

    image_rows = []
    box_rows = []
    unmatched_rows = []
    missing_rows = []

    source_class_counts: Dict[int, int] = {}
    final_class_counts: Dict[int, int] = {}
    not_in_summary_class_counts: Dict[str, int] = {}

    total_images = 0
    total_single_label_files = 0
    total_single_boxes = 0
    total_matched_boxes = 0
    total_unmatched_boxes = 0
    images_with_all_boxes_matched = 0
    images_with_any_unmatched = 0
    images_with_source_label = 0
    images_missing_source_label = 0

    split_summary = {}

    for split in splits:
        split_img_dir = get_split_image_dir(single_data_yaml, split)
        if split_img_dir is None or not split_img_dir.exists():
            print(f"[WARN] split image dir not found: {split}")
            continue

        images = iter_images(split_img_dir)

        split_images = 0
        split_boxes = 0
        split_matched = 0
        split_unmatched = 0
        split_missing_source = 0

        for img_path in images:
            total_images += 1
            split_images += 1

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
                image_rows.append({
                    "split": split,
                    "image_name": img_path.name,
                    "num_single_boxes": 0,
                    "num_matched_boxes": 0,
                    "num_unmatched_boxes": 0,
                    "source_label_exists": 0,
                    "all_boxes_matched": 0,
                })
                continue

            total_single_label_files += 1
            total_single_boxes += len(single_rows)
            split_boxes += len(single_rows)

            source_label_path = label_index.get(img_path.stem, None)

            if source_label_path is None:
                images_missing_source_label += 1
                split_missing_source += 1
                total_unmatched_boxes += len(single_rows)
                split_unmatched += len(single_rows)

                missing_rows.append({
                    "split": split,
                    "image_name": img_path.name,
                    "image_path": str(img_path),
                    "single_label_path": str(single_label_path),
                    "reason": "missing_source_label",
                })

                image_rows.append({
                    "split": split,
                    "image_name": img_path.name,
                    "num_single_boxes": len(single_rows),
                    "num_matched_boxes": 0,
                    "num_unmatched_boxes": len(single_rows),
                    "source_label_exists": 0,
                    "all_boxes_matched": 0,
                    "source_label_path": "",
                })
                continue

            images_with_source_label += 1
            source_rows = read_label_rows(source_label_path)

            matched, unmatched = match_single_boxes_to_source(
                single_rows=single_rows,
                source_rows=source_rows,
                bbox_tol=bbox_tol,
            )

            total_matched_boxes += len(matched)
            total_unmatched_boxes += len(unmatched)
            split_matched += len(matched)
            split_unmatched += len(unmatched)

            if unmatched:
                images_with_any_unmatched += 1
            else:
                images_with_all_boxes_matched += 1

            for u in unmatched:
                unmatched_rows.append({
                    "split": split,
                    "image_name": img_path.name,
                    "image_path": str(img_path),
                    "single_label_path": str(single_label_path),
                    "source_label_path": str(source_label_path),
                    **u,
                })

            recovered_old_classes = set()
            recovered_final_classes = set()
            not_in_summary_names = set()

            for m in matched:
                old_cls = int(m["old_cls"])
                old_name = source_id_to_name.get(old_cls, f"class_{old_cls}")

                source_class_counts[old_cls] = source_class_counts.get(old_cls, 0) + 1
                recovered_old_classes.add(old_cls)

                final_id = final_name_to_id.get(old_name, None)

                if final_id is None:
                    not_in_summary_class_counts[old_name] = not_in_summary_class_counts.get(old_name, 0) + 1
                    not_in_summary_names.add(old_name)
                else:
                    final_class_counts[final_id] = final_class_counts.get(final_id, 0) + 1
                    recovered_final_classes.add(final_id)

                box_rows.append({
                    "split": split,
                    "image_name": img_path.name,
                    "single_label_path": str(single_label_path),
                    "source_label_path": str(source_label_path),
                    "single_box_index": m["single_box_index"],
                    "source_box_index": m["source_box_index"],
                    "bbox_dist": m["bbox_dist"],
                    "old_class_id": old_cls,
                    "old_class_name": old_name,
                    "final_class_id": final_id if final_id is not None else "",
                    "final_class_name": final_id_to_name.get(final_id, "") if final_id is not None else "",
                    "x": f"{m['x']:.6f}",
                    "y": f"{m['y']:.6f}",
                    "w": f"{m['w']:.6f}",
                    "h": f"{m['h']:.6f}",
                })

            image_rows.append({
                "split": split,
                "image_name": img_path.name,
                "image_path": str(img_path),
                "single_label_path": str(single_label_path),
                "source_label_path": str(source_label_path),
                "num_single_boxes": len(single_rows),
                "num_source_boxes": len(source_rows),
                "num_matched_boxes": len(matched),
                "num_unmatched_boxes": len(unmatched),
                "source_label_exists": 1,
                "all_boxes_matched": int(len(unmatched) == 0),
                "old_class_ids": ",".join(map(str, sorted(recovered_old_classes))),
                "final_class_ids": ",".join(map(str, sorted(recovered_final_classes))),
                "not_in_summary_class_names": ",".join(sorted(not_in_summary_names)),
            })

        split_summary[split] = {
            "images": split_images,
            "single_boxes": split_boxes,
            "matched_boxes": split_matched,
            "unmatched_boxes": split_unmatched,
            "missing_source_labels": split_missing_source,
        }

    source_class_rows = []
    for old_cls in sorted(source_class_counts.keys()):
        old_name = source_id_to_name.get(old_cls, f"class_{old_cls}")
        source_class_rows.append({
            "old_class_id": old_cls,
            "old_class_name": old_name,
            "box_count_in_single_boxes": source_class_counts[old_cls],
            "in_summary_multiclass": int(old_name in final_name_to_id),
            "summary_final_class_id": final_name_to_id.get(old_name, ""),
        })

    final_class_rows = []
    for final_id, final_name in final_id_to_name.items():
        final_class_rows.append({
            "final_class_id": final_id,
            "final_class_name": final_name,
            "box_count_recovered_from_single": final_class_counts.get(final_id, 0),
            "appears_in_single": int(final_class_counts.get(final_id, 0) > 0),
        })

    not_in_summary_rows = []
    for name, count in sorted(not_in_summary_class_counts.items(), key=lambda x: x[0]):
        not_in_summary_rows.append({
            "class_name": name,
            "box_count": count,
            "reason": "recovered_from_source_but_not_in_summary_multiclass_classes",
        })

    actual_unique_source_classes = len(source_class_counts)
    actual_unique_final_classes = len([k for k, v in final_class_counts.items() if v > 0])

    report = {
        "inputs": {
            "single_data_yaml": str(single_data_yaml),
            "source_label_root": str(source_label_root),
            "source_classes_txt": str(source_classes_txt),
            "build_summary": str(build_summary),
            "bbox_tol": bbox_tol,
            "splits": splits,
        },
        "expected_from_build_summary": {
            "single_total_images": expected_single["total_images"],
            "single_total_boxes": expected_single["total_boxes"],
            "single_split_counts": expected_single["split_counts"],
            "multiclass_num_classes": expected_multi["num_classes"],
            "multiclass_total_images": expected_multi["total_images"],
            "multiclass_total_boxes": expected_multi["total_boxes"],
        },
        "actual_from_single_dataset": {
            "total_images": total_images,
            "total_single_label_files": total_single_label_files,
            "total_single_boxes": total_single_boxes,
            "split_summary": split_summary,
        },
        "source_label_matching": {
            "images_with_source_label": images_with_source_label,
            "images_missing_source_label": images_missing_source_label,
            "total_matched_boxes": total_matched_boxes,
            "total_unmatched_boxes": total_unmatched_boxes,
            "images_with_all_boxes_matched": images_with_all_boxes_matched,
            "images_with_any_unmatched": images_with_any_unmatched,
        },
        "class_recovery": {
            "unique_source_old_classes_recovered": actual_unique_source_classes,
            "unique_summary_final_classes_recovered": actual_unique_final_classes,
            "expected_summary_final_classes": expected_multi["num_classes"],
            "classes_recovered_but_not_in_summary": len(not_in_summary_class_counts),
            "boxes_recovered_but_not_in_summary": sum(not_in_summary_class_counts.values()),
        },
        "checks": {
            "single_images_match_summary": total_images == expected_single["total_images"],
            "single_boxes_match_summary": total_single_boxes == expected_single["total_boxes"],
            "all_single_boxes_matched_to_source": total_unmatched_boxes == 0,
            "final_class_count_matches_summary": actual_unique_final_classes == expected_multi["num_classes"],
        },
    }

    save_json(out_dir / "audit_report.json", report)
    write_csv(out_dir / "audit_images.csv", image_rows)
    write_csv(out_dir / "audit_boxes.csv", box_rows)
    write_csv(out_dir / "audit_unmatched_boxes.csv", unmatched_rows)
    write_csv(out_dir / "audit_missing_rows.csv", missing_rows)
    write_csv(out_dir / "audit_source_class_distribution.csv", source_class_rows)
    write_csv(out_dir / "audit_summary_final_class_distribution.csv", final_class_rows)
    write_csv(out_dir / "audit_classes_recovered_but_not_in_summary.csv", not_in_summary_rows)

    print("\n[AUDIT DONE]")
    print("single images expected:", expected_single["total_images"])
    print("single images actual  :", total_images)
    print("single boxes expected :", expected_single["total_boxes"])
    print("single boxes actual   :", total_single_boxes)
    print("matched boxes         :", total_matched_boxes)
    print("unmatched boxes       :", total_unmatched_boxes)
    print("source old classes    :", actual_unique_source_classes)
    print("summary final classes expected:", expected_multi["num_classes"])
    print("summary final classes recovered:", actual_unique_final_classes)
    print("classes recovered but not in summary:", len(not_in_summary_class_counts))
    print("boxes recovered but not in summary  :", sum(not_in_summary_class_counts.values()))
    print("")
    print("checks:")
    for k, v in report["checks"].items():
        print(f"  {k}: {v}")
    print("")
    print("report:", out_dir / "audit_report.json")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--single-data-yaml",
        type=str,
        required=True,
        help="Singleclass data.yaml.",
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
        help="build_summary.json from PlantSeg dataset construction.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="./runs_audit/audit_single_to_plantseg_labels",
        help="Output audit directory.",
    )

    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits.",
    )

    parser.add_argument(
        "--bbox-tol",
        type=float,
        default=1e-6,
        help="BBox coordinate tolerance. Use 1e-4 if label decimals differ.",
    )

    args = parser.parse_args()

    splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    audit(
        single_data_yaml=Path(args.single_data_yaml),
        source_label_root=Path(args.source_label_root),
        source_classes_txt=Path(args.source_classes_txt),
        build_summary=Path(args.build_summary),
        out_dir=Path(args.out_dir),
        splits=splits,
        bbox_tol=float(args.bbox_tol),
    )


if __name__ == "__main__":
    main()