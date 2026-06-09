# count_yolo_dataset_stats.py
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


KNOWN_PLANTS = [
    "bell pepper",
    "blueberry",
    "raspberry",
    "strawberry",
    "tomato",
    "potato",
    "soybean",
    "cabbage",
    "banana",
    "citrus",
    "orange",
    "apple",
    "peach",
    "corn",
    "rice",
    "garlic",
    "squash",
    "grape",
    "cherry",
    "durian",
]


def load_yaml(path: Path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("缺少 PyYAML，请先安装：pip install pyyaml") from exc

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_class_names(names_obj):
    """
    支持两种 YOLO names 格式：

    names:
      0: Apple leaf
      1: Corn leaf blight

    或：

    names: [Apple leaf, Corn leaf blight]
    """
    if names_obj is None:
        return {}

    if isinstance(names_obj, list):
        return {i: str(name) for i, name in enumerate(names_obj)}

    if isinstance(names_obj, dict):
        out = {}
        for k, v in names_obj.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        return dict(sorted(out.items()))

    return {}


def resolve_dataset_root(data_yaml_path: Path, data: dict):
    """
    YOLO data.yaml 常见格式：

    path: D:/dataset/PlantDoc
    train: images/train
    val: images/val
    test: images/test

    如果没有 path，则以 data.yaml 所在目录为 root。
    """
    yaml_dir = data_yaml_path.parent

    root = data.get("path", None)
    if root is None:
        return yaml_dir.resolve()

    root = Path(root)
    if root.is_absolute():
        return root.resolve()

    return (yaml_dir / root).resolve()


def resolve_split_paths(data_yaml_path: Path, data: dict, split: str):
    """
    train/val/test 可以是：
    1. 一个目录
    2. 一个 txt 文件，里面每行是图片路径
    3. 一个列表
    """
    root = resolve_dataset_root(data_yaml_path, data)
    value = data.get(split, None)

    if value is None:
        return []

    if isinstance(value, (str, Path)):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        return []

    paths = []
    for item in values:
        p = Path(item)
        if not p.is_absolute():
            p = root / p
        paths.append(p.resolve())

    return paths


def collect_images_from_path(path: Path):
    images = []

    if path.is_file() and path.suffix.lower() == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                p = Path(line)
                if p.exists() and p.suffix.lower() in IMAGE_EXTS:
                    images.append(p.resolve())

        return sorted(set(images))

    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        return [path.resolve()]

    if path.is_dir():
        for ext in IMAGE_EXTS:
            images.extend(path.rglob(f"*{ext}"))
            images.extend(path.rglob(f"*{ext.upper()}"))

        return sorted(set(p.resolve() for p in images))

    return []


def infer_label_path_from_image(image_path: Path):
    """
    默认 YOLO 目录结构：

    images/train/xxx.jpg
    labels/train/xxx.txt

    通过把路径中的 images 替换成 labels 来找到 label。
    """
    parts = list(image_path.parts)

    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")

    s = str(image_path)

    if "images" in s:
        return Path(s.replace("images", "labels")).with_suffix(".txt")

    return image_path.with_suffix(".txt")


def parse_label_file(label_path: Path):
    class_ids = []
    malformed_lines = 0

    if not label_path.exists():
        return class_ids, malformed_lines

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                malformed_lines += 1
                continue

            try:
                cls_id = int(float(parts[0]))
            except Exception:
                malformed_lines += 1
                continue

            class_ids.append(cls_id)

    return class_ids, malformed_lines


def infer_plant_from_class_name(class_name: str):
    """
    从类别名推断植物名。

    适合：
      Apple leaf
      Corn leaf blight
      Tomato Septoria leaf spot
      bell_pepper_blossom_end_rot
      soybean_frog_eye_leaf_spot

    如果类别命名很复杂，建议后续再手动校正输出的 class_summary.csv。
    """
    if not class_name:
        return "unknown"

    name = class_name.strip().lower()
    name = name.replace("___", " ")
    name = name.replace("_", " ")
    name = re.sub(r"[-/]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    for plant in sorted(KNOWN_PLANTS, key=len, reverse=True):
        if name == plant or name.startswith(plant + " "):
            return plant

    if not name:
        return "unknown"

    first_word = name.split(" ")[0]

    if first_word in {"disease", "lesion", "healthy", "no"}:
        return "unknown"

    return first_word


def analyse_split(split_name, images, class_names):
    class_instance_counter = Counter()
    class_image_counter = Counter()
    plant_instance_counter = Counter()
    plant_image_counter = Counter()

    missing_label_images = 0
    empty_label_images = 0
    images_with_labels = 0
    malformed_label_lines = 0
    total_instances = 0

    for image_path in images:
        label_path = infer_label_path_from_image(image_path)

        if not label_path.exists():
            missing_label_images += 1
            continue

        class_ids, malformed = parse_label_file(label_path)
        malformed_label_lines += malformed

        if not class_ids:
            empty_label_images += 1
            continue

        images_with_labels += 1
        total_instances += len(class_ids)

        unique_classes = set(class_ids)

        for cls_id in class_ids:
            class_instance_counter[cls_id] += 1

            class_name = class_names.get(cls_id, f"unknown_class_{cls_id}")
            plant = infer_plant_from_class_name(class_name)
            plant_instance_counter[plant] += 1

        for cls_id in unique_classes:
            class_image_counter[cls_id] += 1

        unique_plants = set()
        for cls_id in unique_classes:
            class_name = class_names.get(cls_id, f"unknown_class_{cls_id}")
            plant = infer_plant_from_class_name(class_name)
            unique_plants.add(plant)

        for plant in unique_plants:
            plant_image_counter[plant] += 1

    present_classes = sorted(class_instance_counter.keys())
    present_plants = sorted(
        plant for plant in plant_instance_counter.keys() if plant != "unknown"
    )

    split_summary = {
        "split": split_name,
        "images": len(images),
        "images_with_labels": images_with_labels,
        "images_missing_label_file": missing_label_images,
        "images_with_empty_label": empty_label_images,
        "instances": total_instances,
        "configured_classes": len(class_names),
        "present_classes": len(present_classes),
        "present_plants_excluding_unknown": len(present_plants),
        "malformed_label_lines": malformed_label_lines,
    }

    class_rows = []
    all_class_ids = sorted(set(class_names.keys()) | set(class_instance_counter.keys()))

    for cls_id in all_class_ids:
        class_name = class_names.get(cls_id, f"unknown_class_{cls_id}")
        plant = infer_plant_from_class_name(class_name)

        class_rows.append(
            {
                "split": split_name,
                "class_id": cls_id,
                "class_name": class_name,
                "plant": plant,
                "instances": class_instance_counter.get(cls_id, 0),
                "images_with_class": class_image_counter.get(cls_id, 0),
            }
        )

    plant_rows = []
    for plant in sorted(plant_instance_counter.keys()):
        plant_rows.append(
            {
                "split": split_name,
                "plant": plant,
                "instances": plant_instance_counter[plant],
                "images_with_plant": plant_image_counter[plant],
            }
        )

    return split_summary, class_rows, plant_rows


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Count YOLO dataset images, classes, instances, and inferred plant types from data.yaml."
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to YOLO data.yaml",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to analyse. Default: all existing train/val/test in data.yaml",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="./runs_dataset_analysis/stats_all",
        help="Output directory",
    )

    args = parser.parse_args()

    data_yaml_path = Path(args.data).resolve()
    data = load_yaml(data_yaml_path)
    class_names = normalize_class_names(data.get("names"))

    if not class_names:
        raise RuntimeError("data.yaml 中没有读取到 names，请检查 data.yaml。")

    if args.splits is None:
        candidate_splits = ["train", "val", "test"]
        splits = [s for s in candidate_splits if s in data]
    else:
        splits = args.splits

    if not splits:
        raise RuntimeError("data.yaml 中没有找到 train / val / test。")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_split_summaries = []
    all_class_rows = []
    all_plant_rows = []

    for split_name in splits:
        split_paths = resolve_split_paths(data_yaml_path, data, split_name)

        images = []
        for p in split_paths:
            images.extend(collect_images_from_path(p))

        images = sorted(set(images))

        if not images:
            print(f"[Warning] split={split_name} 没有找到图片，跳过。")
            continue

        split_summary, class_rows, plant_rows = analyse_split(
            split_name=split_name,
            images=images,
            class_names=class_names,
        )

        all_split_summaries.append(split_summary)
        all_class_rows.extend(class_rows)
        all_plant_rows.extend(plant_rows)

    if not all_split_summaries:
        raise RuntimeError("没有统计到任何 split，请检查 data.yaml 路径配置。")

    total_summary = {
        "split": "TOTAL",
        "images": sum(x["images"] for x in all_split_summaries),
        "images_with_labels": sum(x["images_with_labels"] for x in all_split_summaries),
        "images_missing_label_file": sum(
            x["images_missing_label_file"] for x in all_split_summaries
        ),
        "images_with_empty_label": sum(
            x["images_with_empty_label"] for x in all_split_summaries
        ),
        "instances": sum(x["instances"] for x in all_split_summaries),
        "configured_classes": len(class_names),
        "present_classes": len(
            {
                row["class_id"]
                for row in all_class_rows
                if int(row["instances"]) > 0
            }
        ),
        "present_plants_excluding_unknown": len(
            {
                row["plant"]
                for row in all_plant_rows
                if int(row["instances"]) > 0 and row["plant"] != "unknown"
            }
        ),
        "malformed_label_lines": sum(
            x["malformed_label_lines"] for x in all_split_summaries
        ),
    }

    all_split_summaries.append(total_summary)

    write_csv(
        out_dir / "split_summary.csv",
        all_split_summaries,
        [
            "split",
            "images",
            "images_with_labels",
            "images_missing_label_file",
            "images_with_empty_label",
            "instances",
            "configured_classes",
            "present_classes",
            "present_plants_excluding_unknown",
            "malformed_label_lines",
        ],
    )

    write_csv(
        out_dir / "class_summary.csv",
        all_class_rows,
        [
            "split",
            "class_id",
            "class_name",
            "plant",
            "instances",
            "images_with_class",
        ],
    )

    write_csv(
        out_dir / "plant_summary.csv",
        all_plant_rows,
        [
            "split",
            "plant",
            "instances",
            "images_with_plant",
        ],
    )

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_yaml": str(data_yaml_path),
                "splits": splits,
                "split_summary": all_split_summaries,
                "output_files": [
                    "split_summary.csv",
                    "class_summary.csv",
                    "plant_summary.csv",
                    "summary.json",
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n===== DATASET SUMMARY =====")
    for s in all_split_summaries:
        print(
            f"{s['split']:>8} | "
            f"images={s['images']} | "
            f"instances={s['instances']} | "
            f"configured_classes={s['configured_classes']} | "
            f"present_classes={s['present_classes']} | "
            f"plants={s['present_plants_excluding_unknown']} | "
            f"missing_labels={s['images_missing_label_file']} | "
            f"empty_labels={s['images_with_empty_label']}"
        )

    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()