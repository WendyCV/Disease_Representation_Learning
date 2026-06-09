import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict


IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_image_label_pairs(src_dir: Path) -> List[Tuple[Path, Path]]:
    """
    递归扫描 src_dir，寻找图片与同名 txt 标签配对。
    例如：
      xxx.jpg <-> xxx.txt
    """
    pairs = []
    all_images = []

    for p in src_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_SUFFIXES:
            all_images.append(p)

    for img_path in all_images:
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            pairs.append((img_path, txt_path))
        else:
            print(f"[Warning] 缺少标签文件，跳过: {img_path}")

    return pairs


def split_dataset(
    pairs: List[Tuple[Path, Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    total = len(pairs)
    if total == 0:
        raise ValueError("未找到任何有效的图片-标签配对数据。")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio 必须等于 1.0，当前为 {ratio_sum}"
        )

    random.seed(seed)
    pairs = pairs[:]
    random.shuffle(pairs)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    return train_pairs, val_pairs, test_pairs


def ensure_dirs(root: Path):
    for split in ["train", "val", "test"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def clear_output_dir(root: Path):
    if root.exists():
        shutil.rmtree(root)


def copy_pairs(pairs: List[Tuple[Path, Path]], dst_root: Path, split: str):
    img_dst_dir = dst_root / "images" / split
    lbl_dst_dir = dst_root / "labels" / split

    for img_path, txt_path in pairs:
        shutil.copy2(img_path, img_dst_dir / img_path.name)
        shutil.copy2(txt_path, lbl_dst_dir / txt_path.name)


def write_yaml(yaml_path: Path, dataset_root: Path, class_names: List[str]):
    lines = []
    lines.append(f"path: {dataset_root.as_posix()}")
    lines.append("")
    lines.append("train: images/train")
    lines.append("val: images/val")
    lines.append("test: images/test")
    lines.append("")
    lines.append("names:")

    for i, name in enumerate(class_names):
        lines.append(f"  {i}: {name}")

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(lines), encoding="utf-8")


def parse_label_file(txt_path: Path) -> List[int]:
    """
    读取单个 YOLO 标签文件中的所有类别 id
    """
    class_ids = []
    lines = txt_path.read_text(encoding="utf-8").splitlines()

    for line_idx, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"标签格式错误: {txt_path} 第 {line_idx} 行，"
                f"YOLO 检测标签应为 5 列: class x_center y_center width height"
            )

        try:
            cls_id = int(float(parts[0]))
        except Exception:
            raise ValueError(
                f"标签类别解析失败: {txt_path} 第 {line_idx} 行，类别值为 {parts[0]}"
            )

        class_ids.append(cls_id)

    return class_ids


def validate_labels(
    pairs: List[Tuple[Path, Path]],
    class_names: List[str],
) -> Dict:
    """
    检查：
    1) 标签类别 id 是否越界
    2) 统计每个类别出现次数
    """
    num_classes = len(class_names)
    cls_count = {i: 0 for i in range(num_classes)}
    invalid_records = []

    for _, txt_path in pairs:
        class_ids = parse_label_file(txt_path)
        for cls_id in class_ids:
            if 0 <= cls_id < num_classes:
                cls_count[cls_id] += 1
            else:
                invalid_records.append({
                    "file": str(txt_path),
                    "invalid_class_id": cls_id,
                    "valid_range": f"[0, {num_classes - 1}]",
                })

    return {
        "num_classes": num_classes,
        "class_names": class_names,
        "class_counts": cls_count,
        "invalid_records": invalid_records,
    }


def print_class_stats(stats: Dict):
    print("=" * 80)
    print("标签类别统计")
    print("=" * 80)
    for cls_id, cls_name in enumerate(stats["class_names"]):
        cnt = stats["class_counts"].get(cls_id, 0)
        print(f"[{cls_id}] {cls_name}: {cnt}")

    invalid_records = stats["invalid_records"]
    if invalid_records:
        print("=" * 80)
        print("[Error] 发现越界类别 id：")
        for item in invalid_records[:20]:
            print(
                f"  file={item['file']}, invalid_class_id={item['invalid_class_id']}, "
                f"valid_range={item['valid_range']}"
            )
        if len(invalid_records) > 20:
            print(f"  ... 其余还有 {len(invalid_records) - 20} 条")
        raise ValueError("标签中存在越界类别 id，请先修正后再生成数据集。")


def outputs_already_exist(out_dir: Path, yaml_path: Path) -> bool:
    return out_dir.exists() and yaml_path.exists()


def main():
    parser = argparse.ArgumentParser(description="划分榴莲叶病害检测数据集并生成 YAML")
    parser.add_argument(
        "--src_dir",
        type=str,
        default="./data/labeled_train",
        help="原始带标注数据目录（图片和txt标签）",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./data/det_dataset",
        help="划分后的 YOLO 数据集输出目录",
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default="./data/det_dataset.yaml",
        help="生成的 yaml 文件路径",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--force",
        action="store_true",
        help="若输出目录和 yaml 已存在，是否强制重新生成",
    )

    # 默认类别顺序按你提供的原有标签分类
    parser.add_argument(
        "--class_names",
        nargs="+",
        default=["algal_leaf_spot", "no_disease", "leaf_blight", "leaf_spot"],
        help="类别名称列表，顺序必须与 YOLO 标签类别 id 一致",
    )

    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir).resolve()
    yaml_path = Path(args.yaml_path)

    if not src_dir.exists():
        raise FileNotFoundError(f"源目录不存在: {src_dir}")

    print("=" * 80)
    print("数据准备参数")
    print("=" * 80)
    print(f"src_dir     : {src_dir}")
    print(f"out_dir     : {out_dir}")
    print(f"yaml_path   : {yaml_path}")
    print(f"train/val/test = {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")
    print(f"seed        : {args.seed}")
    print(f"class_names : {args.class_names}")
    print(f"force       : {args.force}")

    if outputs_already_exist(out_dir, yaml_path) and not args.force:
        print("=" * 80)
        print("[Skip] 检测到输出目录和 yaml 已存在，默认不重复生成。")
        print("如需重新划分并覆盖，请加参数：--force")
        print("=" * 80)
        return

    print("=" * 80)
    print("Step 1/5: 扫描图片与标签配对")
    print("=" * 80)
    pairs = find_image_label_pairs(src_dir)
    print(f"找到有效样本数: {len(pairs)}")

    print("=" * 80)
    print("Step 2/5: 检查标签类别合法性")
    print("=" * 80)
    stats = validate_labels(pairs, args.class_names)
    print_class_stats(stats)

    print("=" * 80)
    print("Step 3/5: 划分 train / val / test")
    print("=" * 80)
    train_pairs, val_pairs, test_pairs = split_dataset(
        pairs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"train: {len(train_pairs)}")
    print(f"val  : {len(val_pairs)}")
    print(f"test : {len(test_pairs)}")

    print("=" * 80)
    print("Step 4/5: 生成标准 YOLO 目录")
    print("=" * 80)
    if args.force and out_dir.exists():
        print(f"[Info] --force 已启用，删除旧目录: {out_dir}")
        clear_output_dir(out_dir)

    ensure_dirs(out_dir)
    copy_pairs(train_pairs, out_dir, "train")
    copy_pairs(val_pairs, out_dir, "val")
    copy_pairs(test_pairs, out_dir, "test")
    print(f"数据集已输出到: {out_dir}")

    print("=" * 80)
    print("Step 5/5: 生成 YAML")
    print("=" * 80)
    write_yaml(yaml_path, out_dir, args.class_names)
    print(f"YAML 已生成: {yaml_path}")

    print("=" * 80)
    print("完成。stage2 配置里可使用：")
    print(yaml_path.as_posix())
    print("=" * 80)


if __name__ == "__main__":
    main()