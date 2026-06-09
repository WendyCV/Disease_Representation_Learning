# move_yolo_by_name_or_label.py
import argparse
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_names_from_file(path: str):
    names = []
    if not path:
        return names

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"names file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)

    return names


def normalize_label_ids(labels):
    """
    YOLO label txt 第一列一般是 class id。
    这里统一转成字符串比较，例如 0, 1, 2。
    """
    if not labels:
        return set()

    return {str(int(x)) for x in labels}


def image_name_matched(image_path: Path, name_set: set):
    """
    支持两种写法：
    1. 带扩展名：xxx.jpg
    2. 不带扩展名：xxx
    """
    if not name_set:
        return False

    for name in name_set:
        if name in image_path.name:
            return True
    return False


def label_matched(label_path: Path, label_id_set: set):
    """
    只要一个 label 文件中任意一行的 class id 命中，就认为匹配。
    YOLO格式：
    class_id x_center y_center width height
    """
    if not label_id_set:
        return False

    if not label_path.exists():
        return False

    try:
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if not parts:
                    continue

                class_id = parts[0]
                if class_id in label_id_set:
                    return True

    except UnicodeDecodeError:
        print(f"[WARN] Cannot read label file: {label_path}")
        return False

    return False


def safe_transfer(src: Path, dst: Path, copy_mode=False, overwrite=False, dry_run=False):
    if not src.exists():
        print(f"[WARN] Source not found, skip: {src}")
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        if overwrite:
            if dry_run:
                print(f"[DRY-RUN] overwrite existing: {dst}")
            else:
                dst.unlink()
        else:
            print(f"[SKIP] Target exists: {dst}")
            return False

    action = "COPY" if copy_mode else "MOVE"

    if dry_run:
        print(f"[DRY-RUN] {action}: {src} -> {dst}")
        return True

    if copy_mode:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))

    print(f"[{action}] {src} -> {dst}")
    return True


def collect_images(images_dir: Path):
    images = []
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images.append(p)
    return images


def main():
    parser = argparse.ArgumentParser(
        description="Move or copy YOLO images and labels by image names or class labels."
    )

    parser.add_argument(
        "--src",
        required=True,
        help="源数据集根目录，下面应包含 images/ 和 labels/"
    )

    parser.add_argument(
        "--dst",
        required=True,
        help="目标目录，脚本会自动创建 images/ 和 labels/"
    )

    parser.add_argument(
        "--names",
        nargs="*",
        default=[],
        help="指定图片名，支持多个。可以写 xxx.jpg，也可以只写 xxx"
    )

    parser.add_argument(
        "--names-file",
        default=None,
        help="图片名列表 txt 文件，每行一个图片名，可带扩展名，也可不带扩展名"
    )

    parser.add_argument(
        "--labels",
        nargs="*",
        default=[],
        help="指定 YOLO class id，支持多个，例如 --labels 0 3 5"
    )

    parser.add_argument(
        "--logic",
        choices=["any", "all"],
        default="any",
        help="当同时指定 names 和 labels 时，any 表示满足任一条件；all 表示两个条件都满足。默认 any"
    )

    parser.add_argument(
        "--copy",
        action="store_true",
        help="复制而不是移动。默认是移动"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果目标文件已存在，是否覆盖"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要移动/复制的文件，不真正执行"
    )

    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    src_images_dir = src_root / "images"
    src_labels_dir = src_root / "labels"

    dst_images_dir = dst_root / "images"
    dst_labels_dir = dst_root / "labels"

    if not src_images_dir.exists():
        raise FileNotFoundError(f"images directory not found: {src_images_dir}")

    if not src_labels_dir.exists():
        raise FileNotFoundError(f"labels directory not found: {src_labels_dir}")

    name_list = list(args.names)
    name_list.extend(read_names_from_file(args.names_file))
    name_set = set(name_list)

    label_id_set = normalize_label_ids(args.labels)

    if not name_set and not label_id_set:
        raise ValueError("必须至少指定 --names / --names-file / --labels 其中一种筛选条件。")

    images = collect_images(src_images_dir)

    matched_items = []

    for image_path in images:
        rel_path = image_path.relative_to(src_images_dir)
        label_path = src_labels_dir / rel_path.with_suffix(".txt")

        by_name = image_name_matched(image_path, name_set)
        by_label = label_matched(label_path, label_id_set)

        if name_set and label_id_set:
            if args.logic == "any":
                matched = by_name or by_label
            else:
                matched = by_name and by_label
        elif name_set:
            matched = by_name
        else:
            matched = by_label

        if matched:
            matched_items.append((image_path, label_path, rel_path))

    print(f"\nFound matched images: {len(matched_items)}\n")

    moved_images = 0
    moved_labels = 0
    missing_labels = 0

    for image_path, label_path, rel_path in matched_items:
        dst_image_path = dst_images_dir / rel_path
        dst_label_path = dst_labels_dir / rel_path.with_suffix(".txt")

        ok_img = safe_transfer(
            src=image_path,
            dst=dst_image_path,
            copy_mode=args.copy,
            overwrite=args.overwrite,
            dry_run=args.dry_run
        )

        if ok_img:
            moved_images += 1

        if label_path.exists():
            ok_label = safe_transfer(
                src=label_path,
                dst=dst_label_path,
                copy_mode=args.copy,
                overwrite=args.overwrite,
                dry_run=args.dry_run
            )
            if ok_label:
                moved_labels += 1
        else:
            missing_labels += 1
            print(f"[WARN] Label not found for image: {image_path}")

    print("\nDone.")
    print(f"Images processed: {moved_images}")
    print(f"Labels processed: {moved_labels}")
    print(f"Missing labels: {missing_labels}")


if __name__ == "__main__":
    main()