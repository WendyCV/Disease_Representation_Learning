import argparse
from pathlib import Path
import shutil
import cv2
import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_yolo_yaml(yaml_path: Path):
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    dataset_root = Path(data["path"])

    split_image_dirs = {}

    for split in ["train", "val", "test"]:
        if split in data and data[split] is not None:
            img_dir = Path(data[split])
            if not img_dir.is_absolute():
                img_dir = dataset_root / img_dir
            split_image_dirs[split] = img_dir

    split_label_dirs = {}

    for split, img_dir in split_image_dirs.items():
        parts = list(img_dir.parts)
        if "images" not in parts:
            raise ValueError(f"Cannot infer labels dir from: {img_dir}")

        idx = parts.index("images")
        parts[idx] = "labels"
        split_label_dirs[split] = Path(*parts)

    return split_image_dirs, split_label_dirs


def find_image_for_label(label_path: Path, label_dir: Path, image_dir: Path):
    rel = label_path.relative_to(label_dir).with_suffix("")

    for ext in IMG_EXTS:
        candidate = image_dir / rel.with_suffix(ext)
        if candidate.exists():
            return candidate

    return None


def yolo_to_xyxy(xc, yc, bw, bh, img_w, img_h):
    xc *= img_w
    yc *= img_h
    bw *= img_w
    bh *= img_h

    x1 = xc - bw / 2
    y1 = yc - bh / 2
    x2 = xc + bw / 2
    y2 = yc + bh / 2

    return x1, y1, x2, y2


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    bw = x2 - x1
    bh = y2 - y1
    xc = x1 + bw / 2
    yc = y1 + bh / 2

    return xc / img_w, yc / img_h, bw / img_w, bh / img_h


def fix_one_label(label_path: Path, image_path: Path, dry_run: bool = False):
    img = cv2.imread(str(image_path))
    if img is None:
        return 0, 0

    img_h, img_w = img.shape[:2]

    lines = label_path.read_text(encoding="utf-8").splitlines()

    new_lines = []
    changed_count = 0
    removed_count = 0

    for line in lines:
        raw = line.strip()

        if not raw:
            continue

        parts = raw.split()

        if len(parts) != 5:
            new_lines.append(raw)
            continue

        cls = parts[0]

        try:
            xc, yc, bw, bh = map(float, parts[1:])
        except Exception:
            new_lines.append(raw)
            continue

        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, img_w, img_h)

        old = (x1, y1, x2, y2)

        x1 = max(0.0, min(float(img_w), x1))
        y1 = max(0.0, min(float(img_h), y1))
        x2 = max(0.0, min(float(img_w), x2))
        y2 = max(0.0, min(float(img_h), y2))

        # 如果 clip 后框无效，删除该框
        if x2 <= x1 or y2 <= y1:
            removed_count += 1
            continue

        if old != (x1, y1, x2, y2):
            changed_count += 1

        new_xc, new_yc, new_bw, new_bh = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)

        # 再次限制到 0-1，避免浮点误差
        new_xc = max(0.0, min(1.0, new_xc))
        new_yc = max(0.0, min(1.0, new_yc))
        new_bw = max(0.0, min(1.0, new_bw))
        new_bh = max(0.0, min(1.0, new_bh))

        new_lines.append(
            f"{cls} {new_xc:.6f} {new_yc:.6f} {new_bw:.6f} {new_bh:.6f}"
        )

    if changed_count > 0 or removed_count > 0:
        backup_path = label_path.with_suffix(".txt.bak")

        if not backup_path.exists():
            shutil.copy2(label_path, backup_path)

        if not dry_run:
            label_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    return changed_count, removed_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="YOLO dataset yaml")
    parser.add_argument("--dry-run", action="store_true", help="only check, do not write")
    args = parser.parse_args()

    yaml_path = Path(args.data)

    split_image_dirs, split_label_dirs = load_yolo_yaml(yaml_path)

    total_changed = 0
    total_removed = 0
    total_files = 0

    for split, label_dir in split_label_dirs.items():
        image_dir = split_image_dirs[split]

        if not label_dir.exists():
            continue

        for label_path in label_dir.rglob("*.txt"):
            if label_path.name.lower() == "classes.txt":
                continue

            image_path = find_image_for_label(label_path, label_dir, image_dir)

            if image_path is None:
                continue

            changed, removed = fix_one_label(
                label_path=label_path,
                image_path=image_path,
                dry_run=args.dry_run,
            )

            if changed > 0 or removed > 0:
                total_files += 1
                total_changed += changed
                total_removed += removed

                print(
                    f"[FIX] {split} | {label_path.name} | "
                    f"changed={changed}, removed={removed}"
                )

    print("\n[DONE]")
    print(f"changed boxes: {total_changed}")
    print(f"removed boxes: {total_removed}")
    print(f"affected label files: {total_files}")

    if args.dry_run:
        print("dry-run only, no files were modified.")


if __name__ == "__main__":
    main()