import shutil
from pathlib import Path
import yaml
import random

from yolov8_utils import make_abs_path

# ================================
# 📂 加载配置文件 detect_durian_leaf.yaml
# ================================
yaml_path = make_abs_path("datasets/detect/detect_durian_leaf.yaml")
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)
base_dir = Path(yaml_path)

# 🔠 类别映射
class_names = data['names']

# 📁 原始数据集路径（train/val/test）
splits = ['train', 'val', 'test']
base_dirs = [base_dir.joinpath(data[split]) for split in splits]

# 📁 创建临时分类目录 sorted
sorted_dir = Path('sorted')
sorted_dir.mkdir(exist_ok=True)

# ================================
# 🧩 第一步：将图片按标签类别归类到 sorted/<class> 目录
# ================================
for base_dir in base_dirs:
    image_dir = Path(base_dir)
    label_dir = image_dir.parent.joinpath('labels')

    if not image_dir.exists() or not label_dir.exists():
        print(f"⚠️ [跳过] 不存在的目录：{image_dir} 或 {label_dir}")
        continue

    for image_path in image_dir.glob('*.*'):
        label_path = label_dir.joinpath(image_path.stem + '.txt')

        if not label_path.exists():
            print(f"⚠️ [跳过] 标签文件不存在：{label_path}")
            continue

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            cls_name = class_names[cls_id]
            # //todo:统计分类label取投票最多

            class_output_dir = sorted_dir.joinpath(cls_name)
            class_output_dir.mkdir(parents=True, exist_ok=True)

            dst_file = class_output_dir.joinpath(image_path.name)
            if not dst_file.exists():  # 避免重复拷贝
                shutil.copy(image_path, dst_file)

            # 一个图像有多个类时，只归类一次
            break

print("\n✅ 第一步完成：图片已按类别归类到 'sorted/' 目录。")

# ================================
# 🎯 第二步：按比例划分为 train / val / test
# ================================
train_dir = Path("datasets/classify")
if train_dir.exists() and train_dir.is_dir():
    shutil.rmtree(train_dir)
train_dir.mkdir(parents=True, exist_ok=True)

# 📊 划分比例
split_ratio = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
}

# 🎲 设置随机种子，确保可复现
random.seed(42)

# 🚀 开始按类别划分
for class_dir in sorted_dir.iterdir():
    if not class_dir.is_dir():
        continue

    class_name = class_dir.name
    images = list(class_dir.glob("*.*"))
    random.shuffle(images)

    total = len(images)
    n_train = int(total * split_ratio["train"])
    n_val   = int(total * split_ratio["val"])
    n_test  = total - n_train - n_val

    subsets = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:]
    }

    # 拷贝文件到目标目录
    for split, files in subsets.items():
        target_dir = train_dir.joinpath(split, class_name)
        target_dir.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            dst = target_dir.joinpath(img_path.name)
            shutil.copy(img_path, dst)

        print(f"✅ 类别 [{class_name}] → {split} 集合：共 {len(files)} 张图")

print("\n🎉 第二步完成：所有图像已按比例划分并拷贝到 'datasets/classify/' 目录。")

# data.yaml文件输出
data = {
    "train": "train",
    "val": "val",
    "test": "test",
    "nc": len(class_names),
    "names": {cls_id: cls_name for cls_id, cls_name in enumerate(sorted(class_names))}
}
yaml_path = train_dir.joinpath("data.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(data, f, sort_keys=False)
print(f"data.yaml 已生成在: {yaml_path}")

# ================================
# 🧹 第三步：删除临时目录 sorted
# ================================
if sorted_dir.exists() and sorted_dir.is_dir():
    shutil.rmtree(sorted_dir)
    print("\n🗑️ 第三步完成：已删除临时目录 'sorted/'")
else:
    print("\n⚠️ 第三步完成：临时目录 'sorted/' 不存在，跳过删除。")
