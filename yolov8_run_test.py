import cv2
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from yolov8_model_tools import make_abs_path

def run_find_best_k(image_path):
    from yolov8_model_tools import find_best_k 
    # 测试图片kmean算法的最优k
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab = cv2.merge((l_clahe, a, b))
        optimal_k, k_list, inertia_list = find_best_k(lab)
    # 打印结果和输出
    print("最优聚类数（肘部位置）：k =", optimal_k, inertia_list)
    # 直接显示best_k（肘部法-SSE下降曲率变缓慢）
    plt.plot(k_list, inertia_list, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia (SSE)')
    plt.title('Elbow Method for optimal k')
    plt.show()
    # 通过图片观察k=4或者k=5都可以

import torch

def run_gaussian_weight_map(height, width, sigma):
    from yolov8_utils import get_batch_center_weight_map
    # 计算高斯权重图
    # 计算中心点 (x, y)，注意归一化
    center_x = width / 2.0 / width     # = 0.5
    center_y = height / 2.0 / height   # = 0.5
    centers = torch.tensor([[center_x, center_y]], dtype=torch.float32)
    weight_map = get_batch_center_weight_map(centers, height, width, sigma=sigma)
    # 取出第一张图并 squeeze 成 (H, W)
    weight_map_np = weight_map[0, 0].cpu().numpy()  # shape: (20, 20)
    # 可视化
    plt.figure(figsize=(5, 5))
    plt.imshow(weight_map_np, cmap='hot', interpolation='nearest')
    plt.title(f"Gaussian Weight Map\n{height}x{width}, sigma={sigma}")
    plt.colorbar()
    plt.axis('off')
    plt.show()

def run_center_weight_map(height, width, sigma, k_size):
    from yolov8_utils import get_center_weight_map
    weight_map = get_center_weight_map(height, width, sigma, k_size)
    # 可视化
    weight_map = weight_map.squeeze()
    # 打印高斯权重图的最小值和最大值，检查其范围
    plt.figure(figsize=(5, 5))
    plt.imshow(weight_map.cpu().numpy(), cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    plt.title(f"Gaussian Weight Map\n{height}x{width}, sigma={sigma}, k_size={k_size}")
    plt.colorbar()
    plt.axis('off')
    plt.show()

import torchvision.utils as vutils
from pathlib import Path

def test_loader_and_mask(data_dir, img_sz=640, batch_size=64, show_aug=False):
    from torch.utils.data import DataLoader
    from yolov8_clr_train import ImageForlderLoader, clr_transforms_v2
    # 实例化数据集
    dataset = ImageForlderLoader(target_dir=data_dir, img_sz=img_sz)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 取一个batch
    images, masks, indexes = next(iter(dataloader))  # shapes: [B, 3, H, W], [B, 1, H, W]
    masked_images = images * masks.expand(-1, 3, -1, -1)
    # 可选增强测试（非同步，仅用于测试）
    if show_aug:
        clr_transform_1, clr_transform_2 = clr_transforms_v2(img_sz)

        # 1) 拼接 -> 同步几何增强
        combined = torch.cat([masked_images, masks], dim=1)         # [B,4,H,W]
        combined_aug = clr_transform_1(combined)                     # 仍是 [B,4,H,W]

        # 2) 分离（前3通道图像，后1通道mask）
        augmented_images = combined_aug[:, :3, :, :]
        augmented_masks  = combined_aug[:, 3:, :, :]

        # 3) 让mask回到0/1（二值化，避免插值带来的灰度）
        augmented_masks = (augmented_masks > 0.5).float()

        # 4) 仅对图像做颜色/模糊等增强
        augmented_images = clr_transform_2(augmented_images)
    else:
        augmented_images = None

    save_dir = Path("runs/augment")
    save_dir.mkdir(parents=True, exist_ok=True)
    # 显示图像
    num = min(batch_size, 16)
    for i in range(num):
        fig, axs = plt.subplots(1, 4 if augmented_images is not None else 3, figsize=(12 if augmented_images is not None else 9, 3))
        axs[0].imshow(images[i].clamp(0, 1).permute(1, 2, 0).numpy())
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        axs[1].imshow(masks[i].clamp(0, 1).permute(1, 2, 0).numpy(), cmap='gray')
        axs[1].set_title("Generated Mask")
        axs[1].axis('off')

        axs[2].imshow(masked_images[i].clamp(0, 1).permute(1, 2, 0).numpy())
        axs[2].set_title("Masked Image")
        axs[2].axis('off')

        if augmented_images is not None:
            axs[3].imshow(augmented_images[i].clamp(0, 1).permute(1, 2, 0).numpy())
            axs[3].set_title("Augmented Image")
            axs[3].axis('off')

        vutils.save_image(masks[i], save_dir.joinpath(f"mask_{indexes[i]}.jpg"), normalize=True, value_range=(0, 1))
        vutils.save_image(masked_images[i], save_dir.joinpath(f"masked_image_{indexes[i]}.jpg"), normalize=True, value_range=(0, 1))
        vutils.save_image(augmented_masks[i], save_dir.joinpath(f"augmented_mask_{indexes[i]}.jpg"), normalize=True, value_range=(0, 1))
        vutils.save_image(augmented_images[i], save_dir.joinpath(f"augmented_image_{indexes[i]}.jpg"), normalize=True, value_range=(0, 1))

        plt.tight_layout()
        plt.show()

def run_center_of_mask(data_dir, img_sz=640, batch_size=64):
    from torch.utils.data import DataLoader
    from yolov8_clr_train import ImageForlderLoader
    from yolov8_utils import get_center_of_mask
    # 实例化数据集
    dataset = ImageForlderLoader(target_dir=data_dir, img_sz=img_sz)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 取一个batch
    images, masks = next(iter(dataloader))  # shapes: [B, 3, H, W], [B, 1, H, W]
    masked_images = images * masks.expand(-1, 3, -1, -1)
    # 获取masks的质心
    centers = get_center_of_mask(masks)
    # //todo：将质心标记在masked_images上面
    # 显示图像
    num = min(batch_size, 16)
    for i in range(num):
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        axs[0].imshow(images[i].clamp(0, 1).permute(1, 2, 0).numpy())
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        axs[1].imshow(masks[i].clamp(0, 1).permute(1, 2, 0).numpy(), cmap='gray')
        axs[1].set_title("Generated Mask")
        axs[1].axis('off')

        axs[2].imshow(masked_images[i].clamp(0, 1).permute(1, 2, 0).numpy())
        cy, cx = centers[i]
        axs[2].plot(cx, cy, 'rx', markersize=10, markeredgewidth=2)  # 红色叉号
        axs[2].set_title("Masked Image")
        axs[2].axis('off')

        plt.tight_layout()
        plt.show()


def run_calc_sigma_normalized(ft_size, ct_size, edge_v):
    from yolov8_utils import calc_sigma_normalized
    sigma = calc_sigma_normalized(ft_size, ct_size, edge_v)
    print(sigma)
    return sigma


def roboflow_download():
    from roboflow import Roboflow
    rf = Roboflow(api_key="rcW2Z5nQdGkHxCQVvfEk")
    project = rf.workspace("mintra").project("durian-leaf2")
    version = project.version(4)
    dataset = version.download("yolov8")


def run_poly2bbox(image_in, label_in, out_dir):
    from yolov8_utils import poly2bbox
    poly2bbox(image_in, label_in, out_dir)


def run_graft_detect_header(from_dir, to_dir, hybrid_dir):
    import os
    from yolov8_utils import graft_detect_header
    from yolov8_model_tools import load_model, get_backbone
    from attention import SelfDetectionTrainer
    from_weight = Path(f"runs/detect/train_{from_dir}/weights/best.pt")
    to_weight = Path(f"runs/detect/train_{to_dir}/weights/best.pt")
    model_path = Path(make_abs_path("models/yolov8m.yaml"))
    from_model = load_model(model_path=model_path, task="detect", modify_model=True)
    from_model.load(from_weight)
    to_model = load_model(model_path=model_path, task="detect", modify_model=True)
    to_model.load(to_weight)
    hybrid_pt = Path(f"runs/detect/train_{hybrid_dir}/weights/best.pt")
    hybrid_pt.parent.mkdir(parents=True, exist_ok=True)
    hybrid_model, _ = graft_detect_header(from_model, to_model, hybrid_pt)
    # TODO: 继续训练
    kwargs = { 
        "device": "cuda",
        "epochs": 150,    # 稍延长
        "imgsz": 640,
        "batch": 64, 
        "workers": min(os.cpu_count() // 2, 8),
        "save": True, "exist_ok": True, "cache": "disk",
        "freeze": 0, "seed": 42,
        "optimizer": "AdamW",
        "lr0": 0.0032, "lrf": 0.05,       # 稍慢退火
        "momentum": 0.937, "weight_decay": 0.00020,
        "cos_lr": True, "warmup_epochs": 5,
        "mosaic": 0.05, "mixup": 0.00, "copy_paste": 0.00,
        "degrees": 0.0, "shear": 0.0, "translate": 0.05, "scale": 0.50,
        "hsv_h": 0.015, "hsv_s": 0.30, "hsv_v": 0.28,
        "close_mosaic": int(0.40 * 150),
        "label_smoothing": 0.0,
        "cls": 0.85, "box": 1.25, "dfl": 1.05,
        "iou": 0.62,   # <— 强烈建议：验证/推理放宽到 0.62 提召回
    }
    result = hybrid_model.train(
        task="detect", 
        name=f"train_{hybrid_dir}",
        data=Path(make_abs_path("datasets/detect/detect_durian_leaf.yaml")),
        trainer=SelfDetectionTrainer,
        pretrained=False,
        resume=False,
        **kwargs
    )
    # 计算predict & val结果
    import sys
    import subprocess
    subprocess.run([
        sys.executable,  # 当前解释器路径
        "yolov8_predict_detect.py",
        "--dir_suffix", hybrid_dir,
    ], shell=True, check=True)
    

if __name__ == '__main__':
    import time
    from datetime import datetime
    print(f"[测试]开始测试！时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    t0 = time.time()
    # # 测试图片kmean算法的最优k
    # image_path = Path(make_abs_path("runs")).joinpath("heatmap").joinpath("AlgalLeafSpot042.jpg")
    # run_find_best_k(image_path)
    # # 测试高斯权重图
    # sigma = run_calc_sigma_normalized(20, 12, 0.5)
    # run_gaussian_weight_map(20, 20, sigma=sigma)
    # run_center_weight_map(20, 20, 2, 3)
    # 测试生成mask
    # image_dir = make_abs_path("datasets/classify/train")
    # image_dir = make_abs_path("runs/test")
    # test_loader_and_mask(data_dir=image_dir, img_sz=640, batch_size=4, show_aug=True)
    # 测试mask质心函数
    # image_dir = make_abs_path("datasets/classify/train")
    # run_center_of_mask(data_dir=image_dir, img_sz=640, batch_size=4)
    # 下载数据集合
    # roboflow_download()
    # 多边形标签转换
    # image_in = make_abs_path(fr"D:\Durian_YOLO\datasets\detect\train\images")
    # label_in = make_abs_path(fr"D:\Durian_YOLO\datasets\detect\train\old_labels")
    # out_dir = make_abs_path(fr"D:\Durian_YOLO\datasets\detect\train\labels")
    # run_poly2bbox(image_in, label_in, out_dir)
    # 更换头
    # run_graft_detect_header("baseline", "ft-2a6-r", "hybrid")
    # 输出用时
    elapsed = time.time() - t0
    print(f"[测试]完成测试！用时：{elapsed:.3f} 秒")
