import os
import sys
from pathlib import Path
from copy import deepcopy

import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

# =========================================================
# Path injection
# =========================================================
# 说明：
# 1. CURRENT_DIR：当前脚本所在目录
# 2. PROJECT_ROOT：项目根目录（通常是当前脚本上一级）
# 3. SAM2_ROOT：本地 SAM2 仓库根目录，请按你的实际路径修改
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SAM2_ROOT = r"E:\sam2"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if os.path.isdir(SAM2_ROOT) and SAM2_ROOT not in sys.path:
    sys.path.insert(0, SAM2_ROOT)

from sam2.build_sam import build_sam2  # type: ignore
from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

import warnings
warnings.filterwarnings(action="ignore", category=UserWarning)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# =========================================================
# 全局配置区（所有参数都在这里集中管理）
# =========================================================
# 使用建议：
# 1. 路径相关参数：先改 PATHS
# 2. 如果前景叶片经常被截断：优先调 BOX / MASK / GRABCUT
# 3. 如果背景带进来太多：优先调 SCORE / FILTER / GRABCUT
# 4. 先只改少量参数，不要一次改太多
# =========================================================
CONFIG = {
    # -----------------------------------------------------
    # 路径与运行配置
    # -----------------------------------------------------
    "PATHS": {
        # 输入图像文件夹
        "input_root": [r"./data/unlabeled_train/images"],

        # 输出 mask 文件夹
        "output_root": r"./data/unlabeled_train/foreground_masks",

        # SAM2 配置文件路径
        "model_cfg": os.path.join(SAM2_ROOT, r"sam2\configs\sam2.1\sam2.1_hiera_s.yaml"),

        # SAM2 权重路径
        "checkpoint": os.path.join(SAM2_ROOT, r"checkpoints\sam2.1_hiera_small.pt"),
    },

    "RUNTIME": {
        # 是否覆盖已有输出
        "overwrite": False,

        # 是否保存预览图
        "save_preview": True,

        # 是否保存初始 mask（SAM2 输出，未经过 GrabCut）
        "save_init_mask": False,

        # 设备：自动根据 torch 判断
        "device": "cuda" if torch.cuda.is_available() else "cpu",

        # 是否在 CUDA 上启用 autocast
        "use_autocast_when_cuda": True,
    },

    # -----------------------------------------------------
    # 基础 mask 处理参数
    # -----------------------------------------------------
    "MASK": {
        # 小连通域最小面积，小于该值会被删除
        "min_region_area": 512,

        # refine 时最多保留几个连通域
        # v2 路线建议默认 1 或 2
        "refine_max_components": 2,

        # 平滑（闭运算）核大小
        "smooth_kernel_size": 7,

        # 开运算核大小（当前默认未强制使用，可按需接入）
        "open_kernel_size": 5,

        # 膨胀核大小
        "dilate_kernel_size": 9,

        # 膨胀迭代次数
        "dilate_iterations": 1,

        # 腐蚀核大小
        "erode_kernel_size": 9,

        # 腐蚀迭代次数
        "erode_iterations": 1,
    },

    # -----------------------------------------------------
    # 候选区域打分参数
    # 说明：
    # 这一部分直接影响“选择哪个 SAM2 mask”
    # 如果中心偏置太强，会更容易只保留中间叶片
    # -----------------------------------------------------
    "SCORE": {
        # 各项权重
        "w_area": 0.26,
        "w_center": 0.16,
        "w_central_band": 0.18,
        "w_aspect": 0.12,
        "w_solidity": 0.08,
        "w_fill": 0.10,
        "w_sam_score": 0.16,
        "w_edge_penalty": 0.10,
        "w_bbox_cover_penalty": 0.08,

        # 面积归一化参考值，越小越偏好较小区域，越大越偏好较大区域
        "area_norm_ref": 0.25,

        # 面积过小惩罚阈值
        "small_area_threshold": 0.02,
        "small_area_penalty": 2.0,

        # 面积过大惩罚阈值
        "large_area_threshold": 0.62,
        "large_area_penalty": 2.5,

        # bbox 填充率过低惩罚阈值
        "low_fill_threshold": 0.18,
        "low_fill_penalty": 1.2,

        # bbox 覆盖整图比例过大惩罚阈值
        "large_bbox_cover_threshold": 0.78,
        "large_bbox_cover_penalty": 2.0,

        # 边缘接触宽度
        "edge_width": 18,

        # 中心竖向带范围（百分比）
        "central_band_x0_ratio": 0.30,
        "central_band_x1_ratio": 0.70,

        # aspect ratio 评分区间
        "aspect_low0": 1.10,
        "aspect_low1": 1.35,
        "aspect_high0": 5.5,
        "aspect_high1": 6.8,
    },

    # -----------------------------------------------------
    # 粗 box 提议参数
    # 说明：
    # 这里控制“先用 GrabCut 粗略找目标，再给 SAM2 提示框”
    # -----------------------------------------------------
    "BOX": {
        # 多组候选矩形（相对图像宽高的比例）
        # 当前保持 v2 风格：仍然是偏中心的若干候选框
        "rects": [
            [0.18, 0.08, 0.82, 0.92],
            [0.24, 0.10, 0.76, 0.90],
            [0.12, 0.15, 0.88, 0.85],
            [0.20, 0.18, 0.80, 0.88],
        ],

        # 默认中心 fallback box
        "center_box": [0.16, 0.06, 0.84, 0.94],

        # 对粗略 box 做扩张时的比例
        "expand_scale_x_1": 0.12,
        "expand_scale_y_1": 0.18,
        "expand_scale_x_2": 0.22,
        "expand_scale_y_2": 0.30,

        # 粗略 GrabCut 的迭代次数
        "coarse_grabcut_iter_count": 5,
    },

    # -----------------------------------------------------
    # SAM2 选择参数
    # -----------------------------------------------------
    "SAM2": {
        # SAM2 是否输出多 mask
        "multimask_output": True,
    },

    # -----------------------------------------------------
    # GrabCut 精修参数
    # 说明：
    # 这里的目标不是完美背景分离，而是：
    # 1. 保护初始叶片区域
    # 2. 适度补全边缘
    # 3. 不让 GrabCut 把叶片切掉
    # -----------------------------------------------------
    "GRABCUT": {
        # ROI 扩张比例
        "roi_pad_x": 0.20,
        "roi_pad_y": 0.26,

        # 构建 sure foreground 时的腐蚀核
        "sure_fg_erode_kernel": 7,

        # 构建 probable foreground 时的膨胀核
        "prob_fg_dilate_kernel": 21,

        # ROI 外边界作为 sure background 的边框比例
        "bg_border_ratio_x": 0.06,
        "bg_border_ratio_y": 0.06,

        # GrabCut 迭代次数
        "grabcut_iter_count": 5,

        # 与初始 seed 接近的连通域保留时，seed 膨胀核大小
        "touch_seed_kernel": 23,

        # 最终 refined 与原始 init mask 做并集保护
        "protect_init_union": True,

        # 对 init mask 先做一次轻度膨胀保护召回
        "pre_dilate_before_grabcut": True,
        "pre_dilate_kernel": 7,
    },

    # -----------------------------------------------------
    # 预览图配置
    # -----------------------------------------------------
    "PREVIEW": {
        # 绿色：最终 mask 填充
        "fill_color": [0, 255, 0],

        # 蓝橙色：初始 mask 轮廓
        "init_contour_color": [0, 128, 255],

        # 紫色：最终 mask 轮廓
        "final_contour_color": [255, 0, 255],

        # 候选框颜色列表
        "box_colors": [
            [255, 0, 0],
            [255, 128, 0],
            [255, 0, 255],
            [0, 128, 255],
        ],

        # 填充透明度
        "fill_alpha": 0.40,

        # 轮廓线粗细
        "contour_thickness": 2,

        # 框线粗细
        "box_thickness": 2,
    },
}


# =========================================================
# Basic IO
# =========================================================
from pathlib import Path
from PIL import Image
import numpy as np

# =========================================================
# Basic IO
# =========================================================
from PIL import Image, ImageOps
import numpy as np

# =========================================================
# Basic IO
# =========================================================
def load_image_rgb(path, max_size=None, force_vertical=True, save_when_changed=True):
    """
    读取 RGB 图像，并可选地强制转为竖直方向，同时可限制最大尺寸。

    参数：
        path: 图像路径
        force_vertical: 是否强制图片为竖直方向。
                        True 表示如果宽 > 高，则旋转 90 度。
        max_size: 最大边长限制，例如 1024。
                  如果为 None，则不缩放。
                  如果图像宽或高超过 max_size，则按比例缩小。

    返回：
        numpy.ndarray，RGB 格式
    """
    img = Image.open(path)

    # 先根据 EXIF 自动校正方向，避免手机照片方向异常
    img = ImageOps.exif_transpose(img)

    img = img.convert("RGB")

    # 强制竖直方向：宽大于高，就旋转
    if force_vertical:
        w, h = img.size
        if w > h:
            img = img.rotate(90, expand=True)
            # 如果发生旋转，则覆盖保存原图
            if save_when_changed: img.save(path)

    # 可选：限制最大边长
    if max_size is not None:
        w, h = img.size
        max_side = max(w, h)

        if max_side > max_size:
            scale = max_size / max_side
            new_w = int(w * scale)
            new_h = int(h * scale)

            img = img.resize((new_w, new_h), Image.BILINEAR)
            # 如果发生缩放，则覆盖保存原图
            if save_when_changed: img.save(path)

    return np.array(img)


def save_mask_png(mask, save_path):
    """
    保存二值 mask 为 PNG。

    参数：
        mask: 二值 mask，前景=1，背景=0
        save_path: 输出路径
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_u8).save(save_path)


def iter_images(root_dirs):
    """
    遍历 root_dirs 下所有支持的图像文件。

    参数：
        root_dirs: 根目录

    返回：
        逐个 yield 图像路径
    """
    # 如果传入的是单个路径字符串或 Path，就转成列表
    if isinstance(root_dirs, (str, Path)):
        root_dirs = [root_dirs]

    for root_dir in root_dirs:
        root = Path(root_dir)

        if not root.exists():
            print(f"[WARN] root_dir 不存在，跳过: {root}")
            continue
        
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS:
                yield p


# =========================================================
# Basic mask utilities
# =========================================================
def ensure_binary(mask):
    """
    保证输入 mask 为二值格式（0/1）。

    参数：
        mask: 任意数值 mask

    返回：
        二值 uint8 mask
    """
    return (mask > 0).astype(np.uint8)


def fill_holes(mask):
    """
    填充 mask 内部空洞。

    实现方式：
    1. 先将 mask 乘 255 转成 8-bit 图
    2. 从左上角做 flood fill，填满背景
    3. 对 flood fill 结果取反，得到“孔洞区域”
    4. 与原 mask 做并集，完成孔洞填充

    参数：
        mask: 二值 mask

    返回：
        填洞后的二值 mask
    """
    mask = ensure_binary(mask)
    h, w = mask.shape

    flood = (mask * 255).astype(np.uint8).copy()
    floodfill_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, floodfill_mask, (0, 0), 255)

    flood_inv = cv2.bitwise_not(flood)
    filled = (mask * 255) | flood_inv
    return (filled > 127).astype(np.uint8)


def remove_small_regions(mask, min_area):
    """
    删除过小连通域。

    参数：
        mask: 输入二值 mask
        min_area: 最小保留面积

    返回：
        删除小区域后的 mask
    """
    mask = ensure_binary(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 1
    return out


def keep_top_components(mask, max_components, min_area):
    """
    只保留面积最大的若干个连通域。

    参数：
        mask: 输入二值 mask
        max_components: 最多保留几个连通域
        min_area: 小于该面积的区域直接不考虑

    返回：
        筛选后的 mask
    """
    mask = ensure_binary(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    comps = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            comps.append((i, area))

    if len(comps) == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    comps = sorted(comps, key=lambda x: x[1], reverse=True)[:max_components]

    out = np.zeros_like(mask, dtype=np.uint8)
    for idx, _ in comps:
        out[labels == idx] = 1
    return out


def keep_largest_component(mask):
    """
    只保留最大连通域。
    这是一个便捷函数，当前仍保留给局部候选选择使用。
    """
    return keep_top_components(mask, max_components=1, min_area=1)


def smooth_mask(mask, kernel_size):
    """
    对 mask 做闭运算平滑，主要作用：
    1. 连接轻微断裂的边缘
    2. 平滑锯齿轮廓
    3. 对局部小缝隙有一定补偿

    参数：
        mask: 输入二值 mask
        kernel_size: 椭圆核大小

    返回：
        平滑后的 mask
    """
    mask = ensure_binary(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return ensure_binary(mask)


def open_mask(mask, kernel_size):
    """
    对 mask 做开运算。
    当前主流程未强制使用，但保留下来供后续实验时调用。

    作用：
    1. 去除少量孤立噪点
    2. 切断过细的噪声连接

    参数：
        mask: 输入二值 mask
        kernel_size: 椭圆核大小

    返回：
        开运算后的 mask
    """
    mask = ensure_binary(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return ensure_binary(mask)


def dilate_mask(mask, kernel_size, iterations):
    """
    对 mask 做膨胀。

    作用：
    1. 扩张前景区域
    2. 提高召回
    3. 填补小的缺口

    参数：
        mask: 输入二值 mask
        kernel_size: 椭圆核大小
        iterations: 迭代次数

    返回：
        膨胀后的 mask
    """
    mask = ensure_binary(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.dilate(mask, kernel, iterations=iterations)
    return ensure_binary(mask)


def erode_mask(mask, kernel_size, iterations):
    """
    对 mask 做腐蚀。

    作用：
    1. 收缩前景区域
    2. 提取“更可靠的内部核心区域”
    3. 用于 GrabCut 中构建 sure foreground

    参数：
        mask: 输入二值 mask
        kernel_size: 椭圆核大小
        iterations: 迭代次数

    返回：
        腐蚀后的 mask
    """
    mask = ensure_binary(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.erode(mask, kernel, iterations=iterations)
    return ensure_binary(mask)


def keep_components_touching_seed(mask, seed_mask, dilate_kernel, min_area, max_components):
    """
    只保留与 seed 区域重叠或靠近的连通域。

    这是一个非常关键的函数：
    它的作用不是“只保留最大区域”，而是“保留与原始目标相关的区域”。

    参数：
        mask: 待筛选 mask
        seed_mask: 原始 seed 区域（通常是初始 prompted mask）
        dilate_kernel: 先把 seed 适度膨胀，再判断哪些区域与之相接触
        min_area: 小于该面积的区域直接剔除
        max_components: 最多保留多少个接近 seed 的连通域

    返回：
        筛选后的 mask
    """
    mask = ensure_binary(mask)
    seed_mask = ensure_binary(seed_mask)

    dilated_seed = dilate_mask(seed_mask, kernel_size=dilate_kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    kept = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        comp = np.zeros_like(mask, dtype=np.uint8)
        comp[labels == i] = 1

        if (comp & dilated_seed).sum() > 0:
            kept.append((i, area))

    if len(kept) == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    kept = sorted(kept, key=lambda x: x[1], reverse=True)[:max_components]

    out = np.zeros_like(mask, dtype=np.uint8)
    for idx, _ in kept:
        out[labels == idx] = 1
    return out


def refine_mask(mask, cfg):
    """
    保守式 mask 精修函数。

    当前 v2 路线的原则：
    1. 去掉明显碎片
    2. 填补空洞
    3. 平滑轮廓
    4. 保留有限个主连通域（默认 1 或 2）

    参数：
        mask: 输入 mask
        cfg: 全局配置

    返回：
        精修后的 mask
    """
    mcfg = cfg["MASK"]

    mask = ensure_binary(mask)
    mask = remove_small_regions(mask, min_area=mcfg["min_region_area"])
    mask = fill_holes(mask)
    mask = smooth_mask(mask, kernel_size=mcfg["smooth_kernel_size"])
    mask = fill_holes(mask)
    mask = keep_top_components(
        mask,
        max_components=mcfg["refine_max_components"],
        min_area=mcfg["min_region_area"]
    )
    return ensure_binary(mask)


# =========================================================
# Geometry / scoring helpers
# =========================================================
def bbox_from_mask(mask):
    """
    从二值 mask 计算外接矩形 bbox。

    返回：
        [x0, y0, x1, y1]，若 mask 为空则返回 None
    """
    mask = ensure_binary(mask)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def bbox_area(bbox):
    """
    计算 bbox 面积。
    """
    if bbox is None:
        return 0
    x0, y0, x1, y1 = bbox
    return max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1)


def area_ratio(mask, image_shape):
    """
    前景区域占整张图的比例。
    """
    mask = ensure_binary(mask)
    h, w = image_shape[:2]
    return float(mask.sum() / (h * w + 1e-6))


def bbox_fill_ratio(mask):
    """
    mask 在其外接框中的填充率。

    理解：
    - 如果 fill 很低，说明这个 mask 很稀疏、很碎、或形状不合理
    - 如果 fill 较高，通常表示区域更紧凑
    """
    mask = ensure_binary(mask)
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return 0.0

    area = mask.sum()
    barea = bbox_area(bbox) + 1e-6
    return float(area / barea)


def component_stats(mask):
    """
    统计连通域个数与面积列表。
    """
    mask = ensure_binary(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    areas = []
    for i in range(1, num_labels):
        areas.append(int(stats[i, cv2.CC_STAT_AREA]))

    areas = sorted(areas, reverse=True)
    return {
        "count": len(areas),
        "areas": areas
    }


def center_score(mask):
    """
    中心性得分。

    说明：
    当前 v2 路线仍保留轻度中心偏置，因为你的反馈是：
    v3 太宽泛，v2 更接近你想要的目标。
    所以这个得分不会删除，但后续如果仍然“只抓中间叶”，
    可以优先调小它的权重。
    """
    mask = ensure_binary(mask)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0

    h, w = mask.shape
    cx = xs.mean()
    cy = ys.mean()

    dx = abs(cx - w / 2) / (w / 2 + 1e-6)
    dy = abs(cy - h / 2) / (h / 2 + 1e-6)
    dist = np.sqrt(dx * dx + dy * dy)

    return float(1.0 - np.clip(dist, 0.0, 1.0))


def central_band_score(mask, cfg):
    """
    中央竖带覆盖率得分。

    作用：
    - 鼓励 mask 覆盖图像中央纵向区域
    - 在单主叶片场景中，这通常有帮助
    - 但如果多叶片场景总是偏中间，可以调低该项权重
    """
    scfg = cfg["SCORE"]
    mask = ensure_binary(mask)
    h, w = mask.shape

    band_x0 = int(scfg["central_band_x0_ratio"] * w)
    band_x1 = int(scfg["central_band_x1_ratio"] * w)

    band = np.zeros_like(mask, dtype=np.uint8)
    band[:, band_x0:band_x1] = 1

    inter = (mask & band).sum()
    area = mask.sum() + 1e-6
    return float(inter / area)


def aspect_ratio_score(mask, cfg):
    """
    长宽比得分。

    叶片通常呈一定程度的细长形态，
    但为了避免过度刚性，这里不是硬阈值，而是分段评分。
    """
    scfg = cfg["SCORE"]
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return 0.0

    x0, y0, x1, y1 = bbox
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    ratio = max(h, w) / (min(h, w) + 1e-6)

    low0 = scfg["aspect_low0"]
    low1 = scfg["aspect_low1"]
    high0 = scfg["aspect_high0"]
    high1 = scfg["aspect_high1"]

    if ratio < low0:
        return 0.0
    elif ratio < low1:
        return float((ratio - low0) / (low1 - low0) * 0.6)
    elif ratio <= high0:
        return 1.0
    elif ratio <= high1:
        return float(max(0.5, 1.0 - (ratio - high0) / (high1 - high0) * 0.5))
    else:
        return 0.2


def solidity_score(mask):
    """
    凸包实心度得分。

    作用：
    - 评价轮廓是否过于破碎
    - 过于锯齿、细碎的区域通常 solidity 会偏低
    """
    mask = ensure_binary(mask)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0

    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area <= 1:
        return 0.0

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 1:
        return 0.0

    solidity = area / (hull_area + 1e-6)
    return float(np.clip(solidity, 0.0, 1.0))


def edge_touch_ratio(mask, edge_width):
    """
    mask 与图像边缘区域的接触比例。

    解释：
    - 如果一个候选区域大量贴边，可能意味着选中了背景大块区域
    - 当然真实叶片也可能靠边，所以这里只是惩罚项，不是硬筛选
    """
    mask = ensure_binary(mask)
    h, w = mask.shape

    edge = np.zeros_like(mask, dtype=np.uint8)
    edge[:edge_width, :] = 1
    edge[-edge_width:, :] = 1
    edge[:, :edge_width] = 1
    edge[:, -edge_width:] = 1

    inter = (mask & edge).sum()
    area = mask.sum() + 1e-6
    return float(inter / area)


def bbox_cover_ratio(mask, image_shape):
    """
    候选区域外接框占整图比例。

    如果这个值太大，往往意味着当前区域过于宽泛。
    """
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return 1.0

    bbox_a = bbox_area(bbox)
    h, w = image_shape[:2]
    return float(bbox_a / (h * w + 1e-6))


def preprocess_candidate(mask, cfg):
    """
    对候选 mask 做轻量预处理，再进入打分。

    作用：
    1. 删掉太碎的小块
    2. 填洞
    3. 轻度平滑
    """
    mcfg = cfg["MASK"]
    mask = ensure_binary(mask)
    mask = remove_small_regions(mask, min_area=mcfg["min_region_area"])
    mask = fill_holes(mask)
    mask = smooth_mask(mask, kernel_size=5)
    return mask


def score_leaf_candidate(image_rgb, mask, sam_score, cfg):
    """
    对候选 mask 打分。

    当前 v2 路线：
    - 仍保留一定中心偏置
    - 但不希望完全只看中间，因此所有项均由可配置权重控制

    返回：
        score: 候选得分
        cleaned: 预处理后的 mask
    """
    scfg = cfg["SCORE"]
    cleaned = preprocess_candidate(mask, cfg)

    if cleaned.sum() == 0:
        return -1e9, cleaned

    s_area = area_ratio(cleaned, image_rgb.shape)
    s_center = center_score(cleaned)
    s_cband = central_band_score(cleaned, cfg)
    s_aspect = aspect_ratio_score(cleaned, cfg)
    s_solidity = solidity_score(cleaned)
    s_edge = edge_touch_ratio(cleaned, edge_width=scfg["edge_width"])
    s_bbox_cover = bbox_cover_ratio(cleaned, image_rgb.shape)
    s_fill = bbox_fill_ratio(cleaned)

    if sam_score is None:
        sam_score = 0.5

    score = (
        scfg["w_area"] * np.clip(s_area / scfg["area_norm_ref"], 0.0, 1.0) +
        scfg["w_center"] * s_center +
        scfg["w_central_band"] * s_cband +
        scfg["w_aspect"] * s_aspect +
        scfg["w_solidity"] * s_solidity +
        scfg["w_fill"] * s_fill +
        scfg["w_sam_score"] * float(sam_score) -
        scfg["w_edge_penalty"] * s_edge -
        scfg["w_bbox_cover_penalty"] * s_bbox_cover
    )

    if s_area < scfg["small_area_threshold"]:
        score -= scfg["small_area_penalty"]

    if s_area > scfg["large_area_threshold"]:
        score -= scfg["large_area_penalty"]

    if s_fill < scfg["low_fill_threshold"]:
        score -= scfg["low_fill_penalty"]

    if s_bbox_cover > scfg["large_bbox_cover_threshold"]:
        score -= scfg["large_bbox_cover_penalty"]

    return float(score), cleaned


# =========================================================
# Box utilities
# =========================================================
def expand_box(box, image_shape, scale_x, scale_y):
    """
    对给定 box 做扩张。

    参数：
        box: [x0, y0, x1, y1]
        image_shape: 图像尺寸
        scale_x: 横向扩张比例
        scale_y: 纵向扩张比例
    """
    h, w = image_shape[:2]
    x0, y0, x1, y1 = box

    bw = x1 - x0 + 1
    bh = y1 - y0 + 1

    ex = int(scale_x * bw)
    ey = int(scale_y * bh)

    x0 = max(0, int(x0 - ex))
    y0 = max(0, int(y0 - ey))
    x1 = min(w - 1, int(x1 + ex))
    y1 = min(h - 1, int(y1 + ey))

    return np.array([x0, y0, x1, y1], dtype=np.float32)


def make_center_box(image_shape, cfg):
    """
    构造中心 fallback box。
    当粗略 GrabCut 没能提出可靠 box 时，退回这个默认框。
    """
    bcfg = cfg["BOX"]
    h, w = image_shape[:2]
    x0r, y0r, x1r, y1r = bcfg["center_box"]
    x0 = int(x0r * w)
    y0 = int(y0r * h)
    x1 = int(x1r * w)
    y1 = int(y1r * h)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


# =========================================================
# Coarse box proposal with GrabCut
# =========================================================
def get_grabcut_mask(image_rgb, rect, iter_count):
    """
    使用矩形初始化的 GrabCut 获取粗略前景。

    注意：
    这里不是最终分割结果，只是为了提一个相对合理的 box 给 SAM2。
    """
    h, w = image_rgb.shape[:2]
    x0, y0, x1, y1 = rect

    x = max(0, int(x0))
    y = max(0, int(y0))
    rw = max(1, int(x1 - x0))
    rh = max(1, int(y1 - y0))

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        image_rgb,
        mask,
        (x, y, rw, rh),
        bgd_model,
        fgd_model,
        iterCount=iter_count,
        mode=cv2.GC_INIT_WITH_RECT
    )

    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    return fg


def component_score(mask, image_shape, cfg):
    """
    对粗略 GrabCut 提出的前景连通域打分。
    这个分数用于决定“哪个 coarse component 更像主目标”。
    """
    mask = ensure_binary(mask)
    h, w = image_shape[:2]
    area = mask.sum()
    if area == 0:
        return -1e9

    scfg = cfg["SCORE"]

    ar = area / (h * w + 1e-6)
    cscore = center_score(mask)
    bscore = central_band_score(mask, cfg)
    ascore = aspect_ratio_score(mask, cfg)
    sscore = solidity_score(mask)
    fill = bbox_fill_ratio(mask)
    edge = edge_touch_ratio(mask, edge_width=scfg["edge_width"])
    bbox_cov = bbox_cover_ratio(mask, image_shape)

    score = (
        scfg["w_area"] * np.clip(ar / scfg["area_norm_ref"], 0.0, 1.0) +
        scfg["w_center"] * cscore +
        scfg["w_central_band"] * bscore +
        scfg["w_aspect"] * ascore +
        scfg["w_solidity"] * sscore +
        scfg["w_fill"] * fill -
        scfg["w_edge_penalty"] * edge -
        scfg["w_bbox_cover_penalty"] * bbox_cov
    )

    if ar < scfg["small_area_threshold"]:
        score -= scfg["small_area_penalty"]

    if ar > 0.65:
        score -= 2.0

    if fill < 0.15:
        score -= 1.5

    if bbox_cov > 0.72:
        score -= 2.0

    return float(score)


def propose_box_from_grabcut(image_rgb, cfg):
    """
    通过多组矩形初始化的 GrabCut，提出一个最优粗 box。

    步骤：
    1. 用多组中心偏置矩形做矩形版 GrabCut
    2. 得到粗前景 mask
    3. 提取其中较大连通域
    4. 给每个连通域打分
    5. 选得分最高的 bbox 作为 base_box

    注意：
    这一步仍然是 v2 风格，因此默认倾向于选择“主目标”。
    """
    bcfg = cfg["BOX"]
    h, w = image_rgb.shape[:2]

    rects = []
    for x0r, y0r, x1r, y1r in bcfg["rects"]:
        rects.append([x0r * w, y0r * h, x1r * w, y1r * h])

    best_score = -1e9
    best_box = None

    for rect in rects:
        coarse = get_grabcut_mask(
            image_rgb,
            rect,
            iter_count=bcfg["coarse_grabcut_iter_count"]
        )
        coarse = remove_small_regions(coarse, min_area=1024)
        coarse = fill_holes(coarse)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(coarse, connectivity=8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 1024:
                continue

            comp = np.zeros_like(coarse, dtype=np.uint8)
            comp[labels == i] = 1

            score = component_score(comp, image_rgb.shape, cfg)
            if score > best_score:
                box = bbox_from_mask(comp)
                if box is not None:
                    best_score = score
                    best_box = box

    if best_box is None:
        best_box = make_center_box(image_rgb.shape, cfg)

    return best_box


# =========================================================
# SAM2 predictor
# =========================================================
def build_predictor(model_cfg, checkpoint, device="cuda"):
    """
    构建 SAM2 predictor。
    """
    sam2_model = build_sam2(
        config_file=model_cfg,
        ckpt_path=checkpoint,
        device=device
    )
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor


def select_best_mask_from_single_box(image_rgb, masks, scores, cfg):
    """
    从单个 box 的多候选输出中选一个最优 mask。

    说明：
    SAM2 在同一个 box 下通常会输出多个候选 mask。
    这里不是直接取分数最高的 SAM2 原始分数，
    而是结合我们自己的 leaf_candidate_score 做二次筛选。
    """
    best_score = -1e9
    best_mask = None

    for mask, sam_score in zip(masks, scores):
        mask = ensure_binary(mask)

        score, cleaned = score_leaf_candidate(
            image_rgb=image_rgb,
            mask=mask,
            sam_score=float(sam_score),
            cfg=cfg
        )

        if cleaned.sum() == 0:
            continue

        if score > best_score:
            best_score = score
            best_mask = cleaned

    if best_mask is None:
        h, w = image_rgb.shape[:2]
        best_mask = np.zeros((h, w), dtype=np.uint8)
        best_score = -1e9

    best_mask = refine_mask(best_mask, cfg)
    return best_mask, best_score


def predict_mask_with_box(predictor, image_rgb, base_box, cfg):
    """
    对一个 base_box 生成多组 box 提示，然后从中选最优初始 mask。

    步骤：
    1. 使用 tight box
    2. 使用适度扩张 box
    3. 使用更大扩张 box
    4. 使用中心 fallback box
    5. 分别调用 SAM2
    6. 每个 box 内部选一个最优 mask
    7. 跨 box 再选一个最终最佳 mask

    返回：
        init_mask: SAM2 初始 mask
        candidate_boxes: 所有参与过预测的 box
    """
    bcfg = cfg["BOX"]
    predictor.set_image(image_rgb)

    box_tight = base_box
    box_expand = expand_box(
        base_box,
        image_rgb.shape,
        scale_x=bcfg["expand_scale_x_1"],
        scale_y=bcfg["expand_scale_y_1"]
    )
    box_expand_more = expand_box(
        base_box,
        image_rgb.shape,
        scale_x=bcfg["expand_scale_x_2"],
        scale_y=bcfg["expand_scale_y_2"]
    )
    box_center = make_center_box(image_rgb.shape, cfg)

    candidate_boxes = [
        box_tight,
        box_expand,
        box_expand_more,
        box_center,
    ]

    box_results = []

    for idx, box in enumerate(candidate_boxes):
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=cfg["SAM2"]["multimask_output"],
        )

        best_mask, best_score = select_best_mask_from_single_box(
            image_rgb=image_rgb,
            masks=masks,
            scores=scores,
            cfg=cfg
        )

        box_results.append({
            "box_index": idx,
            "box": box,
            "mask": best_mask,
            "score": best_score,
        })

    # 中心 fallback 的优先策略：
    # 如果 center box 的结果和全局最好结果差距很小，
    # 则优先使用它，以保证 v2 路线的稳定性。
    center_result = box_results[3]
    best_overall = max(box_results, key=lambda x: x["score"])

    if center_result["score"] >= best_overall["score"] - 0.20:
        chosen = center_result
    else:
        chosen = best_overall

    # 对初始 mask 做轻度保护性后处理：
    # 1. 精修
    # 2. 适度膨胀
    # 3. 填洞
    init_mask = refine_mask(chosen["mask"], cfg)

    mcfg = cfg["MASK"]
    init_mask = dilate_mask(
        init_mask,
        kernel_size=mcfg["dilate_kernel_size"],
        iterations=mcfg["dilate_iterations"]
    )
    init_mask = fill_holes(init_mask)
    init_mask = refine_mask(init_mask, cfg)

    return init_mask, candidate_boxes


# =========================================================
# GrabCut refine based on prompted mask
# =========================================================
def build_grabcut_init_mask(roi_shape, local_init_mask, cfg):
    """
    构建 GrabCut 初始化标签图。

    标签含义：
    - cv2.GC_BGD: sure background
    - cv2.GC_PR_BGD: probable background
    - cv2.GC_PR_FGD: probable foreground
    - cv2.GC_FGD: sure foreground

    当前策略：
    1. sure foreground：init mask 轻度腐蚀后的核心区域
    2. probable foreground：init mask 轻度膨胀后的区域
    3. sure background：ROI 外边缘一圈
    4. 其余区域：probable background

    这样做的目的：
    - 尽量保护原始前景
    - 允许 GrabCut 在边缘处做一定补全
    - 避免背景约束过强把叶片削掉
    """
    gcfg = cfg["GRABCUT"]
    h, w = roi_shape[:2]
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    init_mask = ensure_binary(local_init_mask)

    sure_fg = erode_mask(
        init_mask,
        kernel_size=gcfg["sure_fg_erode_kernel"],
        iterations=1
    )
    if sure_fg.sum() == 0:
        sure_fg = init_mask.copy()

    prob_fg = dilate_mask(
        init_mask,
        kernel_size=gcfg["prob_fg_dilate_kernel"],
        iterations=1
    )

    border = np.zeros((h, w), dtype=np.uint8)
    bw = max(6, int(gcfg["bg_border_ratio_x"] * w))
    bh = max(6, int(gcfg["bg_border_ratio_y"] * h))
    border[:bh, :] = 1
    border[-bh:, :] = 1
    border[:, :bw] = 1
    border[:, -bw:] = 1

    gc_mask[border > 0] = cv2.GC_BGD
    gc_mask[prob_fg > 0] = cv2.GC_PR_FGD
    gc_mask[sure_fg > 0] = cv2.GC_FGD

    return gc_mask


def extract_local_roi(image_rgb, init_mask, cfg):
    """
    根据 init_mask 的 bbox，裁出局部 ROI 以执行 GrabCut。

    作用：
    1. 限制 GrabCut 工作范围
    2. 减少它跳到远处背景或其他区域
    3. 提高局部边界精修的稳定性
    """
    gcfg = cfg["GRABCUT"]

    bbox = bbox_from_mask(init_mask)
    if bbox is None:
        return None

    x0, y0, x1, y1 = bbox.astype(int)
    h, w = image_rgb.shape[:2]

    bw = x1 - x0 + 1
    bh = y1 - y0 + 1

    ex = int(gcfg["roi_pad_x"] * bw)
    ey = int(gcfg["roi_pad_y"] * bh)

    rx0 = max(0, x0 - ex)
    ry0 = max(0, y0 - ey)
    rx1 = min(w - 1, x1 + ex)
    ry1 = min(h - 1, y1 + ey)

    roi_img = image_rgb[ry0:ry1 + 1, rx0:rx1 + 1].copy()
    roi_mask = init_mask[ry0:ry1 + 1, rx0:rx1 + 1].copy()

    return roi_img, roi_mask, (rx0, ry0, rx1, ry1)


def refine_prompted_mask_with_grabcut(image_rgb, init_mask, cfg):
    """
    用局部 GrabCut 精修 prompted mask。

    这是主精修函数，逻辑如下：

    步骤1：先对 init_mask 做一次保守式 refine
    步骤2：必要时先轻度膨胀，保护召回
    步骤3：提取局部 ROI
    步骤4：构造 GrabCut 初始标签图
    步骤5：执行 GrabCut
    步骤6：筛掉与 seed 不相干的区域
    步骤7：与原始 seed 做并集，避免叶片被切掉
    步骤8：回贴到整图
    步骤9：再做一次全图级保守 refine

    核心理念：
    - GrabCut 只做精修，不负责重新定义目标
    - 最终始终优先保护 init_mask 的主体信息
    """
    gcfg = cfg["GRABCUT"]
    mcfg = cfg["MASK"]

    init_mask = refine_mask(init_mask, cfg)

    if gcfg["pre_dilate_before_grabcut"]:
        init_mask = dilate_mask(
            init_mask,
            kernel_size=gcfg["pre_dilate_kernel"],
            iterations=1
        )
        init_mask = fill_holes(init_mask)

    if init_mask.sum() == 0:
        return init_mask

    extracted = extract_local_roi(image_rgb, init_mask, cfg)
    if extracted is None:
        return init_mask

    roi_img, roi_init_mask, (rx0, ry0, rx1, ry1) = extracted

    gc_mask = build_grabcut_init_mask(roi_img.shape, roi_init_mask, cfg)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            roi_img,
            gc_mask,
            None,
            bgd_model,
            fgd_model,
            iterCount=gcfg["grabcut_iter_count"],
            mode=cv2.GC_INIT_WITH_MASK
        )
    except cv2.error:
        # 如果 GrabCut 失败，直接返回初始 mask
        return init_mask

    roi_fg = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        1, 0
    ).astype(np.uint8)

    # 仅保留与原始 seed 相连或相近的区域
    roi_fg = keep_components_touching_seed(
        roi_fg,
        roi_init_mask,
        dilate_kernel=gcfg["touch_seed_kernel"],
        min_area=max(256, mcfg["min_region_area"] // 2),
        max_components=mcfg["refine_max_components"]
    )

    # 保护性并集：避免 GrabCut 切掉原始叶片区域
    if gcfg["protect_init_union"]:
        roi_fg = ensure_binary(roi_fg | roi_init_mask)

    roi_fg = fill_holes(roi_fg)
    roi_fg = smooth_mask(roi_fg, kernel_size=mcfg["smooth_kernel_size"])
    roi_fg = refine_mask(roi_fg, cfg)

    if roi_fg.sum() == 0:
        roi_fg = roi_init_mask.copy()

    # 回贴到整图
    refined = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    refined[ry0:ry1 + 1, rx0:rx1 + 1] = roi_fg

    # 再次保护 init_mask 主体
    if gcfg["protect_init_union"]:
        refined = ensure_binary(refined | init_mask)

    refined = refine_mask(refined, cfg)

    return refined


# =========================================================
# Preview
# =========================================================
def draw_mask_contour(canvas, mask, color, thickness=2):
    """
    在画布上绘制 mask 的外轮廓。
    """
    mask = ensure_binary(mask)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, cnts, -1, color, thickness)


def make_preview(image_rgb, init_mask, refined_mask, candidate_boxes, cfg):
    """
    生成预览图。

    可视化内容：
    1. 最终 mask 的填充区域
    2. 初始 mask 轮廓
    3. 最终 mask 轮廓
    4. 候选 box

    方便你快速对比：
    - SAM2 初始结果
    - GrabCut 精修结果
    - 当前 box 提示是否合理
    """
    pcfg = cfg["PREVIEW"]
    overlay = image_rgb.copy()

    fill_color = np.array(pcfg["fill_color"], dtype=np.uint8)
    alpha = pcfg["fill_alpha"]

    overlay[refined_mask > 0] = (
        (1.0 - alpha) * overlay[refined_mask > 0] + alpha * fill_color
    ).astype(np.uint8)

    draw_mask_contour(
        overlay,
        init_mask,
        tuple(pcfg["init_contour_color"]),
        pcfg["contour_thickness"]
    )
    draw_mask_contour(
        overlay,
        refined_mask,
        tuple(pcfg["final_contour_color"]),
        pcfg["contour_thickness"]
    )

    colors = pcfg["box_colors"]
    for i, b in enumerate(candidate_boxes):
        x0, y0, x1, y1 = b.astype(int)
        color = tuple(colors[i % len(colors)])
        cv2.rectangle(
            overlay,
            (x0, y0),
            (x1, y1),
            color,
            pcfg["box_thickness"]
        )

    return overlay


# =========================================================
# Main
# =========================================================
def main():
    """
    主程序入口。

    执行流程：
    1. 读取配置
    2. 初始化 SAM2 predictor
    3. 遍历所有输入图像
    4. 先用粗 GrabCut 提 box
    5. 用 SAM2 根据 box 得到初始 mask
    6. 用 GrabCut 对初始 mask 做局部精修
    7. 保存最终 mask 与预览图
    """
    cfg = deepcopy(CONFIG)

    pcfg = cfg["PATHS"]
    rcfg = cfg["RUNTIME"]

    input_roots = [Path(path) for path in pcfg["input_root"]]
    output_root = Path(pcfg["output_root"])
    model_cfg = pcfg["model_cfg"]
    checkpoint = pcfg["checkpoint"]

    device = rcfg["device"]
    overwrite = rcfg["overwrite"]
    save_preview = rcfg["save_preview"]
    save_init_mask = rcfg["save_init_mask"]

    print("Building SAM2 predictor...")
    print("Input root :", input_roots)
    print("Output root:", output_root)
    print("Model cfg  :", model_cfg)
    print("Checkpoint :", checkpoint)
    print("Device     :", device)

    output_root.mkdir(parents=True, exist_ok=True)
    preview_root = output_root / "_preview"
    init_root = output_root / "_init_mask"

    if save_preview:
        preview_root.mkdir(parents=True, exist_ok=True)
    if save_init_mask:
        init_root.mkdir(parents=True, exist_ok=True)

    predictor = build_predictor(model_cfg, checkpoint, device=device)

    image_paths = list(iter_images(input_roots))
    print(f"Found {len(image_paths)} images.")

    use_autocast = (device == "cuda" and rcfg["use_autocast_when_cuda"])

    for img_path in tqdm(image_paths, desc="Generating prompted+GrabCut SAM2 masks", ncols=168):
        # rel = img_path.relative_to(input_roots)
        # out_mask_path = output_root / rel.with_suffix(".png")
        out_mask_path = output_root / (img_path.stem + ".png")

        if out_mask_path.exists() and not overwrite:
            continue

        image_rgb = load_image_rgb(img_path, max_size=640)

        # -------------------------------------------------
        # Step 1: 用粗略 GrabCut 提一个 base box
        # -------------------------------------------------
        base_box = propose_box_from_grabcut(image_rgb, cfg)

        # -------------------------------------------------
        # Step 2: 用 SAM2 + 多级 box 提示生成初始 mask
        # -------------------------------------------------
        with torch.inference_mode():
            if use_autocast:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    init_mask, candidate_boxes = predict_mask_with_box(
                        predictor, image_rgb, base_box, cfg
                    )
            else:
                init_mask, candidate_boxes = predict_mask_with_box(
                    predictor, image_rgb, base_box, cfg
                )

        init_mask = ensure_binary(init_mask)

        # -------------------------------------------------
        # Step 3: 对初始 mask 做局部 GrabCut 精修
        # -------------------------------------------------
        refined_mask = refine_prompted_mask_with_grabcut(image_rgb, init_mask, cfg)
        refined_mask = ensure_binary(refined_mask)

        # -------------------------------------------------
        # Step 4: 保存结果
        # -------------------------------------------------
        save_mask_png(refined_mask, out_mask_path)

        if save_init_mask:
            # init_mask_path = init_root / rel.with_suffix(".png")
            init_mask_path = init_root / (img_path.stem + ".png")
            save_mask_png(init_mask, init_mask_path)

        if save_preview:
            preview = make_preview(
                image_rgb=image_rgb,
                init_mask=init_mask,
                refined_mask=refined_mask,
                candidate_boxes=candidate_boxes,
                cfg=cfg
            )
            # preview_path = preview_root / rel.with_suffix(".jpg").name
            preview_path = preview_root / (img_path.stem + ".jpg")
            Image.fromarray(preview).save(preview_path)

    print("Done. Prompted+GrabCut masks saved to:", output_root)


if __name__ == "__main__":
    main()