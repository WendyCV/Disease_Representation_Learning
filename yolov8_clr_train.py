import math
import argparse
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.benchmark = False       # 避免反复尝试“最好计划”反而选到不支持路线
torch.backends.cudnn.deterministic = False   # 不强行确定性
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import os
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'

import cv2
from sklearn.cluster import KMeans
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint 
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as VF
from torchvision.transforms import v2
from PIL import Image
from pathlib import Path
import tqdm
import numpy as np

from yolov8_utils import make_abs_path, release_memory
from yolov8_utils import center_coords_of_masks, foreground_attended_loss
from yolov8_utils import dct_saliency, percentile_threshold
from yolov8_utils import ResizeAndPadToSquare
from yolov8_model_tools import load_model, get_backbone, get_attention
from yolov8_model_tools import MultiScaleFeatureExtractor, MultiScaleFeatureProjector, MultiScaleFeatureAttention

# ====================================
# DATASET AND DATA LOADING
# ====================================
class ImageForlderLoader(Dataset):
    def __init__(self, target_dir: str, img_sz=640, edge_mask=True, transform=None):
        self.paths = self.load_forlder(target_dir)
        # scale_factor = max(1.0, min(1.3, scale_factor))
        # 在数据增强已经包括放大再裁剪的能力（放大再裁剪，精度更高/edge_mask会全局质量更好）
        # self.transform = transform if transform else v2.Compose([
        #     v2.Resize(size=int(img_sz * scale_factor)),   # 先放大 X scale_factor
        #     v2.CenterCrop(size=img_sz),                   # 再居中裁剪为目标尺寸
        #     v2.ToImage(),
        #     v2.ConvertImageDtype(dtype=torch.float32),
        # ])
        # 以下策略是最稳健策略
        self.transform = transform if transform else v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            # v2.Resize(size=int(img_sz)),
            ResizeAndPadToSquare(long_size=int(img_sz)),
        ])
        self.edge_mask = edge_mask
        # 缓存数据，每次计算mask很耗时
        self.caches = {}
    
    def load_forlder(self, dir_path):
        target_path = Path(dir_path)
        # 获取当前目录数据
        base = list(sorted(target_path.glob("*.jpg")))
        # 递归子目录数据
        for sub_dir_path in [target_path.joinpath(d) 
                             for d in os.listdir(dir_path) 
                             if osp.isdir(target_path.joinpath(d))]:
            base.extend(self.load_forlder(sub_dir_path))
        # 返回结果
        return base

    def __len__(self) -> int:
        return len(self.paths)
    
    def _apply_saliency_v1_(self, img_np, thresh=(10, 20)):
        # -------- HSV 阈值掩码 --------
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        s_thresh = thresh[0]
        v_thresh = thresh[1]
        mask_dynamic = ((s > s_thresh) & (v > v_thresh)).astype(np.uint8)
        # 返回0/1的mask
        return mask_dynamic
    
    def _apply_saliency_v2_(self, img_np, thresh=(0.05, 0.30)):
        # -------- 🌞 光照均衡：CLAHE 应用于 L 通道 --------
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab = cv2.merge((l_clahe, a, b))

        # -------- Lab 阈值掩码 --------
        L, a, b = cv2.split(lab)
        # L 通道动态范围限制（避免过暗或过亮）
        L_min = percentile_threshold(L, 5)
        L_max = percentile_threshold(L, 95)
        # a 通道动态阈值（绿色一般在a通道低值区）
        a_min = percentile_threshold(a, 1)
        a_max = percentile_threshold(a, 85)
        # b 通道动态阈值（叶子颜色偏黄，b通道有一定范围）
        b_min = percentile_threshold(b, 10)
        b_max = percentile_threshold(b, 95)

        # 构造掩码，注意a通道叶子通常a较低（偏绿），b通道中等偏高（偏黄）
        mask_dynamic = (
            (L >= L_min) & (L <= L_max) &
            (a >= a_min) & (a <= a_max) &
            (b >= b_min) & (b <= b_max)
        ).astype(np.uint8)
        # 返回0/1的mask
        return mask_dynamic
    
    def _apply_saliency_v3_(self, img_np, k_clusters=5):
        # -------- 🌞 光照均衡：CLAHE 应用于 L 通道 --------
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab = cv2.merge((l_clahe, a, b))

        # -------- 1️⃣ Lab 阈值掩码 --------
        h, w, _ = lab.shape
        # 聚类：将像素按颜色聚类（叶片（正常叶、枯叶、病斑等）/树枝/地面/草地/病斑/其他）
        lab_flat = lab.reshape(-1, 3)
        kmeans = KMeans(n_clusters=k_clusters, random_state=0).fit(lab_flat)
        labels = kmeans.labels_.reshape(h, w)
        # 判断叶子类，面积最大的一类
        unique, counts = np.unique(labels, return_counts=True)
        top2_indices = counts.argsort()[-(k_clusters-1):]
        # 构造掩码
        mask_dynamic = np.isin(labels, top2_indices).astype(np.uint8)
        # 返回0/1的mask
        return mask_dynamic
    
    def _apply_saliency_v4_(self, img_np, thresh=(0.05, 0.30)):
        # -------- 1️⃣ dct阈值掩码 --------
        saliency = dct_saliency(img_np)
        threshold = percentile_threshold(saliency, percentile=15)
        threshold = np.clip(threshold, thresh[0], thresh[1])
        mask_dynamic = (saliency > threshold).astype(np.uint8)
        # 返回0/1的mask
        return mask_dynamic
    
    def _apply_saliency_v5_(self, img_np, thresh=(0.05, 0.30)):
        # -------- 1️⃣ dwt阈值掩码 --------
        from yolov8_utils import wavelet_saliency
        saliency = wavelet_saliency(img_np)
        threshold = percentile_threshold(saliency, percentile=15)
        threshold = np.clip(threshold, thresh[0], thresh[1])
        mask_dynamic = (saliency > threshold).astype(np.uint8)
        # 返回0/1的mask
        return mask_dynamic
    
    def _apply_mask_(self, img, k_size=(5, 5)):
        # 转为 NumPy 格式
        img_np = np.array(img)

        # -------- 1️⃣ 显著性和阈值掩码 --------
        # mask_dynamic = self._apply_saliency_v1_(img_np, thresh=(10, 20)) # v1版本: 固定阈值选叶子前景（HSV）
        # mask_dynamic = self._apply_saliency_v2_(img_np, thresh=(0.05, 0.30)) # v2版本：动态阈值选叶子前景（Lab）
        # mask_dynamic = self._apply_saliency_v3_(img_np, k_clusters=4) # v3版本：kmean分类选top_n面积（Lab）
        mask_dynamic = self._apply_saliency_v4_(img_np, thresh=(0.05, 0.30)) # v4版本：dct检查显著性分割前景
        # mask_dynamic = self._apply_saliency_v5_(img_np, thresh=(0.05, 0.30)) # v5版本：dwt检查显著性分割前景

        # -------- 2️⃣ 滤波与形态学清理 --------
        # 先放大为 0~255，便于滤波操作
        mask_np = (mask_dynamic * 255).astype(np.uint8)
        # 中值滤波
        mask_np = cv2.medianBlur(mask_np, k_size[0])
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_size)
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
        # 二值化，确保最终为 0 或 1
        mask_np = (mask_np > 127).astype(np.uint8)

        # # -------- 3️⃣ 返回 PIL 格式掩码 --------
        mask_pil = Image.fromarray(mask_np * 255)
        return mask_pil
    
    def _create_edge_mask_(self, img, edge=64, threshold=30):
        # 转为 NumPy 格式
        img_np = np.array(img)
        h, w, _ = img_np.shape

        # 截取十分之一作为edge
        edge = int(min(h/10, w/10, edge))
        edge_mask = np.ones((h, w), dtype=np.uint8)

        # 定义黑色区域：RGB 全部分量 < threshold
        black_pixels = np.all(img_np < threshold, axis=2)
        edge_mask[:edge, :] &= ~black_pixels[:edge, :]  # 上边缘
        edge_mask[-edge:, :] &= ~black_pixels[-edge:, :]    # 下边缘
        edge_mask[:, :edge] &= ~black_pixels[:, :edge]  # 左边缘
        edge_mask[:, -edge:] &= ~black_pixels[:, -edge:]    # 右边缘
        # 返回数据
        return edge_mask

    def __getitem__(self, index: int):
        image_path = self.paths[index]
        # 如果缓存有，直接返回
        if image_path in self.caches:
            return self.caches[image_path]
        # 读取图像并转化
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            mask_pil = self._apply_mask_(img, k_size=(7, 7))
            # 是否应用edge遮罩
            if self.edge_mask:
                edge_mask = self._create_edge_mask_(img).astype(np.uint8)
                mask_np = np.array(mask_pil, dtype=np.uint8)      # 0/255
                mask_np = (mask_np // 255) * edge_mask * 255      # 仍然 0/255
                mask_pil = Image.fromarray(mask_np, mode="L")     # 回到 PIL L
            # 保持图片和mask统一尺寸
            if mask_pil.mode != "L":
                mask_pil = mask_pil.convert("L")
            # ====== 关键修改：把 mask 作为 alpha，合成 RGBA，统一变换 ======
            img_rgba = img.copy()
            img_rgba.putalpha(mask_pil)  # 将 mask 作为 alpha 通道
            rgba_tensor = self.transform(img_rgba)
            # 拆回 RGB 和 mask
            image = rgba_tensor[:3, :, :]           # [3, H, W], float32 in [0,1]
            mask  = rgba_tensor[3:, :, :]           # [1, H, W], float32 in [0,1]
            # 将 mask 二值化，避免插值带来的软边（如不需要可保留软mask）
            mask = (mask > 0.5).float()
            # 如果有效像素少于64*64，使用全1的mask替代
            threshold = 64 * 64  # 4096
            valid_pixel_count = mask.sum().item()
            if valid_pixel_count < threshold:
                # 生成全1 mask，尺寸与mask相同
                mask = torch.ones_like(mask)
            # 返回mask后的图片
            # masked_images = image * mask.expand(3, -1, -1)
        # 缓存记录
        self.caches[image_path] = (image, mask, index)
        # 返回图像
        return image, mask, index

# ====================================
# ASSITANCE MODEL
# ====================================
class SimpleForegroundAware(nn.Module):
    def __init__(self, in_channels, k=3):
        super(SimpleForegroundAware, self).__init__()
        self._single_layers_ =  nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=k, padding=k // 2),
            nn.BatchNorm2d(in_channels // 2),
            nn.SiLU(inplace=True),
        )
        self._foreground_ = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=k, padding=k // 2),
            nn.BatchNorm2d(in_channels // 4),
            nn.SiLU(inplace=True),

            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()  # 逐元素压缩到[0,1]
        )
        # 初始化权重
        self._init_weights_()

    def _init_weights_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self._single_layers_(x)
        # 输出前景注意力图
        output = self._foreground_(x)
        return output

class SpatialPyramidPooling(nn.Module):
    """
    Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729.
    - mode: {"pool", "dilated", "hybrid"}，默认 "pool"
        * pool: 原 MaxPool 多尺度
        * dilated: 深度可分离 3x3 空洞卷积(按 pool_sizes 推导 dilation)
        * hybrid: pool + dilated 拼接
    """

    def __init__(self, in_channels, pool_sizes=(5, 7, 9, 11), mode="pool"):
        super(SpatialPyramidPooling, self).__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.SiLU(inplace=True)
        )
        assert mode in ("pool", "dilated", "hybrid"), f"不支持模式{mode}"
        self.mode = mode
        # 定义池化尺度
        if self.mode in ("pool", "hybrid"):
            setattr(self, "poolings", nn.ModuleList([
                nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2)
                for pool_size in pool_sizes
            ]))
        # 有 BN + SiLU，轻量且梯度友好
        if self.mode in ("dilated", "hybrid"):
            setattr(self, "dilateds", nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=((k - 1) // 2), dilation=((k - 1) // 2), groups=(in_channels // 2), bias=False),
                    nn.BatchNorm2d(in_channels // 2),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(in_channels // 2),
                    nn.SiLU(inplace=True),
                )
                for k in pool_sizes
            ]))
        # 计算 hybrid 下的拼接通道倍数
        if self.mode == "hybrid":
            cat_mult = 1 + 2 * len(pool_sizes)  # x + pool分支N + dilated分支N
        else:
            cat_mult = 1 + len(pool_sizes)      # x + 分支N
        # 由于 SPP 增加了通道数，使用卷积来调整通道数
        self.expand = nn.Sequential(
            nn.Conv2d((cat_mult * in_channels // 2), in_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        # 先通过输入通道压缩
        x = self.reduce(x)
        # 对每个池化尺度进行池化，池化后的结果会通过拼接进行合并
        spp_feat = [x]
        # max池化
        if self.mode in ("pool", "hybrid") and hasattr(self, "poolings"):
            poolings = getattr(self, "poolings")
            spp_feat.extend([pooling(x) for pooling in poolings])
        # 空洞卷积
        if self.mode in ("dilated", "hybrid") and hasattr(self, "dilateds"):
            dilateds = getattr(self, "dilateds")
            spp_feat.extend([dilated(x) for dilated in dilateds])
        # 合并特征
        spp_feat = torch.cat(spp_feat, dim=1)
        # 降低拼接后的特征图通道数
        return self.expand(spp_feat)

from attention import SE, CBAM

# 多尺度前景关注模块（结合注意力机制）
class MultiScaleForegroundAware(nn.Module):
    def __init__(self, in_channels, multi_scales, pool_sizes, use_depthwise=False, fusion_mode="concat", attn_block="CBAM"):
        super(MultiScaleForegroundAware, self).__init__()
        self.multi_scales = multi_scales
        # 多尺度卷积层
        self._multi_layers_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm2d(in_channels // 2),
                nn.SiLU(inplace=True)
            ) if (not use_depthwise) else nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=k, padding=k // 2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
                
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels // 2),
                nn.SiLU(inplace=True),
            ) for k in self.multi_scales
        ])
        assert fusion_mode in ["concat", "weight", "attention"], f"不支持模式{fusion_mode}"
        self.fusion_mode = fusion_mode
        # 特征融合方式
        if self.fusion_mode == "concat":
            attn_channels = in_channels // 2
        elif self.fusion_mode == "weight":
            hidden_channels = max(4, in_channels // 8)
            setattr(self, "scorer", nn.Sequential(
                nn.Conv2d(in_channels // 2, hidden_channels, kernel_size=3,  padding=1, bias=True),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_channels, 1, kernel_size=3,  padding=1, bias=True)
            ))
            attn_channels = in_channels // 2
        elif self.fusion_mode == "attention":
            assert attn_block in ["CBAM", "SE"], f"不支持该注意力模型{attn_block}"
            # setattr(self, "attention", SE(in_channels // 2))
            setattr(self, "attention", nn.Sequential(
                CBAM(in_channels // 2) if attn_block == "CBAM" 
                else (SE(in_channels // 2) if attn_block == "SE" else nn.Identity())
            ))
            attn_channels = in_channels // 2
        else:
            raise RuntimeError(f"不支持模式{self.fusion_mode}")
        # 1x1卷积层，用于降维
        out_channels = in_channels // 2
        self.squeeze_1x1 = nn.Sequential(
            nn.Conv2d((in_channels // 2) * len(self.multi_scales), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.squeeze_dw = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.squeeze_act = nn.SiLU(inplace=True)
        # --- SPP + SE（上下文 + 通道选择） ---
        self.pool_sizes = pool_sizes
        """
        * pool: 原 MaxPool 多尺度
        * dilated: 深度可分离 3x3 空洞卷积(按 pool_sizes 推导 dilation)
        * hybrid: pool + dilated 拼接
        """
        self.spp = SpatialPyramidPooling(attn_channels, pool_sizes=self.pool_sizes, mode="dilated")  # SPP池化层
        self.se = SE(attn_channels)
        # self.se = nn.Identity()

        # --- 前景概率头：输出 [B,1,H,W] ---
        self._foreground_ = nn.Sequential(
            nn.Conv2d(attn_channels, attn_channels // 2, kernel_size=1),
            nn.BatchNorm2d(attn_channels // 2),
            nn.SiLU(inplace=True),

            nn.Conv2d(attn_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()  # 逐元素压缩到[0,1]
        )
        # 初始化权重
        self._init_weights_()

    def _init_weights_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # 使用不同尺度的卷积提取多尺度特征
        multi_scale_feats = [feat_layer(x) for feat_layer in self._multi_layers_]
        
        # 拼接在通道维度上
        if self.fusion_mode == "concat":
            multi_scale_feats = torch.cat(multi_scale_feats, dim=1)
        # 权重累加特征图
        elif self.fusion_mode == "weight":
            if hasattr(self, "scorer"):
                scorer = getattr(self, "scorer")
                scores = torch.stack([scorer(feat) for feat in multi_scale_feats], dim=1)
                # Sigmoid+归一化
                # weights = torch.sigmoid(scores)
                # weights = F.dropout(weights, p=0.1, training=self.training)
                # weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
                # softmax+dim=1
                weights = torch.softmax(scores, dim=1)
                weighted_feats  = [multi_scale_feats[i] * weights[:, i, ...] for i in range(len(multi_scale_feats))]
                multi_scale_feats = torch.cat(weighted_feats, dim=1)
            else:
                raise RuntimeError(f"{self.fusion_mode}模式必须设定weights")
        # 注意力权重特征图
        elif self.fusion_mode == "attention":
            if hasattr(self, "attention"):
                weights_attn = getattr(self, "attention")
                attented_feats = [weights_attn(feat) for feat in multi_scale_feats]
                multi_scale_feats = torch.cat(attented_feats, dim=1)
            else:
                raise RuntimeError(f"{self.fusion_mode}模式必须设定attn")
        else:
            raise RuntimeError(f"不支持模式{self.fusion_mode}")
        
        # 使用 1x1 卷积来降低通道数量
        y = self.squeeze_1x1(multi_scale_feats)
        y = y + self.squeeze_dw(y) # 轻残差
        multi_scale_feats = self.squeeze_act(y)
        # 使用 SPP 增加多尺度上下文信息
        multi_scale_feats = self.spp(multi_scale_feats)
        # 使用 SE 注意力机制进行特征选择
        multi_scale_feats = self.se(multi_scale_feats)
        # 前景检测：生成前景概率图
        output = self._foreground_(multi_scale_feats)
        return output

from yolov8_utils import get_center_weight_map

class CenterForegroundAttentionModel(nn.Module):
    def __init__(self, in_channels, use_depthwise=False, fusion_mode="concat", attn_block="CBAM"):
        super(CenterForegroundAttentionModel, self).__init__()
        # v1: 3x3单一尺度前景
        # self.foreground_aware = nn.ModuleList([
        #     SimpleForegroundAware(c1, k=3)
        #     for c1 in in_channels
        # ])
        # v2：多尺度融合前景
        self.foreground_aware = nn.ModuleList([
            MultiScaleForegroundAware(
                c1, self.get_multi_scales(c1), self.get_pool_sizes(c1),
                use_depthwise=use_depthwise, fusion_mode=fusion_mode, attn_block=attn_block
            )
            for c1 in in_channels
        ])
    
    def get_multi_scales(self, in_channels):
        if in_channels <= 192: # [B, 192, 80, 80]
            multi_scales = (1, 3, 5)
        elif in_channels <= 384: # [B, 384, 40, 40]
            multi_scales = (3, 5, 7)
        elif in_channels <= 576: # [B, 576, 20, 20]
            multi_scales = (5, 7, 9)
        else:
            multi_scales = (7, 9, 11)
        return multi_scales

    def get_pool_sizes(self, in_channels):
        if in_channels <= 192: # [B, 192, 80, 80]
            pool_sizes = (3, 5, 7)
        elif in_channels <= 384: # [B, 384, 40, 40]
            pool_sizes = (5, 7, 9)
        elif in_channels <= 576: # [B, 576, 20, 20]
            pool_sizes = (7, 9, 11)
        else:
            pool_sizes = (9, 11, 13)
        return pool_sizes
    
    def _center_aware_(self, feat):
        # 获取中心加权图
        _, _, H, W = feat.shape
        weights_map = get_center_weight_map(H, W, sigma=2.0).to(feat.device)
        return weights_map

    def forward(self, features):
        attented_features = []
        for feat, _foreground_aware_ in zip(features, self.foreground_aware):
            attented_feat = _foreground_aware_(feat)
            # 动态调整中心加权
            weighted_feat = attented_feat * self._center_aware_(feat)
            attented_features.append(weighted_feat)
        return attented_features

# ====================================
# MEMORY BANK
# ====================================
class MemoryBank:
    def __init__(self, size: int, feature_dims: int, device: torch.device):
        self.capacity = size
        self.device = device
        self.feature_dims = feature_dims  # 这里存储了特征的维度
        self.banks = [
            torch.empty((0, feature_dim), device=device)  # 每个尺度一个单独的内存库
            for feature_dim in feature_dims
        ]
        self.length = 0  # 初始化长度

    def update(self, features):
        # 确保特征数量与内存库维度匹配
        assert len(features) == len(self.banks), "特征数量与内存库维度不匹配"
        # 更新每个尺度的内存库
        for i, feature in enumerate(features):
            # 分离特征
            feature = feature.detach()
            # 添加到内存库
            self.banks[i] = torch.cat([self.banks[i], feature], dim=0)
            # 如果超过容量，移除最早的特征
            if self.banks[i].size(0) > self.capacity:
                self.banks[i] = self.banks[i][-self.capacity:]
        # 记录len数量
        self.length = self.banks[0].shape[0]  # 更新长度

    def _all(self, num, mode='random'):
        if self.length == 0: # 返回空的tensor
            return [
                torch.empty((0, feature_dim), device=self.device)  # 每个尺度一个单独的内存库
                for feature_dim in self.feature_dims
            ]
        # 确保 num 在合理范围内
        if num <= 0 or num > self.length:
            num = self.length
        # 生成索引
        if mode == 'random':
            indices = torch.randperm(self.length)[:num]
        elif mode == 'recent':
            indices = torch.arange(self.length-num, self.length)
        else:  # 默认返回所有样本
            indices = torch.arange(0, self.length)
        # 为每个尺度选择样本
        sampled_features = []
        for bank in self.banks:
            # 确保索引在有效范围内
            valid_indices = indices[indices < bank.size(0)]
            sampled = bank[valid_indices]
            sampled_features.append(sampled)
        # 返回选择好样本
        return sampled_features

    def _capacity(self): return self.capacity

    def _length(self): return self.length

# ====================================
# MODEL DEFINITION v1
# ====================================
class SimCLRv1YOLOv8(nn.Module):
    def __init__(self, backbone, in_channels, layer_indices, augmentation, normalize=False,
                 projector_dim=256, dropout_r=0.2, memory_size=4096, momentum=0.999):
        super(SimCLRv1YOLOv8, self).__init__()
        if isinstance(augmentation, tuple):
            self.g_augmentation, self.x_augmentation = augmentation
        elif augmentation:
            self.g_augmentation, self.x_augmentation = None, augmentation
        else:
            self.g_augmentation, self.x_augmentation = None, None
        if normalize:
            self.normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            self.normalize = None
        # 进行多尺度改造
        self.layer_indices = layer_indices
        self.in_channels = in_channels
        self.backbone_q = MultiScaleFeatureExtractor(backbone, layer_indices, dropout_r)
        self.attn_module = MultiScaleFeatureAttention(attentions=nn.ModuleList([nn.Identity() for _ in in_channels]))
        # ----------------------------
        # projector（SimCLR v1）
        # ----------------------------
        self.out_features = projector_dim  # 投影维度
        # self.projector_q = self.get_projector(self.in_channels, self.out_features)
        def _projector_builder_(feature_channels=in_channels, output_dim=projector_dim):
            return nn.ModuleList([
                self.get_projector(c1, output_dim, dropout_r)
                for c1 in feature_channels
            ])
        self.projector_q = MultiScaleFeatureProjector(_projector_builder_)
        # ----------------------------
        # Memory bank
        # ----------------------------
        self.memory_bank = MemoryBank(size=memory_size, feature_dims=[self.out_features]*len(in_channels), device=DEVICE)
        # ----------------------------
        # 🔁 Momentum encoder
        # ----------------------------
        import copy
        self.backbone_k = copy.deepcopy(self.backbone_q)
        self.projector_k = copy.deepcopy(self.projector_q)

        for param in self.backbone_k.parameters():
            param.requires_grad = False
        for param in self.projector_k.parameters():
            param.requires_grad = False

        self.momentum = momentum  # momentum coefficient

    @property
    def backbone(self): return self.backbone_q.backbone

    # 计算下一级2的幂数
    def align_pow2(self, c1):
        return 2 ** math.ceil(math.log2(c1))

    def get_projector(self, in_features, out_features, dropout_r=8):
        # 为了更好的对比学习，改进了投影头设计：
        # - 使用两层 MLP，并在中间加入非线性激活（参考 SimCLR 方法）
        # - 使用 BatchNorm 提高训练稳定性
        # - 隐藏层维度设置为输出维度的 2 倍（符合最佳实践）
        # - 使用 GELU 激活函数，在很多场景下性能优于 ReLU
        hidden_dim = self.align_pow2(in_features)
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            nn.Linear(in_features, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_r),

            nn.Linear(hidden_dim, out_features, bias=True),
            nn.BatchNorm1d(out_features, affine=False, momentum=0.05),
        )

    @torch.no_grad()
    def _apply_momentum_update(self):
        # EMA 更新 backbone_k 和 projector_k 参数
        for param_q, param_k in zip(self.backbone_q.parameters(), self.backbone_k.parameters()):
            param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data

        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data
    
    def bank_all(self, num=-1, mode='random'):
        return self.memory_bank._all(num, mode)
    
    def bank_capacity(self):
        return self.memory_bank._capacity()
    
    def bank_length(self):
        return self.memory_bank._length()

    def _forward_backbone_q(self, x):
        """使用 checkpoint 包裹backbone，节省显存"""
        if not x.requires_grad:
            x.requires_grad_(True)
        return checkpoint(self.backbone_q, x, use_reentrant=False)
    
    @torch.no_grad()
    def _forward_backbone_k(self, x):
        """使用 checkpoint 包裹backbone，节省显存"""
        return self.backbone_k(x)
    
    def _apply_augmentation(self, x, masks=None):
        # 获取两次增强结果
        if self.g_augmentation and masks is not None:
            # 1) 拼接 -> 同步几何增强
            combined = torch.cat([x, masks], dim=1)
            combined_aug_1 = self.g_augmentation(combined)
            combined_aug_2 = self.g_augmentation(combined)
            # 2) 分离（前3通道图像，后1通道mask）
            aug_1 = combined_aug_1[:, :3, :, :]  # [B, 3, H, W]
            aug_2 = combined_aug_2[:, :3, :, :]  # [B, 3, H, W]
            masks_1 = combined_aug_1[:, 3:, :, :]  # [B, 1, H, W]
            masks_2 = combined_aug_2[:, 3:, :, :]  # [B, 1, H, W]
            # 3) 让mask回到0/1（二值化，避免插值带来的灰度）
            masks_1 = (masks_1 > 0.5).float()
            masks_2 = (masks_2 > 0.5).float()
        elif self.g_augmentation:
            aug_1 = self.g_augmentation(x)
            aug_2 = self.g_augmentation(x)
            masks_1 = None
            masks_2 = None
        else:
            aug_1 = x
            aug_2 = x
            masks_1 = None
            masks_2 = None

        # 4) 仅对图像做颜色/模糊等增强
        if self.x_augmentation:
            aug_1 = self.x_augmentation(aug_1)
            aug_2 = self.x_augmentation(aug_2)

        # 5) tensor标准化
        if self.normalize:
            aug_1 = self.normalize(aug_1)
            aug_2 = self.normalize(aug_2)

        # 6) 返回结果
        return aug_1, aug_2, masks_1, masks_2
    
    def forward(self, x, masks=None):
        if self.training:
            # 获取两次增强结果
            aug_1, aug_2, masks_1, masks_2 = self._apply_augmentation(x, masks)
            # 获取两次特征表征结果
            rsp_1 = self._forward_backbone_q(aug_1)
            rsp_2 = self._forward_backbone_q(aug_2)
            rsp_anchor = self._forward_backbone_q(x)
            # SimCLR计算loss
            cmp_1 = self.projector_q(self.attn_module(rsp_1))
            cmp_2 = self.projector_q(self.attn_module(rsp_2))
            cmp_anchor = self.projector_q(self.attn_module(rsp_anchor))
            # 更新 memory bank
            self._apply_momentum_update()
            with torch.no_grad():
                k_1 = self._forward_backbone_k(aug_1)
                k_2 = self._forward_backbone_k(aug_2)
                z_1 = self.projector_k(self.attn_module(k_1))
                z_2 = self.projector_k(self.attn_module(k_2))
                self.memory_bank.update(z_1)
                self.memory_bank.update(z_2)
            # 返回结果
            return rsp_1, rsp_2, cmp_1, cmp_2, masks_1, masks_2, rsp_anchor, cmp_anchor
        else:
            rsp_0 = self.backbone_q(x)
            # 返回结果
            return rsp_0

# ====================================
# MODEL DEFINITION v2
# ====================================
class SimCLRv2YOLOv8(SimCLRv1YOLOv8):
    def __init__(self, backbone, in_channels, layer_indices, augmentation, normalize=False,
                 projector_dim=256, dropout_r=0.2, memory_size=4096, momentum=0.999):
        super(SimCLRv2YOLOv8, self).__init__(backbone, in_channels, layer_indices, augmentation, normalize,
                                             projector_dim, dropout_r, memory_size, momentum)

    def get_projector(self, in_features, out_features, dropout_r=1e-8):
        # 重写 projection head 为 3 层 MLP（SimCLR v2）
        hidden_dim = self.align_pow2(in_features)
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(in_features, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),      # 比 ReLU 和 LeakyReLU 捕捉到更多复杂的特征
            nn.Dropout(p=dropout_r),    # Dropout 提高模型对新数据的鲁棒性

            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_r),

            nn.Linear(hidden_dim, out_features, bias=True),
            nn.BatchNorm1d(out_features, affine=False, momentum=0.05),
        )
    
    def _normalize(self, features):
        features = [F.normalize(x, dim=1) for x in features]
        return features
    
    def forward(self, x, masks=None):
        if self.training:
            # 获取两次增强结果
            aug_1, aug_2, masks_1, masks_2 = self._apply_augmentation(x, masks)
            # 获取两次特征表征结果
            rsp_1 = self._forward_backbone_q(aug_1)
            rsp_2 = self._forward_backbone_q(aug_2)
            rsp_anchor = self._forward_backbone_q(x)
            # SimCLR计算loss
            cmp_1 = self._normalize(self.projector_q(self.attn_module(rsp_1)))
            cmp_2 = self._normalize(self.projector_q(self.attn_module(rsp_2)))
            cmp_anchor = self._normalize(self.projector_q(self.attn_module(rsp_anchor)))
            # 更新 memory bank
            self._apply_momentum_update()
            with torch.no_grad():
                k_1 = self._forward_backbone_k(aug_1)
                k_2 = self._forward_backbone_k(aug_2)
                z_1 = self._normalize(self.projector_k(self.attn_module(k_1)))
                z_2 = self._normalize(self.projector_k(self.attn_module(k_2)))
                self.memory_bank.update(z_1)
                self.memory_bank.update(z_2)
            # 返回数据和 negatives，供 loss 使用
            return rsp_1, rsp_2, cmp_1, cmp_2, masks_1, masks_2, rsp_anchor, cmp_anchor
        else:
            rsp_0 = self.backbone_q(x)
            # 返回结果
            return rsp_0

# ====================================
# DATA TRANSFORM DEFINITION
# ==================================== 
class GaussianNoise(torch.nn.Module):
    """基于 PyTorch 的高斯噪声模块"""
    def __init__(self, mean=0., std=0.05):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        if self.training:
            noise = torch.randn_like(tensor) * self.std + self.mean
            return tensor + noise
        return tensor

class CLAHE(nn.Module):
    """基于 PyTorch 的光平衡模块"""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        super(CLAHE, self).__init__()
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def forward(self, tensor):
        # 保存原始精度和设备
        orig_dtype = tensor.dtype
        device = tensor.device

        # 确保是 float32，方便处理 OpenCV 图像
        tensor = tensor.float()

        batch = []
        for img in tensor:  # img shape: [C, H, W]
            img_np = img.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
            img_np = (img_np * 255).astype(np.uint8)

            if img_np.shape[2] == 3:
                lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                l = self.clahe.apply(l)
                lab = cv2.merge((l, a, b))
                img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                img_np = self.clahe.apply(img_np)

            img_np = img_np.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [C, H, W]
            batch.append(img_tensor)

        batch_tensor = torch.stack(batch).to(device)

        # 恢复混合精度
        return batch_tensor.to(orig_dtype)

class EdgeEnhancement(nn.Module):
    """基于 PyTorch 的边缘增强模块"""
    def __init__(self, intensity=0.5):
        super(EdgeEnhancement, self).__init__()
        self.intensity = intensity
        # 创建 Sobel 边缘检测核并注册为缓冲区
        sobel_x = torch.tensor([
            [[[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]]
        ], dtype=torch.float32)
        
        sobel_y = torch.tensor([
            [[[-1, -2, -1],
              [0, 0, 0],
              [1, 2, 1]]]
        ], dtype=torch.float32)
        
        # 注册为缓冲区，不参与梯度更新
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def forward(self, tensor):
        # 保存原始数据类型
        orig_dtype = tensor.dtype
        
        # 获取设备
        device = tensor.device
        
        # 确保核在正确设备上
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)
        
        # 转换为灰度图
        if tensor.shape[1] == 3:  # RGB图像
            gray = 0.299 * tensor[:, 0] + 0.587 * tensor[:, 1] + 0.114 * tensor[:, 2]
            gray = gray.unsqueeze(1)  # [B, 1, H, W]
        else:  # 单通道图像
            gray = tensor
        
        # 应用 Sobel 边缘检测
        edges_x = F.conv2d(gray, sobel_x, padding=1)
        edges_y = F.conv2d(gray, sobel_y, padding=1)
        edges = torch.sqrt(edges_x**2 + edges_y**2)
        
        # 归一化边缘图
        B = edges.shape[0]
        edges = edges.view(B, -1)
        emin, emax = edges.min(dim=1).values.view(B,1,1,1), edges.max(dim=1).values.view(B,1,1,1)
        edges = (edges.view(B,1,*gray.shape[-2:]) - emin) / (emax - emin + 1e-8)
        
        # 创建边缘掩码
        edge_mask = (edges > 0.1).float()
        
        # 膨胀边缘掩码
        kernel = torch.ones(1, 1, 3, 3, device=device)
        dilated_mask = F.conv2d(edge_mask, kernel, padding=1)
        dilated_mask = (dilated_mask > 0).float()
        
        # 增强边缘区域
        enhanced = tensor.clone()
        for c in range(enhanced.shape[1]):
            enhanced[:, c] = torch.where(
                dilated_mask.squeeze(1) > 0,
                torch.clamp(enhanced[:, c] * (1 + self.intensity), 0, 1),
                enhanced[:, c]
            )
        
        return enhanced.to(orig_dtype)

class AdjustGamma(nn.Module):
    def __init__(self, gamma_range=(0.8, 1.2)):
        super(AdjustGamma, self).__init__()
        assert gamma_range[0] > 0, "gamma must be > 0"
        self.gamma_range = gamma_range

    def forward(self, img):
        # img: Tensor in [0,1], shape (C,H,W) or (B,C,H,W)
        # torchvision F.adjust_gamma 支持 tensor（逐通道）
        gamma = torch.empty(1).uniform_(*self.gamma_range).item()
        return VF.adjust_gamma(img, gamma=gamma)


def clr_transforms(img_sz):
    # augmentation = v2.Compose([
    #     v2.Resize(size=img_sz),
    #     v2.RandomApply([v2.RandomAffine(degrees=0, scale=(0.85, 1.5))], p=0.15),
    #     v2.CenterCrop(size=img_sz),
    #     # v2.RandomResizedCrop(size=img_sz, scale=(0.5, 1.0)),
    #     v2.RandomApply([v2.RandomRotation(degrees=15)], p=0.3),
    #     v2.RandomHorizontalFlip(p=0.5),
    #     v2.RandomVerticalFlip(p=0.1),
    #     v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)], p=0.4),
    #     v2.RandomGrayscale(p=0.3),
    #     v2.RandomApply([v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))], p=0.3),
    #     v2.RandomErasing(scale=(0.05, 0.08), ratio=(0.3, 3.3), value='random', p=0.1),
    #     v2.RandomApply([GaussianNoise(mean=0.0, std=0.02)], p=0.2),
    #     v2.RandomApply([CLAHE(clip_limit=0.8, tile_grid_size=(8, 8))], p=0.2),
    #     v2.ConvertImageDtype(),  # 注意：确保在 mask 转换前不要调用
    # ])
    augmentation = v2.Compose(
        list(clr_transforms_v2(img_sz))
    )
    return augmentation

def clr_transforms_v2(img_sz):
    """
    增强=忽略=不变性（不关心表示不敏感）
    1、几何形状增强，表示我们在训练模型对抗几何变形，表示下游任务目标不关心其形状表征。
    2、高斯模糊，表示忽略内部细节，但是关心整体外形。
    3、颜色增强，表示下游任务不关心颜色变化
    """
    return v2.Compose([ # img和mask共享
        # 轻度随机裁剪，尽量保留病斑上下文
        v2.RandomResizedCrop(size=img_sz, scale=(0.92, 1.0), ratio=(0.95, 1.05)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.1), # 增加垂直翻转概率
        # 轻仿射：少量旋转/平移/缩放，既增广又不破坏框/mask一致性
        v2.RandomApply([v2.RandomAffine(degrees=10, translate=(0.03,0.03), scale=(0.95,1.05))], p=0.25),
        v2.RandomApply([v2.RandomRotation(degrees=10)], p=0.15),
        # Perspective：叶片弯曲非刚体，会导致mask失真
        # v2.RandomPerspective(distortion_scale=0.2, p=0.3),
    ]), v2.Compose([ # img增强
        # 轻量色彩抖动（避免把藻斑/枯叶细微色差抹掉）
        v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.005)], p=0.25),
        v2.RandomGrayscale(p=0.05),

        # 模糊与锐化二选一
        # v2.RandomChoice([v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.2)), v2.RandomAdjustSharpness(sharpness_factor=1.4), v2.Identity(),]),
        v2.RandomChoice([
            # 轻模糊：拟合对焦/湿度/雾气；核适度放宽以减少“过锐化”偏置
            v2.GaussianBlur(kernel_size=(3,3), sigma=(0.1,0.8)),
            # 轻锐化：适度提升叶脉/病斑边界，不要过强，否则压召回
            EdgeEnhancement(intensity=0.25),
            # 伽马：模拟顺光/逆光等光照变化
            AdjustGamma(gamma_range=(0.9,1.1)),
            # 恒等：维持分布稳定，避免过拟合到增强
            v2.Identity()
        ], p=[0.1, 0.25, 0.15, 0.5]),
        # v2.RandomApply([v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.8))], p=0.1), # 低概率避免细节过多丢失
        # v2.RandomApply([EdgeEnhancement(intensity=0.3)], p=0.2), # 边缘增强有助于对比学习

        # 先增强细节再遮挡
        v2.RandomApply([CLAHE(clip_limit=0.8, tile_grid_size=(8, 8))], p=0.15), # 提升细节的可见性
        # v2.RandomApply([v2.RandomAdjustSharpness(sharpness_factor=1.5)], p=0.2),

        # 提升稳定性
        # v2.RandomApply([v2.RandomEqualize()], p=0.1),  # 直方图均衡
        v2.RandomApply([v2.ElasticTransform(alpha=20.0)], p=0.05),  # 轻微弹性形变, 模拟风吹叶片形变
        # v2.RandomApply([AdjustGamma(gamma_range=(0.8, 1.2))], p=0.1),  # 轻微伽马变化，拟合光照

        # 轻遮挡（模拟叶片重叠），面积略减、比例收窄，避免吞噬小病斑
        v2.RandomErasing(scale=(0.04, 0.08), ratio=(0.30, 2.50), value='random', p=0.15),
        # 轻噪声：提升鲁棒性，别太大
        v2.RandomApply([GaussianNoise(mean=0.0, std=0.015)], p=0.10),

        # 统一数据类型（若上游未做）
        # v2.ToDtype(torch.float32, scale=True),
    ])

# ====================================
# TRAINNING FUN
# ==================================== 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_worker(worker_id):
    import random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_optimizer_params(_model, lr, decay):
    no_decay_keywords = ['bias', 'norm', 'bn', 'embedding']
    # 处理 clr_model 参数
    decay_params = []
    no_decay_params = []
    geo_params = []
    for name, param in _model.named_parameters():
        if not param.requires_grad: continue
        if "geo" in name and "alpha" in name: geo_params.append(param)
        elif any(keyword in name for keyword in no_decay_keywords): no_decay_params.append(param)
        else: decay_params.append(param)
    params_group = [
        {'params': decay_params, 'lr': lr, 'weight_decay': decay}, 
        {'params': no_decay_params, 'lr': lr, 'weight_decay': 0.0}
    ]
    # 调整geo.alpha参数
    if geo_params and len(geo_params) > 0:
        params_group.append(
            {'params': geo_params, 'lr': lr * 0.1, 'weight_decay': 0.0, "name": "geo_alpha"}
        )
        print("[DETECT geo.alpha]", geo_params)
    return params_group

def clr_model_train(args, pre_train=None):
    # 加载YOLO模型, 获取backbone
    _model = load_model(model_path=args.config, task=args.task, pretrain_path=args.pretrain, modify_model=True)
    backbone, layers_dims, layer_indices = get_backbone(model=_model, task=args.task)
    # 构建model等训练参数
    # clr_augmentation = clr_transforms(img_sz=args.image_size)
    clr_augmentation = clr_transforms_v2(img_sz=args.image_size)
    if args.clr_version == "v2":
        clr_model = SimCLRv2YOLOv8(backbone, layers_dims, layer_indices, clr_augmentation, 
                                   projector_dim=args.proj_dims, dropout_r=args.dropout_r,
                                   memory_size=args.bank_size, momentum=args.bank_momentum)
    else:
        clr_model = SimCLRv1YOLOv8(backbone, layers_dims, layer_indices, clr_augmentation, 
                                   projector_dim=args.proj_dims, dropout_r=args.dropout_r,
                                   memory_size=args.bank_size, momentum=args.bank_momentum)
    # 为clr_model添加注意力机制
    clr_model.attn_module = MultiScaleFeatureAttention(nn.ModuleList([
        get_attention(in_channels, task=args.task)
        for in_channels in layers_dims
    ]))
    # 返回clr模型继续改造
    if pre_train: pre_train(clr_model)
    # 定义辅助网络
    ast_model = CenterForegroundAttentionModel(layers_dims, use_depthwise=args.use_depthwise, fusion_mode=args.fusion_mode)
    print(f"辅助模型：{ast_model}")
    # 确保全部参与训练
    for param in clr_model.parameters():
        param.requires_grad = True
    for param in ast_model.parameters():
        param.requires_grad = True
    g = torch.Generator()
    g.manual_seed(args.seed)
    clr_dataset = ImageForlderLoader(args.image_dir)
    num_workers = min(os.cpu_count(), 8) if os.cpu_count() > 1 else 0
    clr_dataloader = DataLoader(
        clr_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
        drop_last=True,  # 建议对比学习开启，保证BN/配对稳定
        pin_memory=True,
    )
    best_loss = float("inf")
    patience_counter = 0
    best_weights_path = Path(args.output_dir).joinpath("weights").joinpath("best_clr.pt")
    best_weights_path.parent.mkdir(parents=True, exist_ok=True)
    last_weights_path = Path(args.output_dir).joinpath("weights").joinpath("last_clr.pt")
    last_weights_path.parent.mkdir(parents=True, exist_ok=True)
    best_backbone_weights_path = Path(args.output_dir).joinpath("weights").joinpath("best_backbone.pt")
    best_backbone_weights_path.parent.mkdir(parents=True, exist_ok=True)
    last_backbone_weights_path = Path(args.output_dir).joinpath("weights").joinpath("last_backbone.pt")
    last_backbone_weights_path.parent.mkdir(parents=True, exist_ok=True)
    best_pretrain_weights_path = Path(args.output_dir).joinpath("weights").joinpath("best_pretrain.pt")
    best_pretrain_weights_path.parent.mkdir(parents=True, exist_ok=True)
    last_pretrain_weights_path = Path(args.output_dir).joinpath("weights").joinpath("last_pretrain.pt")
    last_pretrain_weights_path.parent.mkdir(parents=True, exist_ok=True)
    # 启动 AMP
    scaler = GradScaler(enabled=(not args.no_amp))
    # 开始训练
    clr_model.to(DEVICE)
    ast_model.to(DEVICE)
    # from pytorch_metric_learning.losses import NTXentLoss
    from yolov8_utils import NTXentLoss, TripletLoss, CenterLoss
    # 计算相似度，正例最大相似，负例最大不相似（类内相似，类间区分）
    ntxent_loss_func = NTXentLoss(temperature=args.temperature, reduction="mean", train_able=False)
    # 计算与锚点L2距离，正例距离最小，负例距离最大（类内聚集，类间分散）
    triplet_loss_func = TripletLoss(margin=args.margin, reduction="mean", use_cosine=True, hard_mining=args.trip_mode, train_able=False)
    # 计算类中心聚集，同一类别具备向心性（缺乏类别信息，无法自监督）
    # center_loss_func = CenterLoss(num_classes=num_classes, feat_dim=256, device=DEVICE)
    # 1️⃣ 获取主网络和辅助网络的参数组
    lr_main, lr_ast = args.lr, args.lr * 0.5
    params_group = get_optimizer_params(clr_model, lr_main, args.weight_decay) + get_optimizer_params(ast_model, lr_ast, args.weight_decay)
    # 1. 新建列表收集附加参数
    extra_params = []
    if isinstance(ntxent_loss_func.temperature, nn.Parameter):
        extra_params.append({"params": [ntxent_loss_func.temperature], "lr": lr_main * 0.2, "weight_decay": 0.0, "name": "ntxent_temperature"})
    if isinstance(triplet_loss_func.margin, nn.Parameter):
        extra_params.append({"params": [triplet_loss_func.margin], "lr": lr_main * 0.4, "weight_decay": 0.0, "name": "triplet_margin"})
    # 2. 附加参数更新到params_group中
    if extra_params and len(extra_params) > 0:
        params_group = params_group + extra_params
        print("[DETECT extra.params]", [p["params"] for p in extra_params])
    # 2️⃣ 构建优化器
    optimizer = torch.optim.AdamW(
        params_group, 
        lr=lr_main,  # 这个学习率仅控制整个优化器的学习率
        betas=(0.9, 0.999), 
        eps=1e-8, 
        weight_decay=args.weight_decay
    )
    # 最大调度学习率
    max_lr_list = [params["lr"] * args.lr_factor for params in params_group]
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr_list,  # 设置每个参数组的最大学习率
        steps_per_epoch=len(clr_dataloader),
        epochs=args.epochs,
        pct_start=0.15,
        anneal_strategy='cos',
        div_factor=10.0,  # 初始 lr = max_lr / div_factor
        final_div_factor=100.0,  # 最终 lr = max_lr / final_div_factor
    )
    # memory bank最小负样本数门槛
    memory_bank_warmup = min(int(args.bank_sample_size), args.bank_size)
    memory_bank_sample_size = args.bank_sample_size
    # 开始训练
    for epoch in range(args.epochs):
        clr_model.train()
        ast_model.train()
        accumulated_step_loss = 0.0
        use_memory_bank = (clr_model.bank_length() >= memory_bank_warmup)
        # 训练进度条
        progress_bar = tqdm.tqdm(clr_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for _, (images, masks, _) in enumerate(progress_bar):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            data = images * masks.expand(-1, 3, -1, -1)
            optimizer.zero_grad(set_to_none=True)
            # 训练backbone
            with autocast(enabled=(not args.no_amp), dtype=torch.bfloat16):  # AMP混合精度
                rsp_1, rsp_2, cmp_1, cmp_2, masks_1, masks_2, _, cmp_anchor = clr_model(data, masks)
                # 计算mlp投影loss
                if use_memory_bank: bank_samples = clr_model.bank_all(num=memory_bank_sample_size)
                else: bank_samples = [None] * len(cmp_1)
                proj_loss = torch.zeros((), device=DEVICE)
                trip_loss = torch.zeros((), device=DEVICE)
                for s_cmp_1, s_cmp_2, s_cmp_anchor, s_bank_samples in zip(cmp_1, cmp_2, cmp_anchor, bank_samples):
                    proj_loss = proj_loss + ntxent_loss_func(s_cmp_1, s_cmp_2, s_bank_samples)
                    trip_loss = trip_loss + triplet_loss_func(s_cmp_anchor, s_cmp_1, s_cmp_2, s_bank_samples)
                # 辅助网络，让backbone更加关注图像中心，前景，树叶
                rsp_1_attn_maps = ast_model(rsp_1)
                rsp_2_attn_maps = ast_model(rsp_2)
                # 计算前景中心感知loss
                fgrd_loss = torch.zeros((), device=DEVICE)
                for s_rsp_1_attn_maps, s_rsp_2_attn_maps in zip(rsp_1_attn_maps, rsp_2_attn_maps):
                    s_gt_coords_1 = center_coords_of_masks(masks=masks_1 if masks_1 is not None else masks, device=s_rsp_1_attn_maps.device)
                    s_gt_coords_2 = center_coords_of_masks(masks=masks_2 if masks_2 is not None else masks, device=s_rsp_2_attn_maps.device)
                    s_fg_loss_1 = foreground_attended_loss(gt_coords=s_gt_coords_1, attn_maps=s_rsp_1_attn_maps, sigma=args.sigma)
                    s_fg_loss_2 = foreground_attended_loss(gt_coords=s_gt_coords_2, attn_maps=s_rsp_2_attn_maps, sigma=args.sigma)
                    fgrd_loss = fgrd_loss + 0.5 * (s_fg_loss_1 + s_fg_loss_2)
                # 计算loss加权
                loss = args.w_proj * proj_loss + args.w_trip * trip_loss + args.w_fgrd * fgrd_loss
                # 让 τ 和 m 保持差异, 使两者不容易走同向
                if extra_params and len(extra_params) > 0:
                    tau, m = ntxent_loss_func._temperature(), triplet_loss_func._margin()
                    base_decouple_loss = (tau - m).pow(2)
                    w_decp = 1.0 if base_decouple_loss.item() <= 0 else (0.01 * loss.item()) / (base_decouple_loss.item() + 1e-8)
                    decouple_loss = w_decp * base_decouple_loss
                    loss = loss + decouple_loss
                    decp_val = decouple_loss.detach().item()
                else:
                    decp_val = 0.0
            # AMP混合精度更新
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) # 取消梯度缩放
            # 裁剪：避免每步构造 params list，直接遍历 param_groups
            for g in optimizer.param_groups: 
                group_name = g.get("name", "")
                if "ntxent_temperature" in group_name or "triplet_margin" in group_name:
                    torch.nn.utils.clip_grad_norm_(g["params"], max_norm=100.0) 
                else: 
                    torch.nn.utils.clip_grad_norm_(g["params"], max_norm=1.0)
            # 记录 step 前的 scale
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # 只有当没有溢出、确实执行了 optimizer.step()，才推进 scheduler
            if scaler.get_scale() >= prev_scale:
                lr_scheduler.step()
            # 此处计算best得分倾向主网络得分
            batch_size = data.size(0)
            step_val = loss.detach().item()
            accumulated_step_loss += step_val * batch_size
            proj_val, trip_val, fgrd_val = proj_loss.detach().item(), trip_loss.detach().item(), fgrd_loss.detach().item()
            # 更新进度条
            progress_bar.set_postfix({
                "loss": f"{step_val:.4f}",
                "proj": f"{proj_val:.4f}",
                "trip": f"{trip_val:.4f}",
                "fgrd": f"{fgrd_val:.4f}",
                "decp": f"{decp_val:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })
            # 计算当前迭代总loss
        # 下一轮epoch训练
        if extra_params and len(extra_params) > 0:
            τ = float(ntxent_loss_func._temperature().detach().cpu())
            τ_grad = (ntxent_loss_func.temperature.grad.norm().item() if ntxent_loss_func.temperature.grad is not None else 0.0)
            m = float(triplet_loss_func._margin().detach().cpu())
            m_grad = (triplet_loss_func.margin.grad.norm().item() if triplet_loss_func.margin.grad is not None else 0.0)
            extra_info = f", τ={τ:.4f}(|grad(τ)|={τ_grad:.2e}), m={m:.4f}(|grad(m)|={m_grad:.2e})"
        else:
            extra_info = ""
        batch_loss = accumulated_step_loss / len(clr_dataset)
        print(f"Epoch {epoch+1:03d}/{args.epochs}, Loss={batch_loss:.4f}{extra_info}, use_memory_bank: {use_memory_bank}")
        # 如果还没有bank计算，loss会不准确
        if use_memory_bank:
            # best_loss保存结构
            if batch_loss < best_loss:
                best_loss = batch_loss
                patience_counter = 0
                # 保存best模型
                torch.save({
                    "epoch": epoch, "loss": best_loss,
                    "model_state_dict": clr_model.state_dict(),
                    "ast_state_dict": ast_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, best_weights_path)
            else:
                patience_counter += 1
                print(f"连续 {patience_counter} 个周期内无提升。最佳损失值为：{best_loss:.4f}")
        # 保存last模型
        torch.save({
            "epoch": epoch, "loss": batch_loss,
            "model_state_dict": clr_model.state_dict(),
            "ast_state_dict": ast_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, last_weights_path)
        # early stopping策略检查
        if patience_counter >= args.patience:
            print(f"在第 {epoch+1} 个周期后触发了提前停止，因为在连续 {args.patience} 个周期内未见改善。")
            break
    # 保存 last_backbone 的权重
    torch.save(clr_model.backbone_q.backbone.state_dict(), last_backbone_weights_path)
    # 保存 last YOLOv8模型保存
    _model.save(last_pretrain_weights_path)
    # 返回best最优结果
    if osp.exists(best_weights_path):
        checkpoint = torch.load(best_weights_path)
        clr_model.load_state_dict(checkpoint["model_state_dict"])
        print(f"已加载来自第 {checkpoint['epoch']+1} 个周期的最佳模型，损失值为：{checkpoint['loss']:.4f}")
        # 保存 best_backbone 的权重
        torch.save(clr_model.backbone_q.backbone.state_dict(), best_backbone_weights_path)
        # 保存 best YOLOv8模型保存
        _model.save(best_pretrain_weights_path)
    print("对比学习训练完成！")

# 适配模型
def pre_train(clr_model:SimCLRv1YOLOv8):
    print(f"调整后模型结构:\n{clr_model}")
    return clr_model

def set_seed(seed:int=42):
    import random
    random.seed(seed)                     # Python 内置随机种子
    np.random.seed(seed)                  # Numpy 随机种子
    torch.manual_seed(seed)               # PyTorch CPU 随机种子
    torch.cuda.manual_seed(seed)          # PyTorch 当前 GPU 随机种子
    torch.cuda.manual_seed_all(seed)      # 所有 GPU
    torch.backends.cudnn.deterministic = True  # 让 cudnn 产生可重复结果
    torch.backends.cudnn.benchmark = False     # 避免非确定性算法优化
    os.environ['PYTHONHASHSEED'] = str(seed)   # Python 哈希种子

def args_parser():
    parser = argparse.ArgumentParser(description="训练YOLOv8特征提取backbon(SimCLR)")
    parser.add_argument("--image_dir", type=str, default="train", help="图片目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--task", type=str, default="detect", choices=["classify", "detect"], help="训练模型类别")
    parser.add_argument("--config", type=str, default="yolov8m.yaml", help="YAML配置")
    parser.add_argument("--pretrain", type=str, default="yolov8m.pt", help="预训练权重")
    parser.add_argument("--epochs", type=int, default=120, help="训练epochs次数")
    parser.add_argument("--patience", type=int, default=20, help="early stopping策略")
    parser.add_argument("--image_size", type=int, default=640, help="图片尺寸")
    parser.add_argument("--batch_size", type=int, default=64, choices=[16, 32, 48, 64], help="训练batch大小")
    parser.add_argument("--lr", type=float, default=2.2e-3, help="训练lr学习率")
    parser.add_argument("--lr_factor", type=float, default=1.0, help="OneCycleLR中max_lr缩放系数")
    parser.add_argument("--weight_decay", type=float, default=3e-4, help="权重衰减（L2正则化强度）")
    parser.add_argument("--proj_dims", type=int, default=256, choices=[128, 256], help="投影头特征维度")
    parser.add_argument("--dropout_r", type=float, default=0.2, help="投影头dropout率")
    parser.add_argument("--clr_version", type=str, default="v2", choices=["v1", "v2"], help="clr版本选择")
    parser.add_argument("--temperature", type=float, default=0.18, help="NTXentLoss函数参数")
    parser.add_argument("--w_proj", type=float, default=1.0, help="NTXentLoss权重")
    parser.add_argument("--margin", type=float, default=0.2, help="TripletLoss函数参数")
    parser.add_argument("--trip_mode", type=str, default="semi", choices=["semi", "hard", "normal"], help="TripletLoss函数参数")
    parser.add_argument("--w_trip", type=float, default=1.0, help="TripletLoss权重")
    # parser.add_argument("--sigma", type=float, default=0.6182, help="ForegroundAttendedLoss函数参数")
    parser.add_argument("--sigma", type=float, default=0.3926, help="ForegroundAttendedLoss函数参数")
    # parser.add_argument("--sigma", type=float, default=0.2123, help="ForegroundAttendedLoss函数参数")
    parser.add_argument("--w_fgrd", type=float, default=1.0, help="ForegroundAttendedLoss权重")
    parser.add_argument("--use_depthwise", action='store_true', help="多尺度特征是否用depthwise")
    parser.add_argument("--fusion_mode", type=str, default="concat", choices=["concat", "weight", "attention"], help="多尺度特征融合模式")
    parser.add_argument("--bank_size", type=int, default=4096, help="训练bank容量大小(256*batch_size)")
    parser.add_argument("--bank_sample_size", type=int, default=320, help="训练bank容量采样(8*batch_size)")
    parser.add_argument("--bank_momentum", type=float, default=0.99, help="动量编码器更新参数")
    parser.add_argument("--output_dir", type=str, default="clr_train", help="训练输出目录")
    parser.add_argument("--dir_suffix", type=str, default="", help="预训练权重目录后缀")
    parser.add_argument("--visualizer", action='store_true', help="是否输出backbone可视化结果")
    parser.add_argument("--no_amp", action='store_true', help="禁用混合精度训练 (默认开启)")
    return parser

if __name__ == '__main__':
    args = args_parser().parse_args()
    set_seed(args.seed)
    """
    # clr_classify训练命令
    python yolov8_clr_train.py --task "classify" --config "yolov8m-cls.yaml" --pretrain "yolov8m-cls.pt"
    # clr_detect训练命令
    python yolov8_clr_train.py --task "detect" --config "yolov8m.yaml" --pretrain "yolov8m.pt"
    """
    # 参数校正目录
    args.image_dir = Path(make_abs_path("datasets")).joinpath(args.task).joinpath(args.image_dir)
    args.config = Path(make_abs_path("models")).joinpath(args.config)
    args.pretrain = Path(make_abs_path("pretrains")).joinpath(args.task).joinpath(args.pretrain)
    args.output_dir = Path(make_abs_path("runs")).joinpath(args.task).joinpath(args.output_dir)
    # clr_model训练
    print("[RUN-args]:", args)
    log_path = args.output_dir.joinpath("run.log")
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "w", errors="ignore", encoding="utf-8") as log:
        log.write(f"{args}")
    # 开始训练
    clr_model_train(args, pre_train)
    # 训练结果目录调整名字
    if args.dir_suffix and args.dir_suffix != "":
        new_dir = args.output_dir.with_name(args.output_dir.name + f"_{args.dir_suffix}")
        os.rename(args.output_dir, new_dir)
    # 可视化backbone特征提取
    release_memory()
    if args.visualizer:
        import sys
        import subprocess
        subprocess.run([
            sys.executable,  # 当前解释器路径
            "yolov8_backbone_visualizer.py",
            "--task", args.task,
            "--proj_dims", str(args.proj_dims),
            "--clr_version", args.clr_version,
            "--dir_suffix", args.dir_suffix,
            "--save_only",
        ], shell=True, check=True)
    # 结束clr训练
