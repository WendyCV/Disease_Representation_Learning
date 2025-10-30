import sys
import os.path as osp
import gc
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_abs_path(fn):
    return osp.abspath(osp.join(osp.dirname(__file__), fn))

base_path = make_abs_path("")

if base_path not in sys.path:
    sys.path.insert(0, base_path)

# 替换为你本地 ultralytics 源码绝对路径
ultralytics_path = make_abs_path("ultralytics")

if ultralytics_path not in sys.path:
    sys.path.insert(0, ultralytics_path)


def calc_sigma_normalized(feature_map_size, center_size, edge_value):
    assert 0 < edge_value < 1, "edge_value 必须在0和1之间"
    radius = math.floor((center_size - 1) / 2)
    sigma_squared = - (radius ** 2) / (2 * math.log(edge_value))
    sigma = math.sqrt(sigma_squared)
    sigma_norm = sigma / feature_map_size
    return sigma_norm


def get_batch_center_weight_map(centers, height, width, sigma=1.0, device=None):
    """
    生成 batch 的高斯中心热图
    centers: Tensor[B, 2] (normalized x, y in [0,1])
    returns: Tensor[B, 1, H, W]
    """
    B = centers.size(0)
    x_range = torch.linspace(0, 1, width, device=device)
    y_range = torch.linspace(0, 1, height, device=device)
    yy, xx = torch.meshgrid(y_range, x_range, indexing="ij")  # [H, W]
    grid = torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]

    centers = centers.view(B, 2, 1, 1)  # [B, 2, 1, 1]
    dist = ((grid - centers) ** 2).sum(dim=1)  # [B, H, W]
    
    heatmaps = torch.exp(-dist / (2 * sigma ** 2))  # Gaussian
    return heatmaps.unsqueeze(1)  # [B, 1, H, W]


def get_center_weight_map(height, width, sigma=2.0, k_size=1):
    # 生成网格坐标
    grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='xy')
        
    # 计算每个位置到中心的距离
    center_x, center_y = width // 2, height // 2
    distance_map = (grid_x - center_x)**2 + (grid_y - center_y)**2
        
    # 计算高斯权重
    max_distance = torch.max(distance_map)
    sigma = sigma * (max_distance / 2)
    gaussian_map = torch.exp(-distance_map / (2 * sigma**2))
        
    # 扩展为[1, 1, height, width]
    gaussian_map = gaussian_map.unsqueeze(0).unsqueeze(0)  # 扩展为4D张量

    # 如果需要平滑处理
    if k_size > 1: # 这里使用卷积进行平滑，如果需要更平滑的效果可以调整卷积核的大小
        kernel = torch.ones(k_size, k_size).to(gaussian_map.device) / (k_size * k_size)  # 均值滤波核
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # 扩展为 [1, 1, k_size, k_size]
        gaussian_map = F.conv2d(gaussian_map, kernel, padding=k_size // 2)

    # 归一化
    normalized_gaussian_map = (gaussian_map - gaussian_map.min()) / (gaussian_map.max() - gaussian_map.min())
    return normalized_gaussian_map


def get_center_of_mask(masks: torch.Tensor) -> list:
    """
    计算 batch 中每个 mask 的重心（质心）坐标。

    参数：
        masks (Tensor): 大小为 (B, 1, H, W)，取值范围为 0 或 1（或概率）

    返回：
        centers (list of tuple): 每个 mask 对应的重心坐标 (center_y, center_x)
    """
    B, _, H, W = masks.shape
    device = masks.device

    # 生成坐标网格
    y_coords = torch.arange(0, H, device=device).view(H, 1).expand(H, W)
    x_coords = torch.arange(0, W, device=device).view(1, W).expand(H, W)

    centers = []
    for i in range(B):
        mask = masks[i, 0]  # 当前样本的 mask，shape = (H, W)
        area = mask.sum()  # 有效区域像素数
        if area == 0:
            # 如果是空 mask，默认中心为图像中心
            center_y = (H - 1) / 2
            center_x = (W - 1) / 2
        else:
            # 计算重心坐标
            center_y = (mask * y_coords).sum() / area
            center_x = (mask * x_coords).sum() / area
        centers.append((center_y.item(), center_x.item()))
    return centers


def cosine_decay_alpha(epoch, max_epochs, max_alpha, min_alpha=0.0):
    """
    计算 alpha 的余弦衰减值（用于动态调整 loss 权重等）

    参数：
        epoch (int): 当前 epoch
        max_epochs (int): 总训练轮数
        max_alpha (float): 初始 alpha 值（衰减前）
        alpha_min (float): 最小 alpha 值（衰减下限）

    返回：
        alpha (float): 当前 epoch 下的 alpha 值
    """
    cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / max_epochs))
    alpha = min_alpha + (max_alpha - min_alpha) * cosine_decay
    return alpha


def exp_decay_alpha(proj_loss, max_alpha=1.0, min_alpha=0.1, proj_center=0.25, proj_scale=5.0):
    exp_input = proj_scale * (proj_loss - proj_center)
    proj_weight = 1 / (1 + math.exp(-exp_input))
    alpha = min_alpha + (max_alpha - min_alpha) * proj_weight
    return alpha


def linear_decay_alpha(proj_loss, proj_loss_range=(0.1, 3.0), max_alpha=1.0, min_alpha=0.05):
    proj_min_loss, proj_max_loss = proj_loss_range
    proj_loss = max(proj_min_loss, min(proj_loss, proj_max_loss))
    weight = (proj_loss - proj_min_loss) / (proj_max_loss - proj_min_loss)
    alpha = min_alpha + (max_alpha - min_alpha) * weight
    return alpha


import cv2 
import numpy as np
from scipy.fftpack import dct, idct, dctn, idctn


def fft_saliency(img_np):
    """
    使用频域分析得到显著性图
    Args:
        img_np: [H, W, 3] - numpy 图像，范围 [0,255]
    Returns:
        saliency: [H, W] - 显著性图
    """
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    img_gray = np.float32(img_gray)

    dft = cv2.dft(img_gray, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    mag = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    mag = np.log(mag + 1)
    saliency = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)

    return saliency


def dct_saliency(img_np):
    """
    使用离散余弦变换提取图像的显著性区域
    Args:
        img_np: [H, W, 3] - numpy 图像（0~255）
    Returns:
        saliency_map: [H, W] - 归一化显著性图
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # 1. DCT
    dct_img = dct(dct(gray.T, norm="ortho").T, norm="ortho")

    # 2. 抑制低频部分（中心）
    h, w = dct_img.shape
    mask = np.ones((h, w), dtype=np.float32)
    center_h, center_w = h // 2, w // 2
    mask[center_h - 5:center_h + 5, center_w - 5:center_w + 5] = 0
    dct_img *= mask

    # 3. 逆 DCT
    idct_img = idct(idct(dct_img.T, norm="ortho").T, norm="ortho")

    # 4. 标准化
    saliency = cv2.normalize(np.abs(idct_img), None, 0, 1, cv2.NORM_MINMAX)

    return saliency

import pywt

def wavelet_saliency(img_np, method="haar"):
    """
    使用小波变换提取图像的显著性区域
    Args:
        img_np: [H, W, 3] - numpy 图像（0~255）
    Returns:
        saliency_map: [H, W] - 归一化显著性图
    """
    # 将图像转换为灰度图
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 1. 离散小波变换 (DWT)，使用 Haar 小波 (db1)
    coeffs = pywt.dwt2(gray, method)  # 'haar' 可以替换为其他小波，如 'db1', 'db2' 等

    # 2. 分解 DWT 输出为 4 个子带（LL, LH, HL, HH）
    LL, (LH, HL, HH) = coeffs

    # 3. 结合低频和高频部分
    combined = np.abs(LL) #+ np.abs(LH) + np.abs(HL) + np.abs(HH)

    # 4. 逆小波变换（IDWT）重构图像
    saliency = pywt.idwt2((combined, (np.zeros_like(LH), np.zeros_like(HL), np.zeros_like(HH))), method)

    # 5. 标准化显著性图
    saliency = cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX)

    return saliency


def dct_saliency_lab(img_np):
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    channel = lab[:, :, 1]  # A 通道更能代表色彩差异
    dct = cv2.dct(np.float32(channel))
    dct_log = np.log(np.abs(dct) + 1)
    saliency = cv2.idct(dct_log)
    saliency = np.abs(saliency)
    saliency = cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX)
    return saliency


def spectral_residual_saliency(img_np):
    sal = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency = sal.computeSaliency(img_np)
    if not success:
        return None
    saliency = cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX)
    return saliency  # float map in [0,1]


def fine_grained_saliency(img_np):
    sal = cv2.saliency.StaticSaliencyFineGrained_create()
    success, saliency = sal.computeSaliency(img_np)
    if not success:
        return None
    saliency = cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX)
    return saliency


def frequency_tuned_saliency(img_np):
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    mean_lab = np.mean(lab.reshape(-1, 3), axis=0)
    saliency = np.linalg.norm(lab - mean_lab, axis=2)
    saliency = cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX)
    return saliency


def otsu_threshold(saliency):
    saliency_255 = (saliency * 255).astype(np.uint8)
    _, threshold = cv2.threshold(saliency_255, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold / 255.0


def percentile_threshold(saliency, percentile=85):
    return np.percentile(saliency, percentile)


def generate_heatmap_from_feature(feat_map, method="mean"):
    """
    从 backbone 输出特征生成 heatmap
    Args:
        feat_map: [B, C, H, W] - 输入特征图
        method: "mean" or "max" - 通道聚合方式
    Returns:
        heatmap: [B, 1, H, W] - 标准化后的热力图
    """
    if method == "mean":
        heatmap = feat_map.mean(dim=1, keepdim=True)
    elif method == "max":
        heatmap, _ = feat_map.max(dim=1, keepdim=True)
    else:
        raise ValueError("method must be 'mean' or 'max'")

    # 标准化热力图到 [0, 1]
    heatmap = (heatmap - heatmap.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]) / \
              (heatmap.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)

    return heatmap


def invariance_map(feat1, feat2):
    """
    输入两个视角下的特征图，计算不变性 heatmap
    feat1, feat2: [B, C, H, W]
    """
    B, C, H, W = feat1.size()
    feat1_flat = F.normalize(feat1.view(B, C, -1), dim=1)  # [B, C, HW]
    feat2_flat = F.normalize(feat2.view(B, C, -1), dim=1)  # [B, C, HW]

    sim = torch.einsum("bcn,bcm->bnm", feat1_flat, feat2_flat)  # [B, HW, HW]
    sim_diag = sim[:, torch.arange(H * W), torch.arange(H * W)]  # [B, HW]

    sim_map = sim_diag.view(B, 1, H, W)
    sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-8)
    return sim_map


def center_coords_of_masks(masks, device):
    _, _, H, W = masks.shape
    gt_coords = get_center_of_mask(masks=masks) #list[(c_y,c_x)]
    gt_coords = torch.tensor(gt_coords, dtype=torch.float32, device=device)
    gt_coords = gt_coords[:, [1, 0]]
    gt_coords[:, 0] /= W
    gt_coords[:, 1] /= H
    # 返回mask质心计算结果
    return gt_coords


def foreground_attended_loss(gt_coords, attn_maps, sigma=0.618, penalty_weight=0.5):
    B, _, H, W = attn_maps.shape

    # -----------------------------
    # 生成 ground-truth 高斯注意力图
    # -----------------------------
    gt_attn_maps = get_batch_center_weight_map(gt_coords, H, W, sigma=sigma, device=attn_maps.device)

    # -----------------------------
    # v1：主 MSE 均方差 Loss
    # -----------------------------
    # attn_loss = F.mse_loss(attn_maps, gt_attn_maps, reduction='mean')

    # -----------------------------
    # v2：主 KL-Divergence 分布匹配 Loss
    # -----------------------------
    pred_log = F.log_softmax(attn_maps.view(B, -1), dim=1)     # 模型输出分布
    gt_prob = F.softmax(gt_attn_maps.view(B, -1), dim=1)    # GT 分布
    attn_loss = F.kl_div(pred_log, gt_prob, reduction='batchmean')

    # -----------------------------
    # 添加反高斯惩罚 Loss（用于压制非中心区域注意力）
    # -----------------------------
    penalty_mask = (1.0 - gt_attn_maps)                  # 反高斯掩码，中心低，外围高
    outside_penalty = (attn_maps * penalty_mask).mean()  # 对应元素相乘再取 mean

    # -----------------------------
    # 合并 loss
    # -----------------------------
    total_loss = attn_loss + penalty_weight * outside_penalty
    return total_loss


# 1. NTXent Loss
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1, reduction='mean', train_able=False):
        super(NTXentLoss, self).__init__()
        if train_able:
            self.temperature_delt = 0.06
            self.temperature_min, self.temperature_max = (temperature-self.temperature_delt), (temperature+self.temperature_delt)
            temperature = _init_logit_(temperature, self.temperature_min, self.temperature_max)
            self.temperature_gain = 1.0
            self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32), requires_grad=True)
        else:
            self.temperature = temperature
        self.reduction = reduction

    def _neg_filter_(self, z, negatives):
        # === 计算 z 和 negatives 之间的余弦相似度 ===
        cos_sim = torch.matmul(z, negatives.T)  # [2B, N] 计算余弦相似度

        # 获取最大的 n 个相似度（最相似的负样本）并删除
        k_n = min(int(z.shape[0] / 2), negatives.size(0))  # 每个 batch 中需要去除的负样本数
        _, top_n_idx = torch.topk(cos_sim, k=k_n, dim=1, largest=True, sorted=False)  # [2B, n]

        # 将top_n_idx展平并去重
        unique_remove_idx = torch.unique(top_n_idx.flatten())  # 也可以使用.view(-1)

        # 创建一个全1的布尔掩码，形状为 [N]
        mask = torch.ones(negatives.shape[0], dtype=torch.bool, device=z.device)

        # 将要去除的索引位置设为0
        mask[unique_remove_idx] = False

        # 使用掩码来筛选负样本
        filtered_negatives = negatives[mask]  # 形状为 [N', D]

        return filtered_negatives
    
    def _temperature(self):
        if isinstance(self.temperature, nn.Parameter):
            return self.temperature_min + (self.temperature_max - self.temperature_min) * torch.sigmoid(self.temperature_gain * self.temperature)
        else:
            return self.temperature

    def forward(self, z1, z2, samples=None):
        batch_size = z1.size(0)
        device = z1.device
        temperature = self._temperature()

        # [2B, D]
        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, p=2, dim=1)

        # [2B, 2B] 相似度矩阵
        sim_matrix = torch.matmul(z, z.T) / temperature

        # 构建正样本索引
        labels = torch.arange(batch_size, device=device)
        pos_idx = torch.cat([labels + batch_size, labels])
        # 提取正样本相似度，shape [2B, 1]
        pos_sim = torch.sum(z * z[pos_idx], dim=1, keepdim=True) / temperature  # [2B, 1]

        # mask：排除正样本和自己
        mask = torch.ones_like(sim_matrix, dtype=torch.bool)
        mask[torch.arange(2 * batch_size), torch.arange(2 * batch_size)] = False
        mask[torch.arange(2 * batch_size), pos_idx] = False
        # 负样本相似度
        neg_sim_original = sim_matrix.masked_select(mask).view(2 * batch_size, -1)  # [2B, N]
        
        # 如果有样本补充负样本
        if samples is not None:
            # === memory bank 模式 ===
            samples = F.normalize(samples.detach(), p=2, dim=1)  # [N, D]
            samples = self._neg_filter_(z, samples)
            neg_sim_samples  = torch.matmul(z, samples.T) / temperature  # [2B, N']

            # 负样本合并即使得 neg_sim 的维度变为 [2B, N + N']
            neg_sim = torch.cat([neg_sim_original, neg_sim_samples], dim=1)
        else:
            neg_sim = neg_sim_original

        # 数值稳定技巧：max subtraction (log-sum-exp trick)
        with torch.no_grad():
            max_val = torch.max(torch.cat([pos_sim, neg_sim], dim=1), dim=1, keepdim=True)[0]

        # 注意：这里确保即使在 autocast 下仍用 float32 做exp/log计算
        pos_exp = torch.exp((pos_sim - max_val).float())
        neg_exp = torch.exp((neg_sim - max_val).float())
        denominator = neg_exp.sum(dim=1, keepdim=True) + pos_exp
        log_prob = torch.log(pos_exp / (denominator + 1e-8))

        loss = -log_prob.squeeze()

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def _init_logit_(val, val_min, val_max):
    ratio = (val - val_min) / (val_max - val_min)
    ratio = max(min(ratio, 0.9999), 1e-8)
    return math.log(ratio / (1 - ratio))

# 2. Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=0.2, reduction='mean', use_cosine=True, hard_mining='semi', train_able=False):
        super(TripletLoss, self).__init__()
        if train_able:
            self.margin_delta = 0.04
            self.margin_min, self.margin_max = (margin-self.margin_delta), (margin+self.margin_delta)
            margin = _init_logit_(margin, self.margin_min, self.margin_max)
            self.margin_gain = 1.0
            self.margin = nn.Parameter(torch.tensor(margin, dtype=torch.float32), requires_grad=True)
        else:
            self.margin = margin
        self.reduction = reduction
        self.use_cosine = use_cosine
        self.hard_mining = hard_mining

    @staticmethod
    def _l2_normalize(x, eps=1e-8):
        return x / (x.norm(p=2, dim=1, keepdim=True) + eps)

    def _pairwise_sim(self, A, B):
        # A: [B, D], B: [N, D] -> sim: [B, N]
        A = self._l2_normalize(A.float())
        B = self._l2_normalize(B.float())
        return A @ B.t().to(dtype=A.dtype)  
    
    def _distance(self, X, Y):
        if self.use_cosine:
            sim = self._pairwise_sim(X, Y)            # [B, B] 或 [B, 1]
            return 1.0 - torch.diag(sim)              # 返回 [B]
        else:
            return F.pairwise_distance(X.float(), Y.float(), p=2).to(X.dtype)     # 返回 [B]

    def _select_negatives(self, anchor, bank, margin, pos_idx=None, pos_sim=None):
        with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
            sim = self._pairwise_sim(anchor.detach(), bank.detach())  # [B, N]

            # 如果提供了正样本的列索引，把它们屏蔽掉（避免选到正样本）
            if pos_idx is not None:
                B = anchor.size(0)
                ar = torch.arange(B, device=anchor.device)
                neg_inf = torch.tensor(float('-inf'), device=sim.device, dtype=sim.dtype)
                sim[ar, pos_idx[:, 0]] = neg_inf  # 屏蔽 z1 正样本
                sim[ar, pos_idx[:, 1]] = neg_inf  # 屏蔽 z2 正样本

            if self.hard_mining == 'hard':
                idx = torch.argmax(sim, dim=1)           # [B]
            elif self.hard_mining == 'semi':
                assert pos_sim is not None, "semi 模式需要提供pos_sim"

                # semi-hard 条件： sim_p - margin < sim_n < sim_p
                lower = (pos_sim - margin).unsqueeze(1)  # [B,1]
                upper = pos_sim.unsqueeze(1)             # [B,1]
                mask = (sim > lower) & (sim < upper)     # [B, N]

                # 若某些样本没有符合条件的 semi-hard 负样本，则回退到 top-k 区间内挑选
                idx_list = []
                k = min(2 * anchor.size(0), bank.size(0))
                _, topk_idx = torch.topk(sim, k=k, dim=1, largest=True, sorted=False)

                for b in range(anchor.size(0)):
                    cand = torch.nonzero(mask[b], as_tuple=False).squeeze(1)
                    if cand.numel() > 0:
                        # 可按分位数挑一个（避免最硬/最软），比如 70% 分位
                        q = int(0.7 * cand.numel())
                        pick = cand[q] if q < cand.numel() else cand[-1]
                    else:
                        # 回退：从 top-k（排除最硬的前1个）里随机挑
                        tk = topk_idx[b]
                        if tk.numel() > 1:
                            pick = tk[torch.randint(low=1, high=tk.numel(), size=(1,)).item()]
                        else:
                            pick = tk[0]
                    idx_list.append(pick)
                idx = torch.stack(idx_list, dim=0)
            else:
                k = int(min(2 * anchor.size(0), bank.size(0)))
                _, topk_idx = torch.topk(sim, k=k, dim=1, largest=True, sorted=False)  # [B, k]
                rnd = torch.randint(low=0, high=k, size=(anchor.size(0),), device=anchor.device)
                idx = topk_idx[torch.arange(anchor.size(0), device=anchor.device), rnd]

        neg = bank[idx].detach()
        return neg
    
    def _margin(self):
        if isinstance(self.margin, nn.Parameter):
            return self.margin_min + (self.margin_max - self.margin_min) * torch.sigmoid(self.margin_gain * self.margin)
        else:
            return self.margin
    
    def forward(self, anchor, z1, z2, samples=None):
        """
        anchor, z1, z2: [B, D]
        samples: [N, D] (memory bank)，若为 None 则用 batch 内部其它样本做负样本
        """
        B = anchor.size(0)
        margin = self._margin()

        # 1) 正样本距离
        z1_pos_dist = self._distance(anchor, z1) # 计算z1正样本距离：anchor 和 z1 # [B]
        z2_pos_dist = self._distance(anchor, z2) # 计算z2正样本距离：anchor 和 z2 # [B]

        # 2) 负样本池
        if samples is None or samples.numel() <= 0:
            samples = torch.cat([z1.detach(), z2.detach()], dim=0) # [2B, D]
            # 对于第 i 个 anchor，正样本列索引是 i（对应 z1[i]）和 i+B（对应 z2[i]）
            pos_idx = torch.stack([torch.arange(B, device=anchor.device),
                torch.arange(B, device=anchor.device) + B], dim=1)  # [B,2]
        else:
            pos_idx = None

        if self.hard_mining == "semi":
            # 计算正样本相似度 sim_p（与 margin 配合成 semi-hard 带宽）
            with torch.no_grad():
                pos_sim_z1 = self._pairwise_sim(anchor, z1).diagonal()  # [B]
                pos_sim_z2 = self._pairwise_sim(anchor, z2).diagonal()  # [B]
                pos_sim = 0.5 * (pos_sim_z1 + pos_sim_z2)               # [B] 稍微平滑
        else:
            pos_sim = None

        # 3) 对每个 anchor 独立选负样本
        z1_neg = self._select_negatives(anchor, samples, margin=margin, pos_idx=pos_idx, pos_sim=pos_sim)  # [B, D]
        z2_neg = self._select_negatives(anchor, samples, margin=margin, pos_idx=pos_idx, pos_sim=pos_sim)  # [B, D]

        # 4) 负样本距离
        z1_neg_dist = self._distance(anchor, z1_neg) # 计算z1负样本距离：anchor 和 z1_neg # [B]
        z2_neg_dist = self._distance(anchor, z2_neg) # 计算z2负样本距离：anchor 和 z2_neg # [B]

        # 计算 Triplet Loss
        z1_loss = torch.relu(z1_pos_dist - z1_neg_dist + margin)  # 对每个样本的差值进行ReLU处理
        z2_loss = torch.relu(z2_pos_dist - z2_neg_dist + margin)  # 对每个样本的差值进行ReLU处理
        loss = 0.5 * (z1_loss + z2_loss)
        
        if self.reduction == 'mean':
            return loss.mean()  
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# 3. Center Loss（需要类别标签，无法自监督）
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device='cpu'):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
    
    def _apply_update_centers_(self, z, labels, alpha):
        with torch.no_grad():
            for class_idx in range(self.centers.size(0)):
                class_mask = (labels == class_idx)
                class_features = z[class_mask]
                
                if class_features.size(0) > 0:
                    class_new = class_features.mean(dim=0)
                    self.centers[class_idx] = alpha*self.centers[class_idx] + (1-alpha)*class_new

    def forward(self, z, labels, alpha=0.5):
        l_centers = self.centers[labels]
        dist = torch.sum((z - l_centers) ** 2, dim=1)
        self._apply_update_centers_(z, labels, alpha)
        return torch.mean(dist)


def release_memory():
    torch.cuda.empty_cache()
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()


def to_cmd_args(args, exclude=[]):
    cmd_args = []
    exclude_set = set(exclude)
    for key, value in vars(args).items():
        if key in exclude_set:
            pass
        elif isinstance(value, bool):
            # 如果是 False，就跳过（可按需保留）
            if value: cmd_args.append(f'--{key}')
        elif value is not None:
            cmd_args.extend([f'--{key}', str(value)])
    return cmd_args


def poly2bbox(image_in, label_in, out_dir):
    from pathlib import Path
    image_dir = Path(image_in)
    label_dir = Path(label_in)
    label_out = Path(out_dir)
    label_out.mkdir(parents=True, exist_ok=True)
    # 支持常见图片后缀
    img_suf = {".jpg", ".jpeg", ".png"}
    img_paths = sorted([p for p in image_dir.glob("*") if p.suffix.lower() in img_suf])
    for img_path in img_paths:
        stem = img_path.stem
        txt_in = label_dir.joinpath( f"{stem}.txt")
        txt_out = label_out.joinpath( f"{stem}.txt")
        # 没有标签文件
        if not txt_in.exists(): continue
        # 读取标签文件
        with open(txt_in, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        # 转换并写出
        is_covert = False
        with open(txt_out, "w", encoding="utf-8") as f:
            for line in lines:
                toks = line.strip().split()
                if len(toks) < 5: continue
                if len(toks) == 5: # cls cx cy w h
                    bbbox_line = line
                elif len(toks) >= 7 and len(toks) % 2 == 1: # cls x1 y1 x2 y2 x3 y3 ...
                    cls_id = int(float(toks[0]))
                    vals = list(map(float, toks[1:]))
                    xs = np.array(vals[0::2], dtype=np.float32)
                    ys = np.array(vals[1::2], dtype=np.float32)
                    x_min, x_max = xs.min(), xs.max()
                    y_min, y_max = ys.min(), ys.max()
                    # 裁界
                    x_min = float(np.clip(x_min, 0.0, 1.0))
                    x_max = float(np.clip(x_max, 0.0, 1.0))
                    y_min = float(np.clip(y_min, 0.0, 1.0))
                    y_max = float(np.clip(y_max, 0.0, 1.0))

                    w = max(0.0, x_max - x_min)
                    h = max(0.0, y_max - y_min)
                    cx = float(np.clip((x_min + w / 2.0), 0.0, 1.0))
                    cy = float(np.clip((y_min + h / 2.0), 0.0, 1.0))

                    bbbox_line = f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

                    is_covert = True
                else:
                    continue
                # 写文件
                f.write(f"{bbbox_line}\n")
                f.flush()
        if is_covert:
            print(f"{img_path}")


def graft_detect_header(from_model, to_model, save_path):
    from ultralytics.nn.modules.head import Detect
    def _find_detect(model):
        for name, module in model.named_modules():
            if isinstance(module, Detect):
                return name, module
        raise RuntimeError("Detect layer not found!")

    from_name, from_head = _find_detect(from_model.model)
    to_name, to_head = _find_detect(to_model.model)

    print(f"[INFO] grafting from '{from_name}' → '{to_name}'")

    fsd = from_model.model.state_dict()
    tsd = to_model.model.state_dict()

    fpre, tpre = from_name + ".", to_name + "."
    count = 0

    # 只复制 Detect 层参数
    for k, v in fsd.items():
        if k.startswith(fpre):
            new_k = k.replace(fpre, tpre)
            if new_k in tsd and tsd[new_k].shape == v.shape:
                tsd[new_k] = v.clone()
                count += 1
            else:
                print(f"[WARN] Skip {k} -> {new_k} (shape mismatch)")
    if count == 0:
        raise RuntimeError("No Detect head parameters copied!")

    to_model.model.load_state_dict(tsd, strict=False)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    to_model.save(str(save_path))
    print(f"[OK] Grafted head saved to: {save_path}")
    return to_model, save_path

import torchvision.transforms.v2.functional as VF
from torchvision.transforms import InterpolationMode

class ResizeAndPadToSquare(nn.Module):
    def __init__(self, long_size, interpolation=InterpolationMode.BILINEAR, antialias=True, fill=(114, 114, 114)):
        super(ResizeAndPadToSquare, self).__init__()
        self.long_size = long_size
        self.interp = interpolation
        self.antialias = antialias
        self.fill = fill

    def forward(self, img):
        # Step 1: 获取原图尺寸
        h, w = VF.get_size(img)
        s = self.long_size / max(h, w)
        new_h, new_w = int(round(h * s)), int(round(w * s))

        # Step 2: 等比例缩放
        img = VF.resize(img, (new_h, new_w), interpolation=self.interp, antialias=self.antialias)

        # Step 3: 计算填充（上下左右）
        pad_top = (self.long_size - new_h) // 2
        pad_bottom = self.long_size - new_h - pad_top
        pad_left = (self.long_size - new_w) // 2
        pad_right = self.long_size - new_w - pad_left

        # Step 4: 使用 F.pad 填充（顺序为 [left, top, right, bottom]）
        img = VF.pad(img, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=self.fill)

        return img
