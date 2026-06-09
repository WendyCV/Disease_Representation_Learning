from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F



# =========================================================
# dynamic scale gating
# =========================================================
class RPDDynamicScaleWeightsGating(nn.Module):
    def __init__(self, in_channels_list=(128, 256, 512), base_weights=(0.30, 0.40, 0.30), alpha=0.25):
        super().__init__()

        self.in_channels_list = list(in_channels_list)
        self.alpha = float(alpha)

        # 对各个尺度进行全局 pooling 并投影到统一低维空间
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(1) for _ in self.in_channels_list])
        self.fcs = nn.ModuleList([nn.Linear(ch, 64) for ch in self.in_channels_list])

        # 保持你原来的 gate 结构，不改 fc1/fc2
        self.gate = nn.Sequential(
            nn.Linear(64 * len(self.in_channels_list), 32),
            nn.ReLU(),
            nn.Linear(32, len(self.in_channels_list)),
            nn.Softmax(dim=-1),
        )

        # 注册 base_weights，保证它能跟随模型到 cuda / cpu
        base = torch.tensor(base_weights, dtype=torch.float32)
        base = base / base.sum().clamp(min=1e-6)
        self.register_buffer("base_weights", base)

    def forward(self, student_feats_list, return_dynamic=False):
        """
        student_feats_list: [feat_L1, feat_L2, feat_L3]

        return:
            effective_weights: [3]
        """
        batch_size = student_feats_list[0].size(0)

        pooled_vectors = []
        for i, feat in enumerate(student_feats_list):
            v = self.pools[i](feat).view(batch_size, -1)
            v = torch.relu(self.fcs[i](v))
            pooled_vectors.append(v)

        fused = torch.cat(pooled_vectors, dim=1)

        # dynamic_weights: [B, 3]
        dynamic_weights = self.gate(fused)

        # batch mean: [3]
        dynamic_weights = dynamic_weights.mean(dim=0)
        dynamic_weights = dynamic_weights / dynamic_weights.sum().clamp(min=1e-6)

        base_weights = self.base_weights.to(device=dynamic_weights.device, dtype=dynamic_weights.dtype)
        base_weights = base_weights / base_weights.sum().clamp(min=1e-6)

        # 关键：base-anchored dynamic scale
        # alpha=0.25 表示 75% 保留静态 [0.30,0.40,0.30]，25% 由动态 gate 调整
        effective_weights = ((1.0 - self.alpha) * base_weights + self.alpha * dynamic_weights)
        effective_weights = effective_weights / effective_weights.sum().clamp(min=1e-6)

        if return_dynamic:
            return effective_weights, dynamic_weights

        return effective_weights

# =========================================================
# Core SSL losses
# =========================================================
def compute_single_scale_global_contrastive_loss(
    query_embedding,
    key_embedding,
    memory_queue,
    temperature: float = 0.2,
):
    query_embedding = F.normalize(query_embedding, dim=1)
    key_embedding = F.normalize(key_embedding, dim=1)

    positive_logits = torch.einsum("bd,bd->b", query_embedding, key_embedding).unsqueeze(1)
    negative_logits = torch.einsum("bd,dk->bk", query_embedding, memory_queue.detach())

    logits = torch.cat([positive_logits, negative_logits], dim=1) / temperature
    labels = torch.zeros(query_embedding.size(0), dtype=torch.long, device=query_embedding.device)
    return F.cross_entropy(logits, labels)


def compute_multiscale_global_contrastive_loss(
    query_embeddings,
    key_embeddings,
    queues,
    temperature: float = 0.2,
):
    losses = [
        compute_single_scale_global_contrastive_loss(q, k, queue, temperature=temperature)
        for q, k, queue in zip(query_embeddings, key_embeddings, queues)
    ]
    return torch.stack(losses).mean()


def compute_single_scale_local_consistency_loss(
    query_local,
    key_local,
    query_mask=None,
    key_mask=None,
):
    """
    query_local, key_local: [B, C, H, W]

    先在空间维度上计算逐位置 cosine similarity，得到 [B, H*W]。
    如果提供 mask，则把 mask resize 到 (H, W) 后再 flatten，和 similarity 对齐。
    """
    spatial_hw = query_local.shape[-2:]  # 关键：必须在 flatten 前保存 (H, W)

    query_local = F.normalize(query_local.flatten(2), dim=1)   # [B, C, HW]
    key_local = F.normalize(key_local.flatten(2), dim=1)       # [B, C, HW]
    similarity = (query_local * key_local).sum(dim=1)          # [B, HW]

    if query_mask is None or key_mask is None:
        return 1.0 - similarity.mean()

    support_mask = ((query_mask + key_mask) > 0.5).float()
    support_mask = F.interpolate(support_mask, size=spatial_hw, mode="nearest")  # [B,1,H,W]
    support_mask = support_mask.flatten(1)                                        # [B, HW]

    valid_count = support_mask.sum(dim=1).clamp(min=1.0)
    return 1.0 - ((similarity * support_mask).sum(dim=1) / valid_count).mean()


def compute_multiscale_local_consistency_loss(
    query_locals,
    key_locals,
    query_mask=None,
    key_mask=None,
):
    losses = [
        compute_single_scale_local_consistency_loss(q_local, k_local, query_mask=query_mask, key_mask=key_mask)
        for q_local, k_local in zip(query_locals, key_locals)
    ]
    return torch.stack(losses).mean()


def compute_single_scale_position_alignment_loss(
    query_pos,
    key_pos,
    query_mask=None,
    key_mask=None,
):
    query_activation = query_pos.mean(dim=1, keepdim=True)
    key_activation = key_pos.mean(dim=1, keepdim=True)

    if query_mask is not None and key_mask is not None:
        support_mask = ((query_mask + key_mask) > 0.5).float()
        support_mask = F.interpolate(support_mask, size=query_pos.shape[-2:], mode="nearest")
        query_activation = query_activation * support_mask
        key_activation = key_activation * support_mask

    query_vector = F.normalize(query_activation.flatten(1), dim=1)
    key_vector = F.normalize(key_activation.flatten(1), dim=1)
    return 1.0 - (query_vector * key_vector).sum(dim=1).mean()


def compute_multiscale_position_alignment_loss(
    query_position_features,
    key_position_features,
    query_mask=None,
    key_mask=None,
):
    losses = [
        compute_single_scale_position_alignment_loss(q_pos, k_pos, query_mask=query_mask, key_mask=key_mask)
        for q_pos, k_pos in zip(query_position_features, key_position_features)
    ]
    return torch.stack(losses).mean()


# =========================================================
# Common helpers
# =========================================================
def _zero_like_from_available_tensors(*tensor_groups):
    """
    从若干可能为 None 的 tensor/list[tensor] 中，找一个可用 tensor，
    返回一个带正确 device/dtype 的 0 标量。
    若全都没有，退回 CPU float 标量 0。
    """
    for group in tensor_groups:
        if group is None:
            continue

        if isinstance(group, (list, tuple)):
            if len(group) > 0 and group[0] is not None:
                t = group[0]
                return t.sum() * 0.0
        elif torch.is_tensor(group):
            return group.sum() * 0.0

    return torch.tensor(0.0)


def resize_binary_mask(mask, target_hw, device=None, dtype=torch.float32):
    if mask is None:
        return None
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    mask = F.interpolate(mask, size=target_hw, mode="nearest")
    mask = (mask > 0.5).float()
    if device is not None:
        mask = mask.to(device=device)
    if dtype is not None:
        mask = mask.to(dtype=dtype)
    return mask


def normalize_heatmap(heatmap):
    flattened = heatmap.flatten(2)
    min_value = flattened.min(dim=-1, keepdim=True).values.unsqueeze(-1)
    max_value = flattened.max(dim=-1, keepdim=True).values.unsqueeze(-1)
    return (heatmap - min_value) / (max_value - min_value + 1e-6)


def normalize_scale_weights(scale_weights, num_scales: int, device, dtype):
    if scale_weights is None:
        weights = torch.ones(num_scales, device=device, dtype=dtype)

    elif torch.is_tensor(scale_weights):
        weights = scale_weights.to(device=device, dtype=dtype)

        if weights.dim() == 2:
            weights = weights.mean(dim=0)

        if weights.numel() < num_scales:
            pad_value = weights[-1:].expand(num_scales - weights.numel())
            weights = torch.cat([weights, pad_value], dim=0)
        elif weights.numel() > num_scales:
            weights = weights[:num_scales]

    else:
        weights = list(scale_weights)
        if len(weights) < num_scales:
            weights = weights + [weights[-1]] * (num_scales - len(weights))
        elif len(weights) > num_scales:
            weights = weights[:num_scales]
        weights = torch.tensor(weights, device=device, dtype=dtype)

    return weights / weights.sum().clamp(min=1e-6)


# backward-compatible alias
resolve_scale_weights = normalize_scale_weights


def weighted_spatial_average(value_map, weight_map=None, eps: float = 1e-6):
    if value_map.dim() == 3:
        value_map = value_map.unsqueeze(1)
    if weight_map is None:
        return value_map.mean()
    if weight_map.dim() == 3:
        weight_map = weight_map.unsqueeze(1)
    numerator = (value_map * weight_map).sum(dim=(1, 2, 3))
    denominator = weight_map.sum(dim=(1, 2, 3)).clamp(min=eps)
    return (numerator / denominator).mean()


def compute_channelwise_cosine_map(student_feature, teacher_feature, eps: float = 1e-6):
    student_feature = F.normalize(student_feature, dim=1, eps=eps)
    teacher_feature = F.normalize(teacher_feature, dim=1, eps=eps)
    return (student_feature * teacher_feature).sum(dim=1, keepdim=True)


def masked_average_pool(feature_map, mask, eps: float = 1e-6):
    if mask is None:
        return feature_map.mean(dim=(2, 3))
    mask = resize_binary_mask(mask, feature_map.shape[-2:], device=feature_map.device, dtype=feature_map.dtype)
    numerator = (feature_map * mask).sum(dim=(2, 3))
    denominator = mask.sum(dim=(2, 3)).clamp(min=eps)
    return numerator / denominator


def select_teacher_outputs(model_outputs, view_key: str = "q1", teacher_source: str = "snapshot"):
    if teacher_source == "snapshot":
        return model_outputs["t1"] if view_key == "q1" else model_outputs["t2"]
    if teacher_source == "online":
        return model_outputs[view_key]
    if teacher_source == "momentum":
        return model_outputs["k1"] if view_key == "q1" else model_outputs["k2"]
    raise ValueError(f"Unsupported teacher_source: {teacher_source}")


def build_teacher_focus_mask(
    teacher_position_feature,
    fg_mask=None,
    use_teacher_topk: bool = True,
    teacher_topk_ratio: float = 0.35,
    min_region_pixels: int = 16,
):
    teacher_heatmap = normalize_heatmap(teacher_position_feature.abs().mean(dim=1, keepdim=True))

    foreground_mask = resize_binary_mask(
        fg_mask,
        teacher_heatmap.shape[-2:],
        device=teacher_heatmap.device,
        dtype=teacher_heatmap.dtype,
    )
    if foreground_mask is None:
        foreground_mask = torch.ones_like(teacher_heatmap)

    if not use_teacher_topk:
        return foreground_mask

    batch_size, _, height, width = teacher_heatmap.shape
    region_mask = torch.zeros_like(teacher_heatmap)
    flat_heatmap = teacher_heatmap.flatten(2)
    flat_foreground = (foreground_mask.flatten(2) > 0.5)
    flat_region = region_mask.flatten(2)

    for batch_index in range(batch_size):
        valid_indices = torch.nonzero(flat_foreground[batch_index, 0], as_tuple=False).squeeze(1)
        if valid_indices.numel() == 0:
            valid_indices = torch.arange(height * width, device=teacher_heatmap.device)

        topk_count = max(min_region_pixels, int(valid_indices.numel() * teacher_topk_ratio))
        topk_count = min(topk_count, valid_indices.numel())

        valid_values = flat_heatmap[batch_index, 0, valid_indices]
        topk_local_indices = torch.topk(valid_values, k=topk_count, largest=True).indices
        chosen_indices = valid_indices[topk_local_indices]
        flat_region[batch_index, 0, chosen_indices] = 1.0

        if flat_region[batch_index, 0].sum() < min_region_pixels:
            flat_region[batch_index, 0, valid_indices] = 1.0

    return region_mask


# =========================================================
# Auxiliary embedding transfer
# =========================================================
def compute_cosine_embedding_loss(student_embedding, teacher_embedding, detach_teacher: bool = True):
    if detach_teacher:
        teacher_embedding = teacher_embedding.detach()
    student_embedding = F.normalize(student_embedding, dim=1)
    teacher_embedding = F.normalize(teacher_embedding, dim=1)
    return 1.0 - (student_embedding * teacher_embedding).sum(dim=1).mean()


def compute_single_scale_local_embedding_transfer(
    raw_local_embedding,
    teacher_local_embedding,
    mask=None,
    detach_teacher: bool = True,
):
    if detach_teacher:
        teacher_local_embedding = teacher_local_embedding.detach()
    region_mask = resize_binary_mask(
        mask,
        raw_local_embedding.shape[-2:],
        device=raw_local_embedding.device,
        dtype=raw_local_embedding.dtype,
    )
    cosine_map = compute_channelwise_cosine_map(raw_local_embedding, teacher_local_embedding)
    return 1.0 - weighted_spatial_average(cosine_map, region_mask)


# backward-compatible alias
compute_local_aux_embedding_transfer_single_scale = compute_single_scale_local_embedding_transfer


def compute_multiscale_aux_embedding_transfer(
    raw_global_embeddings,
    teacher_global_embeddings,
    raw_local_embeddings,
    teacher_local_embeddings,
    foreground_mask=None,
    mask=None,
    lambda_global=1.0,
    lambda_local=1.0,
    global_weight=None,
    local_weight=None,
    detach_teacher=True,
    local_scale_weights=None,
):
    """
    弱辅助 embedding transfer。

    兼容：
    - foreground_mask / mask
    - lambda_global / global_weight
    - lambda_local / local_weight
    """

    if mask is not None and foreground_mask is None:
        foreground_mask = mask
    if global_weight is not None:
        lambda_global = global_weight
    if local_weight is not None:
        lambda_local = local_weight

    zero = _zero_like_from_available_tensors(
        raw_global_embeddings,
        teacher_global_embeddings,
        raw_local_embeddings,
        teacher_local_embeddings,
    )

    if (
        raw_global_embeddings is None
        or teacher_global_embeddings is None
        or raw_local_embeddings is None
        or teacher_local_embeddings is None
    ):
        return zero, zero, zero

    if len(raw_global_embeddings) == 0 or len(teacher_global_embeddings) == 0:
        return zero, zero, zero
    if len(raw_local_embeddings) == 0 or len(teacher_local_embeddings) == 0:
        return zero, zero, zero

    global_losses = []
    for raw_global, teacher_global in zip(raw_global_embeddings, teacher_global_embeddings):
        global_losses.append(
            compute_cosine_embedding_loss(
                student_embedding=raw_global,
                teacher_embedding=teacher_global,
                detach_teacher=detach_teacher,
            )
        )
    aux_global_loss = torch.stack(global_losses).mean()

    scale_weights = normalize_scale_weights(
        scale_weights=local_scale_weights,
        num_scales=len(raw_local_embeddings),
        device=raw_local_embeddings[0].device,
        dtype=raw_local_embeddings[0].dtype,
    )

    local_losses = []
    for scale_index, (raw_local, teacher_local) in enumerate(zip(raw_local_embeddings, teacher_local_embeddings)):
        local_loss = compute_single_scale_local_embedding_transfer(
            raw_local_embedding=raw_local,
            teacher_local_embedding=teacher_local,
            mask=foreground_mask,
            detach_teacher=detach_teacher,
        )
        local_losses.append(local_loss * scale_weights[scale_index])

    aux_local_loss = torch.stack(local_losses).sum()
    aux_total_loss = lambda_global * aux_global_loss + lambda_local * aux_local_loss
    return aux_total_loss, aux_global_loss, aux_local_loss


# =========================================================
# Direct raw spatial supervision
# =========================================================
def compute_single_scale_raw_mask_loss(
    raw_feature_map,
    mask,
    foreground_background_margin: float = 0.15,
):
    if mask is None:
        return raw_feature_map.sum() * 0.0

    raw_heatmap = normalize_heatmap(raw_feature_map.abs().mean(dim=1, keepdim=True))
    foreground_mask = resize_binary_mask(
        mask,
        raw_heatmap.shape[-2:],
        device=raw_heatmap.device,
        dtype=raw_heatmap.dtype,
    )
    background_mask = 1.0 - foreground_mask

    intersection = (raw_heatmap * foreground_mask).sum(dim=(1, 2, 3))
    union = raw_heatmap.sum(dim=(1, 2, 3)) + foreground_mask.sum(dim=(1, 2, 3)) + 1e-6
    dice_loss = 1.0 - (2.0 * intersection + 1e-6) / union

    foreground_mean = (raw_heatmap * foreground_mask).sum(dim=(1, 2, 3)) / foreground_mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
    background_mean = (raw_heatmap * background_mask).sum(dim=(1, 2, 3)) / background_mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
    margin_loss = F.relu(foreground_background_margin - (foreground_mean - background_mean))
    return (dice_loss + margin_loss).mean()


def compute_multiscale_raw_mask_loss(
    raw_feature_maps,
    mask,
    scale_weights=None,
    foreground_background_margin: float = 0.15,
):
    scale_weights = normalize_scale_weights(
        scale_weights,
        len(raw_feature_maps),
        raw_feature_maps[0].device,
        raw_feature_maps[0].dtype,
    )
    losses = [
        compute_single_scale_raw_mask_loss(
            raw_feature_map,
            mask,
            foreground_background_margin=foreground_background_margin,
        ) * scale_weights[index]
        for index, raw_feature_map in enumerate(raw_feature_maps)
    ]
    return torch.stack(losses).sum()


def compute_single_scale_raw_foreground_consistency(
    raw_feature_map_1,
    raw_feature_map_2,
    mask_1=None,
    mask_2=None,
):
    pooled_1 = masked_average_pool(raw_feature_map_1, mask_1)
    pooled_2 = masked_average_pool(raw_feature_map_2, mask_2)
    pooled_1 = F.normalize(pooled_1, dim=1)
    pooled_2 = F.normalize(pooled_2, dim=1)
    return 1.0 - (pooled_1 * pooled_2).sum(dim=1).mean()


def compute_multiscale_raw_foreground_consistency_loss(
    raw_feature_maps_1,
    raw_feature_maps_2,
    mask_1=None,
    mask_2=None,
    scale_weights=None,
):
    scale_weights = normalize_scale_weights(
        scale_weights,
        len(raw_feature_maps_1),
        raw_feature_maps_1[0].device,
        raw_feature_maps_1[0].dtype,
    )
    losses = [
        compute_single_scale_raw_foreground_consistency(
            raw_feature_1,
            raw_feature_2,
            mask_1=mask_1,
            mask_2=mask_2,
        ) * scale_weights[index]
        for index, (raw_feature_1, raw_feature_2) in enumerate(zip(raw_feature_maps_1, raw_feature_maps_2))
    ]
    return torch.stack(losses).sum()


# backward-compatible alias
compute_multiscale_raw_foreground_consistency = compute_multiscale_raw_foreground_consistency_loss


# =========================================================
# Snapshot position-bias transfer
# =========================================================
def compute_single_scale_snapshot_bias_loss(
    raw_feature_map,
    teacher_position_feature,
    fg_mask=None,
    loss_type: str = "kl",
    student_temperature: float = 0.10,
    teacher_temperature: float = 0.07,
    use_teacher_topk: bool = True,
    teacher_topk_ratio: float = 0.35,
    min_region_pixels: int = 16,
    detach_teacher: bool = True,
):
    if detach_teacher:
        teacher_position_feature = teacher_position_feature.detach()

    focus_mask = build_teacher_focus_mask(
        teacher_position_feature,
        fg_mask=fg_mask,
        use_teacher_topk=use_teacher_topk,
        teacher_topk_ratio=teacher_topk_ratio,
        min_region_pixels=min_region_pixels,
    )

    raw_heatmap = normalize_heatmap(raw_feature_map.abs().mean(dim=1, keepdim=True))
    teacher_heatmap = normalize_heatmap(teacher_position_feature.abs().mean(dim=1, keepdim=True))

    if loss_type == "smooth_l1":
        diff = F.smooth_l1_loss(raw_heatmap, teacher_heatmap, reduction="none")
        return weighted_spatial_average(diff, focus_mask)

    if loss_type == "cosine":
        raw_vector = F.normalize((raw_heatmap * focus_mask).flatten(1), dim=1)
        teacher_vector = F.normalize((teacher_heatmap * focus_mask).flatten(1), dim=1)
        return 1.0 - (raw_vector * teacher_vector).sum(dim=1).mean()

    if loss_type == "kl":
        batch_size = raw_heatmap.shape[0]
        flat_raw = raw_heatmap.flatten(1)
        flat_teacher = teacher_heatmap.flatten(1)
        flat_region = focus_mask.flatten(1) > 0.5
        losses = []
        for batch_index in range(batch_size):
            valid_region = flat_region[batch_index]
            if valid_region.sum() < 2:
                valid_region = torch.ones_like(valid_region, dtype=torch.bool)
            raw_logits = flat_raw[batch_index, valid_region] / max(student_temperature, 1e-6)
            teacher_logits = flat_teacher[batch_index, valid_region] / max(teacher_temperature, 1e-6)
            raw_log_prob = F.log_softmax(raw_logits, dim=0)
            teacher_prob = F.softmax(teacher_logits, dim=0)
            losses.append(F.kl_div(raw_log_prob, teacher_prob, reduction="batchmean"))
        return torch.stack(losses).mean()

    raise ValueError(f"Unsupported snapshot bias loss type: {loss_type}")


def compute_multiscale_snapshot_bias_loss(
    raw_feature_maps,
    teacher_position_features,
    fg_mask=None,
    scale_weights=None,
    loss_type: str = "kl",
    student_temperature: float = 0.10,
    teacher_temperature: float = 0.07,
    use_teacher_topk: bool = True,
    teacher_topk_ratio: float = 0.35,
    min_region_pixels: int = 16,
    detach_teacher: bool = True,
):
    zero = _zero_like_from_available_tensors(raw_feature_maps, teacher_position_features)
    if raw_feature_maps is None or teacher_position_features is None:
        return zero
    if len(raw_feature_maps) == 0 or len(teacher_position_features) == 0:
        return zero

    scale_weights = normalize_scale_weights(
        scale_weights,
        len(raw_feature_maps),
        raw_feature_maps[0].device,
        raw_feature_maps[0].dtype,
    )
    losses = []
    for index, (raw_feature_map, teacher_position_feature) in enumerate(zip(raw_feature_maps, teacher_position_features)):
        losses.append(
            compute_single_scale_snapshot_bias_loss(
                raw_feature_map,
                teacher_position_feature,
                fg_mask=fg_mask,
                loss_type=loss_type,
                student_temperature=student_temperature,
                teacher_temperature=teacher_temperature,
                use_teacher_topk=use_teacher_topk,
                teacher_topk_ratio=teacher_topk_ratio,
                min_region_pixels=min_region_pixels,
                detach_teacher=detach_teacher,
            ) * scale_weights[index]
        )
    return torch.stack(losses).sum()



# =========================================================
# Raw Prior Distillation (RPD)
# =========================================================
def _resolve_teacher_branch_name(teacher_branch, scale_index: int):
    """teacher_branch may be a string or a per-scale list/tuple of strings."""
    if isinstance(teacher_branch, (list, tuple)):
        if len(teacher_branch) == 0:
            return "pos_feats"
        if scale_index < len(teacher_branch):
            return str(teacher_branch[scale_index])
        return str(teacher_branch[-1])
    return str(teacher_branch or "pos_feats")


def _select_raw_prior_teacher_outputs(model_outputs, view_key: str, teacher_source: str):
    """Select teacher-side outputs for RPD."""
    if teacher_source == "online":
        return model_outputs[view_key]
    if teacher_source == "momentum":
        return model_outputs["k1"] if view_key == "q1" else model_outputs["k2"]
    if teacher_source == "snapshot":
        return model_outputs["t1"] if view_key == "q1" else model_outputs["t2"]
    raise ValueError(f"Unsupported RPD teacher_source: {teacher_source}")


def _normalize_heatmap_for_rpd(heatmap, eps: float = 1e-6):
    """Normalize each sample heatmap to [0, 1]."""
    return normalize_heatmap(heatmap).clamp(0.0, 1.0)


def compute_single_scale_raw_prior_distillation_loss(
    raw_feature_map,
    teacher_feature_map,
    mask=None,
    loss_type: str = "smooth_l1",
    student_temperature: float = 0.10,
    teacher_temperature: float = 0.07,
    detach_teacher: bool = True,
    eps: float = 1e-6,
):
    """
    Raw Prior Distillation (RPD).

    Current original modes:
    - smooth_l1: symmetric heatmap regression
    - mse:       symmetric heatmap regression
    - cosine:    masked heatmap-vector cosine
    - kl:        masked spatial distribution distillation

    Added improved modes:
    - asym_smooth_l1:
        only penalizes raw_heatmap when it is lower than teacher_heatmap.
        It does not force raw_heatmap to decrease when raw is higher than teacher.
    - asym_mse:
        same asymmetric idea, using squared error on the positive deficit.

    Motivation:
    The original symmetric RPD can make raw features too close to the teacher
    position prior. This may suppress raw backbone candidate responses that are
    useful for small or weak lesions. The asymmetric version preserves candidate
    coverage while still transferring reliable teacher foreground prior.
    """
    if raw_feature_map is None or teacher_feature_map is None:
        return _zero_like_from_available_tensors(raw_feature_map, teacher_feature_map)

    if detach_teacher:
        teacher_feature_map = teacher_feature_map.detach()

    raw_heatmap = _normalize_heatmap_for_rpd(
        raw_feature_map.abs().mean(dim=1, keepdim=True),
        eps=eps,
    )
    teacher_heatmap = _normalize_heatmap_for_rpd(
        teacher_feature_map.abs().mean(dim=1, keepdim=True),
        eps=eps,
    )

    if teacher_heatmap.shape[-2:] != raw_heatmap.shape[-2:]:
        teacher_heatmap = F.interpolate(
            teacher_heatmap,
            size=raw_heatmap.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    region_mask = resize_binary_mask(
        mask,
        raw_heatmap.shape[-2:],
        device=raw_heatmap.device,
        dtype=raw_heatmap.dtype,
    )
    if region_mask is None:
        region_mask = torch.ones_like(raw_heatmap)

    # -----------------------------
    # Original symmetric RPD losses
    # -----------------------------
    if loss_type == "smooth_l1":
        diff = F.smooth_l1_loss(raw_heatmap, teacher_heatmap, reduction="none")
        return weighted_spatial_average(diff, region_mask)

    if loss_type == "mse":
        diff = (raw_heatmap - teacher_heatmap) ** 2
        return weighted_spatial_average(diff, region_mask)

    # -------------------------------------------------------
    # Improved RPD: asymmetric candidate-preserving variants
    # -------------------------------------------------------
    if loss_type == "asym_smooth_l1":
        # Only punish under-response:
        # teacher high but raw low -> loss
        # raw high but teacher low -> no loss
        #
        # margin avoids forcing raw to exactly match teacher.
        margin = 0.02

        deficit = F.relu(teacher_heatmap - raw_heatmap - margin)

        # Weight more on reliable teacher-prior positions.
        # This makes the loss focus on strong teacher foreground prior,
        # not every weak teacher activation.
        teacher_weight = teacher_heatmap.detach().clamp(0.0, 1.0)

        diff = F.smooth_l1_loss(deficit, torch.zeros_like(deficit), reduction="none")
        return weighted_spatial_average(diff, region_mask * teacher_weight)

    if loss_type == "asym_mse":
        margin = 0.02
        deficit = F.relu(teacher_heatmap - raw_heatmap - margin)
        teacher_weight = teacher_heatmap.detach().clamp(0.0, 1.0)
        diff = deficit ** 2
        return weighted_spatial_average(diff, region_mask * teacher_weight)
    
    if loss_type == "asym_l1":
        # Candidate-preserving asymmetric RPD.
        #
        # Only penalize under-response:
        # teacher high, raw low -> loss
        # raw high, teacher low -> no loss
        #
        # Compared with asym_smooth_l1, this keeps the loss magnitude usable.
        margin = 0.00

        deficit = F.relu(teacher_heatmap - raw_heatmap - margin)

        # Do not use teacher_heatmap alone as weight, otherwise weak teacher
        # regions may make the loss too small. This keeps all foreground
        # positions active while still emphasizing teacher-high regions.
        teacher_weight = 0.5 + 0.5 * teacher_heatmap.detach()

        diff = deficit
        return weighted_spatial_average(diff, region_mask * teacher_weight)

    # -----------------------------
    # Original cosine / KL options
    # -----------------------------
    if loss_type == "cosine":
        raw_vector = F.normalize((raw_heatmap * region_mask).flatten(1), dim=1, eps=eps)
        teacher_vector = F.normalize((teacher_heatmap * region_mask).flatten(1), dim=1, eps=eps)
        return 1.0 - (raw_vector * teacher_vector).sum(dim=1).mean()

    if loss_type == "kl":
        flat_raw = raw_heatmap.flatten(1)
        flat_teacher = teacher_heatmap.flatten(1)
        flat_region = region_mask.flatten(1) > 0.5
        losses = []
        for batch_index in range(raw_heatmap.shape[0]):
            valid = flat_region[batch_index]
            if valid.sum() < 2:
                valid = torch.ones_like(valid, dtype=torch.bool)

            raw_logits = flat_raw[batch_index, valid] / max(float(student_temperature), eps)
            teacher_logits = flat_teacher[batch_index, valid] / max(float(teacher_temperature), eps)

            raw_log_prob = F.log_softmax(raw_logits, dim=0)
            teacher_prob = F.softmax(teacher_logits, dim=0)

            losses.append(F.kl_div(raw_log_prob, teacher_prob, reduction="batchmean"))

        return torch.stack(losses).mean()

    raise ValueError(f"Unsupported raw prior distillation loss_type: {loss_type}")


def compute_multiscale_raw_prior_distillation_loss(
    raw_feature_maps,
    teacher_outputs,
    mask=None,
    teacher_branch="pos_feats",
    scale_weights=None,
    loss_type: str = "smooth_l1",
    student_temperature: float = 0.10,
    teacher_temperature: float = 0.07,
    detach_teacher: bool = True,
):
    """Multi-scale RPD loss with optional per-scale teacher branches."""
    zero = _zero_like_from_available_tensors(raw_feature_maps)
    if raw_feature_maps is None or teacher_outputs is None:
        return zero
    if len(raw_feature_maps) == 0:
        return zero

    scale_weights = normalize_scale_weights(
        scale_weights,
        len(raw_feature_maps),
        raw_feature_maps[0].device,
        raw_feature_maps[0].dtype,
    )

    losses = []
    for scale_index, raw_feature_map in enumerate(raw_feature_maps):
        branch_name = _resolve_teacher_branch_name(teacher_branch, scale_index)
        teacher_feature_maps = None if teacher_outputs is None else teacher_outputs.get(branch_name, None)
        if teacher_feature_maps is None or scale_index >= len(teacher_feature_maps):
            losses.append(raw_feature_map.sum() * 0.0)
            continue

        teacher_feature_map = teacher_feature_maps[scale_index]
        single_loss = compute_single_scale_raw_prior_distillation_loss(
            raw_feature_map=raw_feature_map,
            teacher_feature_map=teacher_feature_map,
            mask=mask,
            loss_type=loss_type,
            student_temperature=student_temperature,
            teacher_temperature=teacher_temperature,
            detach_teacher=detach_teacher,
        )
        losses.append(single_loss * scale_weights[scale_index])

    return torch.stack(losses).sum()


# =========================================================
# Total stage-1 loss
# =========================================================
def compute_stage1_total_loss(
    model_outputs,
    mask_view_1=None,
    mask_view_2=None,
    temperature: float = 0.2,
    base_global_weight: float = 1.0,
    base_local_weight: float = 0.3,
    base_position_weight: float = 0.1,
    use_pos: bool = True,
    use_mask: bool = True,
    aux_embedding_enabled: bool = False,
    aux_teacher_source: str = "snapshot",
    aux_embedding_weight: float = 0.10,
    aux_local_weight: float = 0.20,
    detach_teacher: bool = True,
    aux_embedding_scale_weights=None,
    raw_spatial_enabled: bool = True,
    raw_mask_weight: float = 0.25,
    raw_consistency_weight: float = 0.15,
    raw_mask_scale_weights=None,
    raw_consistency_scale_weights=None,
    foreground_background_margin: float = 0.15,
    raw_prior_distillation_enabled: bool = False,
    raw_prior_teacher_source: str = "online",
    raw_prior_teacher_branch="pos_feats",
    raw_prior_weight: float = 0.0,
    raw_prior_loss_type: str = "smooth_l1",
    raw_prior_scale_weights=None,
    raw_prior_detach_teacher: bool = True,
    raw_prior_student_temperature: float = 0.10,
    raw_prior_teacher_temperature: float = 0.07,
):
    query_view_1 = model_outputs["q1"]
    query_view_2 = model_outputs["q2"]
    key_view_1 = model_outputs["k1"]
    key_view_2 = model_outputs["k2"]
    queues = model_outputs["queues"]

    if not use_mask:
        mask_view_1 = None
        mask_view_2 = None

    loss_global_12 = compute_multiscale_global_contrastive_loss(
        query_view_1["global_embs"],
        key_view_2["global_embs"],
        queues,
        temperature=temperature,
    )
    loss_global_21 = compute_multiscale_global_contrastive_loss(
        query_view_2["global_embs"],
        key_view_1["global_embs"],
        queues,
        temperature=temperature,
    )
    loss_global = 0.5 * (loss_global_12 + loss_global_21)

    loss_local_12 = compute_multiscale_local_consistency_loss(
        query_view_1["local_embs"],
        key_view_2["local_embs"],
        query_mask=mask_view_1,
        key_mask=mask_view_2,
    )
    loss_local_21 = compute_multiscale_local_consistency_loss(
        query_view_2["local_embs"],
        key_view_1["local_embs"],
        query_mask=mask_view_2,
        key_mask=mask_view_1,
    )
    loss_local = 0.5 * (loss_local_12 + loss_local_21)

    if use_pos:
        loss_position_12 = compute_multiscale_position_alignment_loss(
            query_view_1["pos_feats"],
            key_view_2["pos_feats"],
            query_mask=mask_view_1,
            key_mask=mask_view_2,
        )
        loss_position_21 = compute_multiscale_position_alignment_loss(
            query_view_2["pos_feats"],
            key_view_1["pos_feats"],
            query_mask=mask_view_2,
            key_mask=mask_view_1,
        )
        loss_position = 0.5 * (loss_position_12 + loss_position_21)
    else:
        loss_position = query_view_1["global_embs"][0].sum() * 0.0

    zero_for_aux = _zero_like_from_available_tensors(
        query_view_1.get("raw_feats", None),
        query_view_1.get("global_embs", None),
        query_view_1.get("raw_global_embs", None),
    )

    if aux_embedding_enabled:
        teacher_view_1 = select_teacher_outputs(model_outputs, view_key="q1", teacher_source=aux_teacher_source)
        teacher_view_2 = select_teacher_outputs(model_outputs, view_key="q2", teacher_source=aux_teacher_source)

        student_raw_global_1 = query_view_1.get("raw_global_embs", None)
        student_raw_global_2 = query_view_2.get("raw_global_embs", None)
        student_raw_local_1 = query_view_1.get("raw_local_embs", None)
        student_raw_local_2 = query_view_2.get("raw_local_embs", None)

        teacher_global_1 = None if teacher_view_1 is None else teacher_view_1.get("global_embs", None)
        teacher_global_2 = None if teacher_view_2 is None else teacher_view_2.get("global_embs", None)
        teacher_local_1 = None if teacher_view_1 is None else teacher_view_1.get("local_embs", None)
        teacher_local_2 = None if teacher_view_2 is None else teacher_view_2.get("local_embs", None)

        aux_embedding_1, aux_embedding_global_1, aux_embedding_local_1 = compute_multiscale_aux_embedding_transfer(
            raw_global_embeddings=student_raw_global_1,
            teacher_global_embeddings=teacher_global_1,
            raw_local_embeddings=student_raw_local_1,
            teacher_local_embeddings=teacher_local_1,
            foreground_mask=mask_view_1,
            lambda_global=1.0,
            lambda_local=aux_local_weight,
            detach_teacher=detach_teacher,
            local_scale_weights=aux_embedding_scale_weights,
        )

        aux_embedding_2, aux_embedding_global_2, aux_embedding_local_2 = compute_multiscale_aux_embedding_transfer(
            raw_global_embeddings=student_raw_global_2,
            teacher_global_embeddings=teacher_global_2,
            raw_local_embeddings=student_raw_local_2,
            teacher_local_embeddings=teacher_local_2,
            foreground_mask=mask_view_2,
            lambda_global=1.0,
            lambda_local=aux_local_weight,
            detach_teacher=detach_teacher,
            local_scale_weights=aux_embedding_scale_weights,
        )

        aux_embedding_loss = 0.5 * (aux_embedding_1 + aux_embedding_2)
        aux_embedding_global_loss = 0.5 * (aux_embedding_global_1 + aux_embedding_global_2)
        aux_embedding_local_loss = 0.5 * (aux_embedding_local_1 + aux_embedding_local_2)
    else:
        aux_embedding_loss = zero_for_aux
        aux_embedding_global_loss = zero_for_aux
        aux_embedding_local_loss = zero_for_aux

    if raw_spatial_enabled and use_mask:
        raw_mask_loss_1 = compute_multiscale_raw_mask_loss(
            query_view_1["raw_feats"],
            mask_view_1,
            scale_weights=raw_mask_scale_weights,
            foreground_background_margin=foreground_background_margin,
        )
        raw_mask_loss_2 = compute_multiscale_raw_mask_loss(
            query_view_2["raw_feats"],
            mask_view_2,
            scale_weights=raw_mask_scale_weights,
            foreground_background_margin=foreground_background_margin,
        )
        raw_mask_loss = 0.5 * (raw_mask_loss_1 + raw_mask_loss_2)

        raw_consistency_loss = compute_multiscale_raw_foreground_consistency_loss(
            query_view_1["raw_feats"],
            query_view_2["raw_feats"],
            mask_1=mask_view_1,
            mask_2=mask_view_2,
            scale_weights=raw_consistency_scale_weights,
        )
    else:
        raw_mask_loss = query_view_1["raw_feats"][0].sum() * 0.0
        raw_consistency_loss = raw_mask_loss

    if raw_prior_distillation_enabled and raw_prior_weight > 0:
        teacher_view_1 = _select_raw_prior_teacher_outputs(
            model_outputs,
            view_key="q1",
            teacher_source=raw_prior_teacher_source,
        )
        teacher_view_2 = _select_raw_prior_teacher_outputs(
            model_outputs,
            view_key="q2",
            teacher_source=raw_prior_teacher_source,
        )
        raw_prior_loss_1 = compute_multiscale_raw_prior_distillation_loss(
            raw_feature_maps=query_view_1["raw_feats"],
            teacher_outputs=teacher_view_1,
            mask=mask_view_1,
            teacher_branch=raw_prior_teacher_branch,
            scale_weights=raw_prior_scale_weights,
            loss_type=raw_prior_loss_type,
            student_temperature=raw_prior_student_temperature,
            teacher_temperature=raw_prior_teacher_temperature,
            detach_teacher=raw_prior_detach_teacher,
        )
        raw_prior_loss_2 = compute_multiscale_raw_prior_distillation_loss(
            raw_feature_maps=query_view_2["raw_feats"],
            teacher_outputs=teacher_view_2,
            mask=mask_view_2,
            teacher_branch=raw_prior_teacher_branch,
            scale_weights=raw_prior_scale_weights,
            loss_type=raw_prior_loss_type,
            student_temperature=raw_prior_student_temperature,
            teacher_temperature=raw_prior_teacher_temperature,
            detach_teacher=raw_prior_detach_teacher,
        )
        raw_prior_loss = 0.5 * (raw_prior_loss_1 + raw_prior_loss_2)
    else:
        raw_prior_loss = query_view_1["raw_feats"][0].sum() * 0.0

    total_loss = base_global_weight * loss_global + base_local_weight * loss_local
    if use_pos:
        total_loss = total_loss + base_position_weight * loss_position
    if aux_embedding_enabled and aux_embedding_weight > 0:
        total_loss = total_loss + aux_embedding_weight * aux_embedding_loss
    if raw_spatial_enabled and use_mask and raw_mask_weight > 0:
        total_loss = total_loss + raw_mask_weight * raw_mask_loss
    if raw_spatial_enabled and use_mask and raw_consistency_weight > 0:
        total_loss = total_loss + raw_consistency_weight * raw_consistency_loss
    if raw_prior_distillation_enabled and raw_prior_weight > 0:
        total_loss = total_loss + raw_prior_weight * raw_prior_loss

    return {
        "loss_total": total_loss,
        "loss_global": loss_global,
        "loss_local": loss_local,
        "loss_pos": loss_position,
        "loss_embed": aux_embedding_loss,
        "loss_embed_global": aux_embedding_global_loss,
        "loss_embed_local": aux_embedding_local_loss,
        "loss_raw_mask": raw_mask_loss,
        "loss_raw_fg_cons": raw_consistency_loss,
        "loss_raw_prior": raw_prior_loss,
    }


# Backward-compatible alias.
compute_total_loss = compute_stage1_total_loss