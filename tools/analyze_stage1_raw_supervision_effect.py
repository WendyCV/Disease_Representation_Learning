import os
import sys
import json
import csv
import copy
import argparse
from collections import defaultdict

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# =========================================================
# Path injection
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.dataset import UnlabeledLeafContrastiveDataset
from train_stage1 import build_stage1_model as build_train_model


# =========================================================
# IO / config
# =========================================================
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def to_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


# =========================================================
# Current project config readers
# =========================================================
def get_aux_cfg(cfg):
    return cfg["loss"].get("aux_embedding", {})


def get_raw_cfg(cfg):
    return cfg["loss"].get("raw_spatial", {})


def get_use_pos(cfg):
    return cfg.get("ablation", {}).get("use_pos", True)


def get_use_mask(cfg):
    return cfg.get("ablation", {}).get("use_mask", True)


def get_teacher_source(cfg):
    """
    snapshot_bias 已删除后，分析脚本中的 teacher_source 仅由 aux_embedding 决定。
    如果 aux_embedding 未启用，则不需要 teacher，默认返回 online。
    """
    aux_cfg = get_aux_cfg(cfg)
    if not aux_cfg.get("enabled", False):
        return "online"
    return aux_cfg.get("teacher_source", "online")


# =========================================================
# Dataset
# =========================================================
def build_dataloader(cfg, batch_size=None, max_workers=None):
    dataset = UnlabeledLeafContrastiveDataset(
        root_dir=cfg["data"]["train_dir"],
        image_size=cfg["data"]["image_size"],
        mask_root_dir=cfg["data"].get("mask_root_dir", None),
        mask_mode=cfg["data"].get("mask_mode", "sam2"),
        edge_mask=cfg["data"].get("edge_mask", True),
        min_valid_pixels=cfg["data"].get("min_valid_pixels", 4096),
        use_cache=cfg["data"].get("use_cache", True),
        mask_suffix=cfg["data"].get("mask_suffix", ".png"),
        missing_mask_policy=cfg["data"].get("missing_mask_policy", "ones"),
        external_mask_threshold=cfg["data"].get("external_mask_threshold", 127),
        external_mask_median_blur=cfg["data"].get("external_mask_median_blur", 0),
        external_mask_open_kernel=cfg["data"].get("external_mask_open_kernel", 0),
        external_mask_close_kernel=cfg["data"].get("external_mask_close_kernel", 0),
    )

    bs = batch_size if batch_size is not None else cfg["train"]["batch_size"]
    nw = max_workers if max_workers is not None else cfg["data"]["num_workers"]

    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if nw > 0 else False,
        prefetch_factor=2 if nw > 0 else None,
    )
    return dataset, loader


# =========================================================
# Model / checkpoint
# =========================================================
def build_model(cfg, device):
    """
    直接复用训练脚本里的 build_stage1_model，
    保证与训练时的 yaml -> model 构造逻辑完全一致。
    """
    model = build_train_model(cfg, device)
    return model.to(device)


def recover_runtime_flags_from_state_dict(model, state_dict):
    """
    snapshot_ready 不是 parameter，不会自动在 state_dict 中恢复。
    如果 checkpoint 里有 snapshot_teacher 权重，则推断 snapshot 已经捕获完成。
    """
    if hasattr(model, "snapshot_teacher") and model.snapshot_teacher is not None:
        has_snapshot_weights = any(k.startswith("snapshot_teacher.") for k in state_dict.keys())
        if has_snapshot_weights and hasattr(model, "snapshot_ready"):
            model.snapshot_ready = True


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    model.load_state_dict(state_dict, strict=True)
    recover_runtime_flags_from_state_dict(model, state_dict)
    return model, ckpt


# =========================================================
# Tensor helpers
# =========================================================
def _zero_like_from_available_tensors(*tensor_groups):
    for group in tensor_groups:
        if group is None:
            continue
        if isinstance(group, (list, tuple)):
            if len(group) > 0 and group[0] is not None:
                return group[0].sum() * 0.0
        elif torch.is_tensor(group):
            return group.sum() * 0.0
    return torch.tensor(0.0)


def _resize_mask(mask, target_hw, device=None, dtype=torch.float32):
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


def _resolve_scale_weights(scale_weights, num_scales, device, dtype):
    if scale_weights is None:
        scale_weights = [1.0] * num_scales
    scale_weights = list(scale_weights)
    if len(scale_weights) < num_scales:
        scale_weights = scale_weights + [scale_weights[-1]] * (num_scales - len(scale_weights))
    elif len(scale_weights) > num_scales:
        scale_weights = scale_weights[:num_scales]
    scale_weights = torch.tensor(scale_weights, device=device, dtype=dtype)
    return scale_weights / scale_weights.sum().clamp(min=1e-6)


def _weighted_spatial_average(value_map, weight_map=None, eps=1e-6):
    if value_map.dim() == 3:
        value_map = value_map.unsqueeze(1)
    if weight_map is None:
        return value_map.mean()
    if weight_map.dim() == 3:
        weight_map = weight_map.unsqueeze(1)
    numerator = (value_map * weight_map).sum(dim=(1, 2, 3))
    denominator = weight_map.sum(dim=(1, 2, 3)).clamp(min=eps)
    return (numerator / denominator).mean()


def _normalize_heatmap(heatmap):
    flattened = heatmap.flatten(2)
    min_value = flattened.min(dim=-1, keepdim=True).values.unsqueeze(-1)
    max_value = flattened.max(dim=-1, keepdim=True).values.unsqueeze(-1)
    return (heatmap - min_value) / (max_value - min_value + 1e-6)


def _channelwise_cosine_map(student_feature, teacher_feature, eps=1e-6):
    student_feature = F.normalize(student_feature, dim=1, eps=eps)
    teacher_feature = F.normalize(teacher_feature, dim=1, eps=eps)
    return (student_feature * teacher_feature).sum(dim=1, keepdim=True)


def _masked_avg_pool(feature_map, mask, eps=1e-6):
    if mask is None:
        return feature_map.mean(dim=(2, 3))
    mask = _resize_mask(mask, feature_map.shape[-2:], device=feature_map.device, dtype=feature_map.dtype)
    numerator = (feature_map * mask).sum(dim=(2, 3))
    denominator = mask.sum(dim=(2, 3)).clamp(min=eps)
    return numerator / denominator


# =========================================================
# Analysis losses
# =========================================================
def cosine_embed_loss(student_embedding, teacher_embedding, detach_teacher=True):
    if detach_teacher:
        teacher_embedding = teacher_embedding.detach()
    student_embedding = F.normalize(student_embedding, dim=1)
    teacher_embedding = F.normalize(teacher_embedding, dim=1)
    return 1.0 - (student_embedding * teacher_embedding).sum(dim=1).mean()


def local_embed_distill_loss_single_scale(raw_local_embedding, teacher_local_embedding, mask=None, detach_teacher=True):
    if detach_teacher:
        teacher_local_embedding = teacher_local_embedding.detach()
    region_mask = _resize_mask(mask, raw_local_embedding.shape[-2:], device=raw_local_embedding.device, dtype=raw_local_embedding.dtype)
    cosine_map = _channelwise_cosine_map(raw_local_embedding, teacher_local_embedding)
    return 1.0 - _weighted_spatial_average(cosine_map, region_mask)


def multiscale_embed_distill_loss(
    raw_global_embs,
    teacher_global_embs,
    raw_local_embs,
    teacher_local_embs,
    mask=None,
    lambda_global=1.0,
    lambda_local=1.0,
    detach_teacher=True,
    local_scale_weights=None,
):
    zero = _zero_like_from_available_tensors(
        raw_global_embs,
        teacher_global_embs,
        raw_local_embs,
        teacher_local_embs,
    )

    if (
        raw_global_embs is None
        or teacher_global_embs is None
        or raw_local_embs is None
        or teacher_local_embs is None
    ):
        return zero, zero, zero

    if len(raw_global_embs) == 0 or len(teacher_global_embs) == 0:
        return zero, zero, zero
    if len(raw_local_embs) == 0 or len(teacher_local_embs) == 0:
        return zero, zero, zero

    global_losses = [
        cosine_embed_loss(raw_global, teacher_global, detach_teacher=detach_teacher)
        for raw_global, teacher_global in zip(raw_global_embs, teacher_global_embs)
    ]
    aux_global_loss = torch.stack(global_losses).mean()

    scale_weights = _resolve_scale_weights(
        local_scale_weights,
        len(raw_local_embs),
        raw_local_embs[0].device,
        raw_local_embs[0].dtype,
    )

    local_losses = []
    for scale_index, (raw_local, teacher_local) in enumerate(zip(raw_local_embs, teacher_local_embs)):
        local_loss = local_embed_distill_loss_single_scale(
            raw_local,
            teacher_local,
            mask=mask,
            detach_teacher=detach_teacher,
        )
        local_losses.append(local_loss * scale_weights[scale_index])

    aux_local_loss = torch.stack(local_losses).sum()
    aux_total_loss = lambda_global * aux_global_loss + lambda_local * aux_local_loss
    return aux_total_loss, aux_global_loss, aux_local_loss


def raw_mask_supervision_single_scale(raw_feature_map, mask, fg_bg_margin=0.15):
    if mask is None:
        return raw_feature_map.sum() * 0.0

    raw_heatmap = _normalize_heatmap(raw_feature_map.abs().mean(dim=1, keepdim=True))
    foreground_mask = _resize_mask(mask, raw_heatmap.shape[-2:], device=raw_heatmap.device, dtype=raw_heatmap.dtype)
    background_mask = 1.0 - foreground_mask

    intersection = (raw_heatmap * foreground_mask).sum(dim=(1, 2, 3))
    union = raw_heatmap.sum(dim=(1, 2, 3)) + foreground_mask.sum(dim=(1, 2, 3)) + 1e-6
    dice_loss = 1.0 - (2.0 * intersection + 1e-6) / union

    foreground_mean = (raw_heatmap * foreground_mask).sum(dim=(1, 2, 3)) / foreground_mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
    background_mean = (raw_heatmap * background_mask).sum(dim=(1, 2, 3)) / background_mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
    margin_loss = F.relu(fg_bg_margin - (foreground_mean - background_mean))
    return (dice_loss + margin_loss).mean()


def multi_scale_raw_mask_supervision_loss(raw_feature_maps, mask, scale_weights=None, fg_bg_margin=0.15):
    scale_weights = _resolve_scale_weights(scale_weights, len(raw_feature_maps), raw_feature_maps[0].device, raw_feature_maps[0].dtype)
    losses = [
        raw_mask_supervision_single_scale(raw_feature_map, mask, fg_bg_margin=fg_bg_margin) * scale_weights[index]
        for index, raw_feature_map in enumerate(raw_feature_maps)
    ]
    return torch.stack(losses).sum()


def raw_foreground_consistency_single_scale(raw_feature_map_1, raw_feature_map_2, mask_1=None, mask_2=None):
    pooled_1 = _masked_avg_pool(raw_feature_map_1, mask_1)
    pooled_2 = _masked_avg_pool(raw_feature_map_2, mask_2)
    pooled_1 = F.normalize(pooled_1, dim=1)
    pooled_2 = F.normalize(pooled_2, dim=1)
    return 1.0 - (pooled_1 * pooled_2).sum(dim=1).mean()


def multi_scale_raw_foreground_consistency_loss(raw_feature_maps_1, raw_feature_maps_2, mask_1=None, mask_2=None, scale_weights=None):
    scale_weights = _resolve_scale_weights(scale_weights, len(raw_feature_maps_1), raw_feature_maps_1[0].device, raw_feature_maps_1[0].dtype)
    losses = [
        raw_foreground_consistency_single_scale(raw_feature_1, raw_feature_2, mask_1=mask_1, mask_2=mask_2) * scale_weights[index]
        for index, (raw_feature_1, raw_feature_2) in enumerate(zip(raw_feature_maps_1, raw_feature_maps_2))
    ]
    return torch.stack(losses).sum()


# =========================================================
# Teacher selection / forward
# =========================================================
def select_teacher_outputs(outputs, view_key="q1", teacher_source="online"):
    if teacher_source == "online":
        return outputs["q1"] if view_key == "q1" else outputs["q2"]
    if teacher_source == "momentum":
        return outputs["k1"] if view_key == "q1" else outputs["k2"]
    if teacher_source == "snapshot":
        return outputs["t1"] if view_key == "q1" else outputs["t2"]
    raise ValueError(f"Unsupported teacher_source: {teacher_source}")


def forward_for_analysis(model, x1, x2, fg1, fg2, teacher_source="online"):
    """
    当前工程版本专用：
    - q1/q2 来自 online
    - k1/k2 仅在 teacher_source == momentum 时使用
    - t1/t2 仅在 teacher_source == snapshot 且 snapshot_ready=True 时使用
    """
    q1 = model.online(x1, fg_mask=fg1)
    q2 = model.online(x2, fg_mask=fg2)

    k1, k2 = None, None
    t1, t2 = None, None

    with torch.no_grad():
        if teacher_source == "momentum":
            k1 = model.momentum(x1, fg_mask=fg1)
            k2 = model.momentum(x2, fg_mask=fg2)

        if teacher_source == "snapshot":
            if not hasattr(model, "snapshot_teacher") or model.snapshot_teacher is None:
                raise RuntimeError("Current model has no snapshot_teacher, but teacher_source='snapshot' was requested.")
            if not getattr(model, "snapshot_ready", False):
                raise RuntimeError("snapshot_teacher exists, but snapshot_ready=False. The loaded checkpoint may not contain a captured snapshot teacher.")
            t1 = model.snapshot_teacher(x1, fg_mask=fg1)
            t2 = model.snapshot_teacher(x2, fg_mask=fg2)

    return {
        "q1": q1,
        "q2": q2,
        "k1": k1,
        "k2": k2,
        "t1": t1,
        "t2": t2,
    }


# =========================================================
# Analysis loss assembly
# =========================================================
def compute_analysis_losses(outputs, m1, m2, cfg):
    aux_cfg = get_aux_cfg(cfg)
    raw_cfg = get_raw_cfg(cfg)

    q1, q2 = outputs["q1"], outputs["q2"]

    teacher_source_for_aux = aux_cfg.get("teacher_source", "online")
    teacher1 = select_teacher_outputs(outputs, "q1", teacher_source_for_aux)
    teacher2 = select_teacher_outputs(outputs, "q2", teacher_source_for_aux)

    if aux_cfg.get("enabled", False):
        loss_embed_1, loss_embed_global_1, loss_embed_local_1 = multiscale_embed_distill_loss(
            raw_global_embs=q1.get("raw_global_embs", None),
            teacher_global_embs=teacher1.get("global_embs", None),
            raw_local_embs=q1.get("raw_local_embs", None),
            teacher_local_embs=teacher1.get("local_embs", None),
            mask=m1,
            lambda_global=1.0,
            lambda_local=aux_cfg.get("local_weight", 0.2),
            detach_teacher=aux_cfg.get("detach_teacher", True),
            local_scale_weights=aux_cfg.get("scale_weights", [0.1, 0.45, 0.45]),
        )
        loss_embed_2, loss_embed_global_2, loss_embed_local_2 = multiscale_embed_distill_loss(
            raw_global_embs=q2.get("raw_global_embs", None),
            teacher_global_embs=teacher2.get("global_embs", None),
            raw_local_embs=q2.get("raw_local_embs", None),
            teacher_local_embs=teacher2.get("local_embs", None),
            mask=m2,
            lambda_global=1.0,
            lambda_local=aux_cfg.get("local_weight", 0.2),
            detach_teacher=aux_cfg.get("detach_teacher", True),
            local_scale_weights=aux_cfg.get("scale_weights", [0.1, 0.45, 0.45]),
        )
        loss_embed = 0.5 * (loss_embed_1 + loss_embed_2)
        loss_embed_global = 0.5 * (loss_embed_global_1 + loss_embed_global_2)
        loss_embed_local = 0.5 * (loss_embed_local_1 + loss_embed_local_2)
    else:
        zero = q1["raw_feats"][0].sum() * 0.0
        loss_embed = zero
        loss_embed_global = zero
        loss_embed_local = zero

    if raw_cfg.get("enabled", True):
        loss_raw_mask_1 = multi_scale_raw_mask_supervision_loss(
            q1["raw_feats"],
            m1,
            raw_cfg.get("mask_scale_weights", [0.1, 0.45, 0.45]),
            raw_cfg.get("foreground_background_margin", 0.15),
        )
        loss_raw_mask_2 = multi_scale_raw_mask_supervision_loss(
            q2["raw_feats"],
            m2,
            raw_cfg.get("mask_scale_weights", [0.1, 0.45, 0.45]),
            raw_cfg.get("foreground_background_margin", 0.15),
        )
        loss_raw_mask = 0.5 * (loss_raw_mask_1 + loss_raw_mask_2)

        loss_raw_fg_cons = multi_scale_raw_foreground_consistency_loss(
            q1["raw_feats"],
            q2["raw_feats"],
            mask_1=m1,
            mask_2=m2,
            scale_weights=raw_cfg.get("consistency_scale_weights", [0.1, 0.45, 0.45]),
        )
    else:
        zero = q1["raw_feats"][0].sum() * 0.0
        loss_raw_mask = zero
        loss_raw_fg_cons = zero

    lambda_embed = aux_cfg.get("weight", 0.10)
    lambda_raw_mask = raw_cfg.get("mask_weight", 0.25)
    lambda_raw_fg_cons = raw_cfg.get("consistency_weight", 0.15)

    raw_only_loss = lambda_raw_mask * loss_raw_mask + lambda_raw_fg_cons * loss_raw_fg_cons
    teacher_guided_loss = raw_only_loss + lambda_embed * loss_embed

    debug = {
        "q1_raw_requires_grad": [bool(f.requires_grad) for f in q1["raw_feats"]],
        "q2_raw_requires_grad": [bool(f.requires_grad) for f in q2["raw_feats"]],
        "snapshot_ready": bool(getattr(outputs.get("model_ref", None), "snapshot_ready", False)) if outputs.get("model_ref", None) is not None else None,
        "teacher_available": {
            "t1": outputs.get("t1") is not None,
            "t2": outputs.get("t2") is not None,
            "k1": outputs.get("k1") is not None,
            "k2": outputs.get("k2") is not None,
        },
        "raw_mask": {"value": to_float(loss_raw_mask), "requires_grad": bool(loss_raw_mask.requires_grad)},
        "raw_fg_cons": {"value": to_float(loss_raw_fg_cons), "requires_grad": bool(loss_raw_fg_cons.requires_grad)},
        "embed": {"value": to_float(loss_embed), "requires_grad": bool(loss_embed.requires_grad)},
    }

    return {
        "loss_embed": loss_embed,
        "loss_embed_global": loss_embed_global,
        "loss_embed_local": loss_embed_local,
        "loss_raw_mask": loss_raw_mask,
        "loss_raw_fg_cons": loss_raw_fg_cons,
        "raw_only_loss": raw_only_loss,
        "teacher_guided_loss": teacher_guided_loss,
        "debug": debug,
    }


# =========================================================
# Feature statistics
# =========================================================
@torch.no_grad()
def compute_feat_stats(feat_list, mask):
    stats = []
    for i, feat in enumerate(feat_list, start=1):
        hm = _normalize_heatmap(feat.abs().mean(dim=1, keepdim=True))
        fg = _resize_mask(mask, hm.shape[-2:], device=hm.device, dtype=hm.dtype)
        bg = 1.0 - fg
        fg_mean = (hm * fg).sum(dim=(1, 2, 3)) / fg.sum(dim=(1, 2, 3)).clamp(min=1.0)
        bg_mean = (hm * bg).sum(dim=(1, 2, 3)) / bg.sum(dim=(1, 2, 3)).clamp(min=1.0)
        stats.append({
            "scale": i,
            "fg_mean": float(fg_mean.mean().cpu()),
            "bg_mean": float(bg_mean.mean().cpu()),
            "fg_minus_bg": float((fg_mean - bg_mean).mean().cpu()),
            "fg_div_bg": float((fg_mean / bg_mean.clamp(min=1e-6)).mean().cpu()),
        })
    return stats


@torch.no_grad()
def compute_raw_pos_fg_cos(raw_feats, pos_feats, mask):
    stats = []
    for i, (raw, pos) in enumerate(zip(raw_feats, pos_feats), start=1):
        fg = _resize_mask(mask, raw.shape[-2:], device=raw.device, dtype=raw.dtype)
        raw_fg = raw * fg
        pos_fg = pos * fg
        cos_map = _channelwise_cosine_map(raw_fg, pos_fg)
        cos = _weighted_spatial_average(cos_map, fg)
        stats.append({"scale": i, "fg_cosine": float(cos.cpu())})
    return stats


def average_scale_dicts(list_of_scale_dicts):
    if len(list_of_scale_dicts) == 0:
        return []
    num_scales = len(list_of_scale_dicts[0])
    out = []
    for scale_index in range(num_scales):
        keys = list(list_of_scale_dicts[0][scale_index].keys())
        merged = {}
        for key in keys:
            if key == "scale":
                merged[key] = list_of_scale_dicts[0][scale_index][key]
            else:
                merged[key] = sum(item[scale_index][key] for item in list_of_scale_dicts) / len(list_of_scale_dicts)
        out.append(merged)
    return out


# =========================================================
# Gradient analysis
# =========================================================
def categorize_param_name(name):
    lname = name.lower()
    if not lname.startswith("online"):
        return "non_online"
    if "pos" in lname:
        return "online_position"
    if "proj" in lname or "global" in lname or "local" in lname or "head" in lname:
        return "online_heads_projectors"
    return "online_backbone_like"


def collect_grad_stats(model, topk=20):
    group_norm_sq = defaultdict(float)
    rows = []

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        grad_norm = p.grad.detach().norm().item()
        group = categorize_param_name(name)
        group_norm_sq[group] += grad_norm ** 2
        rows.append((name, grad_norm, group, tuple(p.shape)))

    group_norm = {k: (v ** 0.5) for k, v in group_norm_sq.items()}
    rows = sorted(rows, key=lambda x: x[1], reverse=True)[:topk]

    return {
        "group_grad_norm": group_norm,
        "top_param_grads": [
            {"name": n, "grad_norm": g, "group": grp, "shape": list(shape)}
            for n, g, grp, shape in rows
        ],
    }


def trainable_param_summary(model):
    total = 0
    trainable = 0
    online_total = 0
    online_trainable = 0
    backbone_total = 0
    backbone_trainable = 0

    for name, p in model.named_parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num

        if name.startswith("online"):
            online_total += num
            if p.requires_grad:
                online_trainable += num

        if name.startswith("online.backbone"):
            backbone_total += num
            if p.requires_grad:
                backbone_trainable += num

    return {
        "total_params": total,
        "trainable_params": trainable,
        "online_total": online_total,
        "online_trainable": online_trainable,
        "online_backbone_total": backbone_total,
        "online_backbone_trainable": backbone_trainable,
    }


# =========================================================
# Single-step probe
# =========================================================
def run_single_step_probe(model, batch, cfg, device, step_lr=1e-3):
    probe_model = copy.deepcopy(model).to(device)
    probe_model.train()

    x1 = batch["x1"].to(device, non_blocking=True)
    x2 = batch["x2"].to(device, non_blocking=True)
    m1 = batch["m1"].to(device, non_blocking=True)
    m2 = batch["m2"].to(device, non_blocking=True)

    use_mask = get_use_mask(cfg)
    fg1 = m1 if use_mask else None
    fg2 = m2 if use_mask else None

    teacher_source = get_teacher_source(cfg)

    with torch.no_grad():
        out_before = forward_for_analysis(probe_model, x1, x2, fg1, fg2, teacher_source=teacher_source)
        out_before["model_ref"] = probe_model
        before_raw = average_scale_dicts([
            compute_feat_stats(out_before["q1"]["raw_feats"], m1),
            compute_feat_stats(out_before["q2"]["raw_feats"], m2),
        ])

    optimizer = torch.optim.SGD(probe_model.parameters(), lr=step_lr, momentum=0.0, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)

    out_step = forward_for_analysis(probe_model, x1, x2, fg1, fg2, teacher_source=teacher_source)
    out_step["model_ref"] = probe_model
    loss_dict = compute_analysis_losses(out_step, m1, m2, cfg)

    if loss_dict["teacher_guided_loss"].requires_grad:
        probe_target = loss_dict["teacher_guided_loss"]
        probe_name = "teacher_guided_loss"
    elif loss_dict["raw_only_loss"].requires_grad:
        probe_target = loss_dict["raw_only_loss"]
        probe_name = "raw_only_loss"
    else:
        return {
            "probe_loss_name": "none",
            "probe_loss": 0.0,
            "single_step_raw_deltas": [],
            "probe_debug": loss_dict["debug"],
        }

    probe_target.backward()
    optimizer.step()

    with torch.no_grad():
        out_after = forward_for_analysis(probe_model, x1, x2, fg1, fg2, teacher_source=teacher_source)
        out_after["model_ref"] = probe_model
        after_raw = average_scale_dicts([
            compute_feat_stats(out_after["q1"]["raw_feats"], m1),
            compute_feat_stats(out_after["q2"]["raw_feats"], m2),
        ])

    deltas = []
    for b, a in zip(before_raw, after_raw):
        deltas.append({
            "scale": b["scale"],
            "before_fg_minus_bg": b["fg_minus_bg"],
            "after_fg_minus_bg": a["fg_minus_bg"],
            "delta_fg_minus_bg": a["fg_minus_bg"] - b["fg_minus_bg"],
            "before_fg_div_bg": b["fg_div_bg"],
            "after_fg_div_bg": a["fg_div_bg"],
            "delta_fg_div_bg": a["fg_div_bg"] - b["fg_div_bg"],
        })

    return {
        "probe_loss_name": probe_name,
        "probe_loss": to_float(probe_target),
        "single_step_raw_deltas": deltas,
        "probe_debug": loss_dict["debug"],
    }


# =========================================================
# Default checkpoint path
# =========================================================
def default_ckpt_path(cfg, ckpt_name="best.pth"):
    exp_name = cfg.get("experiment", {}).get("name", None)
    if exp_name:
        return os.path.join(cfg["train"]["save_dir"], exp_name, ckpt_name)

    use_pos = get_use_pos(cfg)
    use_mask = get_use_mask(cfg)

    if use_pos and use_mask:
        exp_name = "use_pos_mask"
    elif (not use_pos) and use_mask:
        exp_name = "wo_pos"
    elif use_pos and (not use_mask):
        exp_name = "wo_mask"
    else:
        exp_name = "wo_pos_wo_mask"

    return os.path.join(cfg["train"]["save_dir"], exp_name, ckpt_name)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/ssl_config.yaml")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--ckpt_name", type=str, default="best.pth")
    parser.add_argument("--num_batches", type=int, default=2)
    parser.add_argument("--analysis_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--step_lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="./runs/glcp_stage1_yolo_det/analysis_raw_supervision")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    use_pos = get_use_pos(cfg)
    use_mask = get_use_mask(cfg)
    teacher_source = get_teacher_source(cfg)

    ckpt_path = args.ckpt if args.ckpt else default_ckpt_path(cfg, ckpt_name=args.ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = args.output_dir
    ensure_dir(out_dir)

    _, loader = build_dataloader(cfg, batch_size=args.analysis_batch_size, max_workers=args.num_workers)

    model = build_model(cfg, device)
    model, _ = load_checkpoint(model, ckpt_path, device)
    model.train()

    if hasattr(model, "assert_backbone_trainable"):
        print("[Check]", model.assert_backbone_trainable(raise_if_false=False))

    print("=" * 80)
    print("Analyze raw supervision effect")
    print(f"Config    : {args.config}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device    : {device}")
    print(f"Use pos   : {use_pos}")
    print(f"Use mask  : {use_mask}")
    print(f"Teacher   : {teacher_source}")
    print("Trainable param summary:")
    print(json.dumps(trainable_param_summary(model), indent=2, ensure_ascii=False))
    print("=" * 80)

    all_batch_results = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.num_batches:
            break

        x1 = batch["x1"].to(device, non_blocking=True)
        x2 = batch["x2"].to(device, non_blocking=True)
        m1 = batch["m1"].to(device, non_blocking=True)
        m2 = batch["m2"].to(device, non_blocking=True)

        fg1 = m1 if use_mask else None
        fg2 = m2 if use_mask else None

        with torch.no_grad():
            outputs = forward_for_analysis(model, x1, x2, fg1, fg2, teacher_source=teacher_source)
            outputs["model_ref"] = model

            raw_stats = average_scale_dicts([
                compute_feat_stats(outputs["q1"]["raw_feats"], m1),
                compute_feat_stats(outputs["q2"]["raw_feats"], m2),
            ])

            pos_stats = average_scale_dicts([
                compute_feat_stats(outputs["q1"]["pos_feats"], m1),
                compute_feat_stats(outputs["q2"]["pos_feats"], m2),
            ]) if use_pos else []

            raw_pos_cos = average_scale_dicts([
                compute_raw_pos_fg_cos(outputs["q1"]["raw_feats"], outputs["q1"]["pos_feats"], m1),
                compute_raw_pos_fg_cos(outputs["q2"]["raw_feats"], outputs["q2"]["pos_feats"], m2),
            ]) if use_pos else []

        model.zero_grad(set_to_none=True)
        outputs_grad = forward_for_analysis(model, x1, x2, fg1, fg2, teacher_source=teacher_source)
        outputs_grad["model_ref"] = model
        loss_dict_grad = compute_analysis_losses(outputs_grad, m1, m2, cfg)

        raw_only_loss = loss_dict_grad["raw_only_loss"]
        raw_only_grad = bool(raw_only_loss.requires_grad)
        if raw_only_grad:
            raw_only_loss.backward()
            grad_raw_only = collect_grad_stats(model, topk=20)
        else:
            grad_raw_only = {"group_grad_norm": {}, "top_param_grads": []}

        model.zero_grad(set_to_none=True)

        outputs_grad2 = forward_for_analysis(model, x1, x2, fg1, fg2, teacher_source=teacher_source)
        outputs_grad2["model_ref"] = model
        loss_dict_grad2 = compute_analysis_losses(outputs_grad2, m1, m2, cfg)

        teacher_guided_loss = loss_dict_grad2["teacher_guided_loss"]
        teacher_guided_grad = bool(teacher_guided_loss.requires_grad)
        if teacher_guided_grad:
            teacher_guided_loss.backward()
            grad_teacher_guided = collect_grad_stats(model, topk=20)
        else:
            grad_teacher_guided = {"group_grad_norm": {}, "top_param_grads": []}

        model.zero_grad(set_to_none=True)

        probe_result = run_single_step_probe(model, batch, cfg, device, step_lr=args.step_lr)

        batch_result = {
            "batch_idx": batch_idx,
            "loss_values": {
                "loss_embed": to_float(loss_dict_grad2["loss_embed"]),
                "loss_embed_global": to_float(loss_dict_grad2["loss_embed_global"]),
                "loss_embed_local": to_float(loss_dict_grad2["loss_embed_local"]),
                "loss_raw_mask": to_float(loss_dict_grad2["loss_raw_mask"]),
                "loss_raw_fg_cons": to_float(loss_dict_grad2["loss_raw_fg_cons"]),
                "raw_only_loss": to_float(raw_only_loss),
                "raw_only_requires_grad": raw_only_grad,
                "teacher_guided_loss": to_float(teacher_guided_loss),
                "teacher_guided_requires_grad": teacher_guided_grad,
            },
            "debug": loss_dict_grad2["debug"],
            "raw_stats": raw_stats,
            "pos_stats": pos_stats,
            "raw_pos_fg_cos": raw_pos_cos,
            "grad_raw_only": grad_raw_only,
            "grad_teacher_guided": grad_teacher_guided,
            "single_step_probe": probe_result,
        }
        all_batch_results.append(batch_result)

        print(f"\n[Batch {batch_idx}]")
        print("Losses:")
        print(json.dumps(batch_result["loss_values"], indent=2, ensure_ascii=False))
        print("Debug:")
        print(json.dumps(batch_result["debug"], indent=2, ensure_ascii=False))

        print("Raw fg-bg stats:")
        for row in raw_stats:
            print(f"  L{row['scale']}: fg={row['fg_mean']:.4f}, bg={row['bg_mean']:.4f}, fg-bg={row['fg_minus_bg']:.4f}, fg/bg={row['fg_div_bg']:.4f}")

        if use_pos:
            print("Pos fg-bg stats:")
            for row in pos_stats:
                print(f"  L{row['scale']}: fg={row['fg_mean']:.4f}, bg={row['bg_mean']:.4f}, fg-bg={row['fg_minus_bg']:.4f}, fg/bg={row['fg_div_bg']:.4f}")

            print("Raw-pos fg cosine:")
            for row in raw_pos_cos:
                print(f"  L{row['scale']}: fg_cos={row['fg_cosine']:.4f}")

        print("Grad group norms (raw_only):")
        print(json.dumps(grad_raw_only["group_grad_norm"], indent=2, ensure_ascii=False))

        print("Grad group norms (teacher_guided):")
        print(json.dumps(grad_teacher_guided["group_grad_norm"], indent=2, ensure_ascii=False))

        print("Single-step raw delta:")
        if len(probe_result["single_step_raw_deltas"]) == 0:
            print("  (no step update; probe loss had no grad)")
        else:
            print(f"  probe_loss_name: {probe_result['probe_loss_name']}")
            for row in probe_result["single_step_raw_deltas"]:
                print(f"  L{row['scale']}: Δ(fg-bg)={row['delta_fg_minus_bg']:.6f}, Δ(fg/bg)={row['delta_fg_div_bg']:.6f}")

    summary = {
        "config_path": args.config,
        "ckpt_path": ckpt_path,
        "analysis_batch_size": args.analysis_batch_size,
        "num_batches": len(all_batch_results),
        "step_lr": args.step_lr,
        "teacher_source": teacher_source,
        "trainable_param_summary": trainable_param_summary(model),
        "batches": all_batch_results,
    }

    with open(os.path.join(out_dir, "analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(out_dir, "analysis_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "batch_idx", "scale",
            "raw_fg_mean", "raw_bg_mean", "raw_fg_minus_bg", "raw_fg_div_bg",
            "pos_fg_mean", "pos_bg_mean", "pos_fg_minus_bg", "pos_fg_div_bg",
            "raw_pos_fg_cos",
            "loss_raw_mask", "loss_raw_fg_cons", "loss_embed",
            "raw_only_requires_grad", "teacher_guided_requires_grad",
            "probe_loss_name", "probe_delta_fg_minus_bg", "probe_delta_fg_div_bg",
        ])

        for b in all_batch_results:
            losses = b["loss_values"]
            raw_stats = b["raw_stats"]
            pos_stats = b["pos_stats"] if len(b["pos_stats"]) > 0 else [{} for _ in raw_stats]
            cos_stats = b["raw_pos_fg_cos"] if len(b["raw_pos_fg_cos"]) > 0 else [{} for _ in raw_stats]

            if len(b["single_step_probe"]["single_step_raw_deltas"]) > 0:
                probe_stats = b["single_step_probe"]["single_step_raw_deltas"]
            else:
                probe_stats = [{"delta_fg_minus_bg": "", "delta_fg_div_bg": ""} for _ in raw_stats]

            for rs, ps, cs, pr in zip(raw_stats, pos_stats, cos_stats, probe_stats):
                writer.writerow([
                    b["batch_idx"],
                    rs["scale"],
                    rs.get("fg_mean", ""),
                    rs.get("bg_mean", ""),
                    rs.get("fg_minus_bg", ""),
                    rs.get("fg_div_bg", ""),
                    ps.get("fg_mean", ""),
                    ps.get("bg_mean", ""),
                    ps.get("fg_minus_bg", ""),
                    ps.get("fg_div_bg", ""),
                    cs.get("fg_cosine", ""),
                    losses["loss_raw_mask"],
                    losses["loss_raw_fg_cons"],
                    losses["loss_embed"],
                    losses["raw_only_requires_grad"],
                    losses["teacher_guided_requires_grad"],
                    b["single_step_probe"].get("probe_loss_name", ""),
                    pr.get("delta_fg_minus_bg", ""),
                    pr.get("delta_fg_div_bg", ""),
                ])

    print("\nSaved:")
    print(f"  JSON: {os.path.join(out_dir, 'analysis_summary.json')}")
    print(f"  CSV : {csv_path}")


if __name__ == "__main__":
    main()