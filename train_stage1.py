from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Tuple

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from datasets.stage1_ssl_dataset import Stage1ContrastiveDataset
from losses.stage1_ssl_losses import compute_stage1_total_loss, RPDDynamicScaleWeightsGating
from models.stage1_ssl_model import Stage1SslModel
from utils.checkpoint import save_checkpoint
from utils.config_utils import load_yaml_config, summarize_stage1_config, validate_stage1_config
from utils.meter import AverageMeter
from utils.seed import set_seed


def build_stage1_dataloader(config: Dict) -> Tuple[Stage1ContrastiveDataset, DataLoader]:
    dataset = Stage1ContrastiveDataset(
        root_dir=config["data"]["train_dir"],
        image_size=config["data"]["image_size"],
        mask_root_dir=config["data"].get("mask_root_dir"),
        mask_mode=config["data"].get("mask_mode", "sam2"),
        edge_mask=config["data"].get("edge_mask", False),
        min_valid_pixels=config["data"].get("min_valid_pixels", 4096),
        use_cache=config["data"].get("use_cache", True),
        mask_suffix=config["data"].get("mask_suffix", ".png"),
        missing_mask_policy=config["data"].get("missing_mask_policy", "ones"),
        external_mask_threshold=config["data"].get("external_mask_threshold", 127),
        external_mask_median_blur=config["data"].get("external_mask_median_blur", 0),
        external_mask_open_kernel=config["data"].get("external_mask_open_kernel", 0),
        external_mask_close_kernel=config["data"].get("external_mask_close_kernel", 0),

        # 新增：控制 DSC_* 增补图片数量
        extra_image_prefixes=config["data"].get("extra_image_prefixes", "DSC_"),
        extra_max_images=config["data"].get("extra_max_images", None),
        extra_ratio_to_base=config["data"].get("extra_ratio_to_base", None),
        extra_sampling=config["data"].get("extra_sampling", "even"),
        extra_random_seed=config["data"].get("extra_random_seed", 2026),
    )

    batch_size = int(config["data"].get("batch_size", config["train"]["batch_size"]))
    num_workers = int(config["data"].get("num_workers", 0))
    pin_memory = config["data"].get("pin_memory", False)
    persistent_workers = config["data"].get("persistent_workers", False)
    prefetch_factor = config["data"].get("prefetch_factor", None)
    prefetch_factor = prefetch_factor if num_workers > 0 else None
    prefetch_factor = int(prefetch_factor) if prefetch_factor else None
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    return dataset, dataloader


def build_stage1_model(config: Dict, device: torch.device, use_pos: bool | None = None) -> Stage1SslModel:
    aux_embedding_cfg = config["loss"].get("aux_embedding", {})
    snapshot_cfg = config["model"].get("snapshot_teacher", {})
    runtime_cfg = config.get("runtime", {})

    resolved_use_pos = runtime_cfg.get("use_pos", config.get("ablation", {}).get("use_pos", True)) if use_pos is None else use_pos
    use_snapshot_teacher = runtime_cfg.get("needs_snapshot_teacher", snapshot_cfg.get("enabled", False))

    model = Stage1SslModel(
        yolo_model=config["model"]["yolo_model"],
        nc=config["model"].get("nc"),
        layer_indices=tuple(config["model"]["layer_indices"]),
        image_size=config["data"]["image_size"],
        proj_dim=config["model"]["proj_dim"],
        local_dim=config["model"]["local_dim"],
        queue_size=config["model"]["queue_size"],
        momentum=config["model"]["momentum"],
        sppf_indice=config["model"].get("sppf_indice", 9),
        use_pos=resolved_use_pos,
        pos_pe_channels=config["model"].get("pos_pe_channels", 64),
        pos_pe_spans=config["model"].get("pos_pe_spans", [1, 1, 1]),
        scale_weight_before=config["model"].get("scale_weight_before", True),
        pos_init_scales=config["model"].get("pos_init_scales", [0.1, 0.5, 1.0]),
        pos_enable_fg_guidance=config["model"].get("pos_enable_fg_guidance", True),
        pos_fg_gate_init=config["model"].get("pos_fg_gate_init", 1.0),
        enable_raw_projection=aux_embedding_cfg.get("enabled", False),
        separate_projector=config["model"].get("separate_projector", False),
        use_snapshot_teacher=use_snapshot_teacher,
        verbose=True
    )
    return model.to(device)


def build_experiment_name(config: Dict) -> str:
    return config["experiment"]["name"]


def initialize_csv_logger(save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train_log.csv")
    if os.path.exists(log_path):
        os.remove(log_path)
    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "epoch", "lr", "snapshot_ready", "loss_total", "loss_global", "loss_local", "loss_pos",
            "loss_embed", "loss_embed_global", "loss_embed_local", "loss_raw_mask", "loss_raw_fg_cons", "loss_raw_prior",
        ])
    return log_path


def append_csv_log(log_path: str, epoch: int, learning_rate: float, snapshot_ready: bool, stats: Dict[str, float]) -> None:
    with open(log_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            epoch, learning_rate, int(snapshot_ready),
            round(stats["loss_total"], 8), round(stats["loss_global"], 8), round(stats["loss_local"], 8),
            round(stats["loss_pos"], 8), round(stats["loss_embed"], 8), round(stats["loss_embed_global"], 8),
            round(stats["loss_embed_local"], 8), round(stats["loss_raw_mask"], 8), round(stats["loss_raw_fg_cons"], 8), round(stats.get("loss_raw_prior", 0.0), 8),
        ])


def save_runtime_config(config: Dict, save_dir: str) -> None:
    config_path = os.path.join(save_dir, "config_used.yaml")
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


import math


def get_scheduled_rpd_weight(
    epoch: int,
    total_epochs: int,
    base_weight: float,
    schedule_cfg: dict | None = None,
) -> float:
    """
    Epoch-wise RPD weight schedule.

    Supported:
    - no schedule: fixed base_weight
    - linear_decay
    - cosine_decay

    For cosine_decay:
        before start_ratio: base_weight
        between start_ratio and end_ratio: cosine decay from base_weight to min_weight
        after end_ratio: min_weight
    """
    if schedule_cfg is None:
        return float(base_weight)

    if not bool(schedule_cfg.get("enabled", False)):
        return float(base_weight)

    schedule_type = str(schedule_cfg.get("type", "cosine_decay"))

    start_ratio = float(schedule_cfg.get("start_ratio", 0.30))
    end_ratio = float(schedule_cfg.get("end_ratio", 0.90))
    min_weight = float(schedule_cfg.get("min_weight", 0.03))

    if total_epochs <= 1:
        return float(base_weight)

    progress = float(epoch) / float(total_epochs - 1)

    if progress <= start_ratio:
        return float(base_weight)

    if progress >= end_ratio:
        return float(min_weight)

    decay_progress = (progress - start_ratio) / max(end_ratio - start_ratio, 1e-8)

    if schedule_type == "linear_decay":
        current_weight = base_weight * (1.0 - decay_progress) + min_weight * decay_progress

    elif schedule_type == "cosine_decay":
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        current_weight = min_weight + (base_weight - min_weight) * cosine_factor

    else:
        raise ValueError(f"Unsupported RPD weight schedule type: {schedule_type}")

    return float(current_weight)


def train_one_epoch(
        model: Stage1SslModel, dataloader: DataLoader, optimizer, device: torch.device, config: Dict, epoch_index: int, total_epochs: int,
        scaler: GradScaler | None = None, amp_enabled: bool = False, rpd_scale_weights_gating = None
) -> Dict[str, float]:
    model.train()

    meters = {name: AverageMeter() for name in [
        "loss_total", "loss_global", "loss_local", "loss_pos", "loss_embed", "loss_embed_global",
        "loss_embed_local", "loss_raw_mask", "loss_raw_fg_cons", "loss_raw_prior",
    ]}

    base_loss_cfg = config["loss"]["base"]
    aux_embedding_cfg = config["loss"].get("aux_embedding", {})
    raw_spatial_cfg = config["loss"].get("raw_spatial", {})
    raw_prior_cfg = config["loss"].get("raw_prior_distillation", {})
    raw_prior_weight_decay_cfg = raw_prior_cfg.get("weight_schedule", None)

    use_pos = config["runtime"]["use_pos"]
    use_mask = config["runtime"]["use_mask"]

    progress_bar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch_index}", ncols=168)

    for batch in progress_bar:
        view_1 = batch["x1"].to(device, non_blocking=True)
        view_2 = batch["x2"].to(device, non_blocking=True)
        mask_1 = batch["m1"].to(device, non_blocking=True)
        mask_2 = batch["m2"].to(device, non_blocking=True)

        fg_mask_1 = mask_1 if use_mask else None
        fg_mask_2 = mask_2 if use_mask else None

        current_raw_prior_weight = get_scheduled_rpd_weight(
            epoch=epoch_index,
            total_epochs=total_epochs,
            base_weight=float(raw_prior_cfg.get("weight", 0.0)),
            schedule_cfg=raw_prior_weight_decay_cfg
        )

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled):
            model_outputs = model(view_1, view_2, fg_mask_1, fg_mask_2)
            #####################################################################################
            if rpd_scale_weights_gating is not None:
                effective_w1, _ = rpd_scale_weights_gating([f.detach() for f in model_outputs["q1"]["raw_feats"]], return_dynamic=True)
                effective_w2, _ = rpd_scale_weights_gating([f.detach() for f in model_outputs["q2"]["raw_feats"]], return_dynamic=True)
                # 两个 view 共同决定一组 shared scale weights
                rpd_effective_scale_weights = 0.5 * (effective_w1 + effective_w2)
                rpd_effective_scale_weights = rpd_effective_scale_weights / rpd_effective_scale_weights.sum().clamp(min=1e-6)
            else:
                rpd_effective_scale_weights = torch.tensor(raw_prior_cfg.get("scale_weights", [0.30, 0.40, 0.30]), device=device, dtype=torch.float32)
            #####################################################################################
            loss_dict = compute_stage1_total_loss(
                model_outputs=model_outputs,
                mask_view_1=mask_1,
                mask_view_2=mask_2,
                temperature=config["model"]["temperature"],
                base_global_weight=base_loss_cfg["global"],
                base_local_weight=base_loss_cfg["local"],
                base_position_weight=base_loss_cfg["position"],
                use_pos=use_pos,
                use_mask=use_mask,
                aux_embedding_enabled=aux_embedding_cfg.get("enabled", False),
                aux_teacher_source=aux_embedding_cfg.get("teacher_source", "snapshot"),
                aux_embedding_weight=aux_embedding_cfg.get("weight", 0.10),
                aux_local_weight=aux_embedding_cfg.get("local_weight", 0.20),
                detach_teacher=aux_embedding_cfg.get("detach_teacher", True),
                aux_embedding_scale_weights=aux_embedding_cfg.get("scale_weights", [0.10, 0.45, 0.45]),
                raw_spatial_enabled=raw_spatial_cfg.get("enabled", True),
                raw_mask_weight=raw_spatial_cfg.get("mask_weight", 0.25),
                raw_consistency_weight=raw_spatial_cfg.get("consistency_weight", 0.15),
                raw_mask_scale_weights=raw_spatial_cfg.get("mask_scale_weights", [0.10, 0.45, 0.45]),
                raw_consistency_scale_weights=raw_spatial_cfg.get("consistency_scale_weights", [0.10, 0.45, 0.45]),
                foreground_background_margin=raw_spatial_cfg.get("foreground_background_margin", 0.15),
                raw_prior_distillation_enabled=(
                    raw_prior_cfg.get("enabled", False)
                    and epoch_index >= int(raw_prior_cfg.get("start_epoch", 1))
                ),
                raw_prior_teacher_source=raw_prior_cfg.get("teacher_source", "online"),
                raw_prior_teacher_branch=raw_prior_cfg.get("teacher_branch", "pos_feats"),
                raw_prior_weight=current_raw_prior_weight,
                raw_prior_loss_type=raw_prior_cfg.get("loss_type", "smooth_l1"),
                raw_prior_scale_weights=rpd_effective_scale_weights,
                raw_prior_detach_teacher=raw_prior_cfg.get("detach_teacher", True),
                raw_prior_student_temperature=float(raw_prior_cfg.get("student_temperature", 0.10)),
                raw_prior_teacher_temperature=float(raw_prior_cfg.get("teacher_temperature", 0.07)),
            )

        if amp_enabled and scaler is not None:
            scaler.scale(loss_dict["loss_total"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict["loss_total"].backward()
            optimizer.step()

        with torch.no_grad():
            keys_per_scale = []
            for key_embeddings_1, key_embeddings_2 in zip(model_outputs["k1"]["global_embs"], model_outputs["k2"]["global_embs"]):
                keys = torch.cat([key_embeddings_1, key_embeddings_2], dim=0).float()
                keys_per_scale.append(keys)
            model.queue.enqueue_dequeue(keys_per_scale)

        batch_size = view_1.size(0)
        for metric_name, meter in meters.items():
            meter.update(loss_dict[metric_name].item(), batch_size)

        progress_bar.set_postfix({
            "snap": int(getattr(model, "snapshot_ready", False)),
            "L_total": f"{meters['loss_total'].avg:.4f}",
            "L_g": f"{meters['loss_global'].avg:.4f}",
            "L_l": f"{meters['loss_local'].avg:.4f}",
            "L_p": f"{meters['loss_pos'].avg:.4f}",
            "L_e": f"{meters['loss_embed'].avg:.4f}",
            "L_m": f"{meters['loss_raw_mask'].avg:.4f}",
            "L_r": f"{meters['loss_raw_fg_cons'].avg:.4f}",
            "L_rpd": f"{meters['loss_raw_prior'].avg:.4f}",
        })

    last_rpd_effective_scale_weights = rpd_effective_scale_weights.detach().float().cpu()
    epoch_stats = {name: meter.avg for name, meter in meters.items()}
    epoch_stats["rpd_effective_scale_weights"] = last_rpd_effective_scale_weights.tolist()
    return epoch_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the stage-1 SSL model with orthogonal ablations.")
    parser.add_argument("--config", type=str, default="./configs/ssl_config.yaml")
    args = parser.parse_args()

    config = validate_stage1_config(load_yaml_config(args.config))
    set_seed(config["train"]["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")
    experiment_name = build_experiment_name(config)
    save_dir = os.path.join(config["train"]["save_dir"], experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    log_path = initialize_csv_logger(save_dir)
    save_runtime_config(config, save_dir)

    print("=" * 80)
    print("Loaded config:", args.config)
    print("Device       :", device)
    print("Save dir     :", save_dir)
    print("Log path     :", log_path)
    print("Config summary:")
    for key, value in summarize_stage1_config(config).items():
        print(f"  {key}: {value}")
    print("=" * 80)

    _, dataloader = build_stage1_dataloader(config)
    model = build_stage1_model(config, device=device)
    backbone_status = model.assert_backbone_trainable(raise_if_false=True)
    print("[Check] backbone_total/trainable:", backbone_status["backbone_total"], "/", backbone_status["backbone_trainable"])
    #################################################################################################################
    rpd_cfg = config["loss"].get("raw_prior_distillation", {})
    rpd_scale_weights_gating_enable = rpd_cfg.get("dynamic_scale_gating", {}).get("enabled", False)
    if rpd_scale_weights_gating_enable:
        rpd_scale_weights = rpd_cfg.get("scale_weights", [0.30, 0.40, 0.30])
        rpd_scale_weights_gating_alpha = rpd_cfg.get("dynamic_scale_gating", {}).get("alpha", 0.0)
        rpd_scale_weights_gate = RPDDynamicScaleWeightsGating(
            in_channels_list=model.online.out_dims,
            base_weights=rpd_scale_weights,
            alpha=rpd_scale_weights_gating_alpha
        ).to(device)
    else:
        rpd_scale_weights_gate = None
    #################################################################################################################
    training_params = list(model.parameters())
    if rpd_scale_weights_gate is not None:
        training_params += list(rpd_scale_weights_gate.parameters())
    
    optimizer = AdamW(
        training_params,
        lr=config["train"]["lr"], 
        weight_decay=config["train"]["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["train"]["epochs"])
    amp_enabled = bool(config["train"].get("amp", False)) and device.type == "cuda"
    print("[AMP] enabled:", amp_enabled)
    scaler = GradScaler(enabled=amp_enabled)

    snapshot_enabled = config["runtime"]["needs_snapshot_teacher"]
    snapshot_freeze_after_epoch = config["runtime"]["snapshot_freeze_after_epoch"]
    if snapshot_enabled and snapshot_freeze_after_epoch == 0 and not model.snapshot_ready:
        model.capture_snapshot_teacher()

    total_epochs = config["train"]["epochs"] + 1
    best_total_loss = float("inf")
    for epoch_index in range(1, config["train"]["epochs"] + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_stats = train_one_epoch(model, dataloader, optimizer, device, config, epoch_index, total_epochs, scaler, amp_enabled, rpd_scale_weights_gate)
        scheduler.step()
        ############################################################################################################
        rpd_sw = epoch_stats.get("rpd_effective_scale_weights", None)
        rpd_sw_str = ("None" if rpd_sw is None else "[" + ", ".join(f"{w:.4f}" for w in rpd_sw) + "]")
        print(
            f"[Epoch {epoch_index:03d}] snapshot_ready={int(getattr(model, 'snapshot_ready', False))}, "
            f"loss_total={epoch_stats['loss_total']:.4f}, "
            f"loss_global={epoch_stats['loss_global']:.4f}, "
            f"loss_local={epoch_stats['loss_local']:.4f}, "
            f"loss_pos={epoch_stats['loss_pos']:.4f}, "
            f"loss_embed={epoch_stats['loss_embed']:.4f}, "
            f"loss_raw_mask={epoch_stats['loss_raw_mask']:.4f}, "
            f"loss_raw_fg_cons={epoch_stats['loss_raw_fg_cons']:.4f}, "
            f"loss_raw_prior={epoch_stats.get('loss_raw_prior', 0.0):.4f}, "
            f"rpd_scale_weights={rpd_sw_str}, "
        )
        ##############################################################################################################
        append_csv_log(log_path, epoch_index, current_lr, getattr(model, "snapshot_ready", False), epoch_stats)

        if snapshot_enabled and (not model.snapshot_ready) and epoch_index == snapshot_freeze_after_epoch:
            model.capture_snapshot_teacher()

        state = {
            "epoch": epoch_index,
            "model": model.state_dict(),
            "rpd_scale_weights_gate": (
                rpd_scale_weights_gate.state_dict()
                if rpd_scale_weights_gate is not None else None
            ),
            "optimizer": optimizer.state_dict(),
            "stats": epoch_stats,
            "config": config,
            "runtime": {
                "snapshot_ready": bool(getattr(model, "snapshot_ready", False)),
                "experiment_name": experiment_name,
            },
        }
        save_checkpoint(state=state, save_dir=save_dir, filename="last.pth")
        if epoch_stats["loss_total"] < best_total_loss and (not snapshot_enabled or (model.snapshot_ready and epoch_index > snapshot_freeze_after_epoch)):
            best_total_loss = epoch_stats["loss_total"]
            save_checkpoint(state=state, save_dir=save_dir, filename="best.pth")


if __name__ == "__main__":
    try:
        main()
    finally:
        from utils.utils import cleanup_memory
        cleanup_memory()
