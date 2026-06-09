import argparse
import os
import json
from pathlib import Path
from typing import Dict, List

import yaml
import torch

from utils.checkpoint import transfer_stage1_backbone_to_yolo, save_transfer_report
from models.yolo_model import YOLO
from models.stage2_spatial_alignment import build_spatial_alignment_trainer
from models.stage2_foreground_prior_distillation import build_fpd_trainer
from models.stage2_leaf_prior_auxiliary import build_leafaux_trainer
from models.stage2_leaf_prior_objectness_transfer import build_lpot_trainer
from models.stage2_detector_coupled import normalize_detector_coupled_cfg, build_detector_coupled_trainer

PROJECT_ROOT = Path(__file__).resolve().parent

BACKBONE_STAGE_GROUPS: Dict[str, List[int]] = {
    "stage_b1": [0, 1, 2, 3, 4],
    "stage_b2": [5, 6],
    "stage_b3": [7, 8],
    "stage_b1_b2": [0, 1, 2, 3, 4, 5, 6],
    "stage_b2_b3": [5, 6, 7, 8],
    "stage_all": [0, 1, 2, 3, 4, 5, 6, 7, 8],
}


def resolve_path(path_str: str, base_dir: Path = None) -> str:
    if not path_str:
        return path_str
    p = Path(path_str)
    if p.is_absolute():
        return str(p.resolve())
    if base_dir is not None:
        return str((base_dir / p).resolve())
    return str((PROJECT_ROOT / p).resolve())


def load_config(config_path: str):
    config_abs = resolve_path(config_path)
    with open(config_abs, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg, config_abs


def normalize_freeze_indices(freeze_cfg: dict) -> List[int]:
    stage_names = freeze_cfg.get("stage_names", []) or []
    explicit = freeze_cfg.get("layer_indices", []) or []

    indices = set(int(i) for i in explicit)
    unknown_stage_names = []
    for name in stage_names:
        if name not in BACKBONE_STAGE_GROUPS:
            unknown_stage_names.append(name)
            continue
        indices.update(BACKBONE_STAGE_GROUPS[name])

    if unknown_stage_names:
        raise ValueError(
            f"Unknown freeze.stage_names: {unknown_stage_names}. "
            f"Available names: {list(BACKBONE_STAGE_GROUPS.keys())}"
        )
    return sorted(indices)


def normalize_spatial_alignment_cfg(spatial_cfg: dict, cfg: dict) -> dict:
    if not spatial_cfg or not spatial_cfg.get("enabled", False):
        return {"enabled": False}

    teacher_ckpt_path = str(spatial_cfg.get("teacher_ckpt_path", "")).strip()
    if not teacher_ckpt_path:
        teacher_ckpt_path = str(cfg.get("stage1_init", {}).get("ckpt_path", "")).strip()
    if not teacher_ckpt_path:
        raise ValueError(
            "spatial_alignment.enabled=true, but neither spatial_alignment.teacher_ckpt_path "
            "nor stage1_init.ckpt_path is provided."
        )

    teacher_ssl_config = str(spatial_cfg.get("teacher_ssl_config", "./configs/stage1_ablation_generated/use_pos_mask.yaml")).strip()
    teacher_branch = str(spatial_cfg.get("teacher_branch", "pos_feats")).strip()
    teacher_feature_indices = [int(x) for x in (spatial_cfg.get("teacher_feature_indices", [0, 1, 2]) or [])]
    student_layer_indices = [int(x) for x in (spatial_cfg.get("student_layer_indices", [4, 6, 8]) or [])]
    layer_weights = [float(x) for x in (spatial_cfg.get("layer_weights", [0.4, 1.0, 0.8]) or [])]
    alpha = float(spatial_cfg.get("alpha", 0.05))
    log_interval = int(spatial_cfg.get("log_interval", 20))

    if len(teacher_feature_indices) == 0:
        raise ValueError("spatial_alignment.teacher_feature_indices cannot be empty")
    if len(teacher_feature_indices) != len(student_layer_indices):
        raise ValueError(
            "spatial_alignment.teacher_feature_indices and student_layer_indices must have the same length. "
            f"Got {teacher_feature_indices} vs {student_layer_indices}"
        )
    if len(layer_weights) != len(teacher_feature_indices):
        raise ValueError(
            "spatial_alignment.layer_weights must match the number of aligned layers. "
            f"Got layer_weights={layer_weights}, layer_count={len(teacher_feature_indices)}"
        )
    if alpha <= 0:
        raise ValueError(f"spatial_alignment.alpha must be > 0, but got {alpha}")

    return {
        "enabled": True,
        "teacher_ssl_config": teacher_ssl_config,
        "teacher_ckpt_path": teacher_ckpt_path,
        "teacher_branch": teacher_branch,
        "teacher_feature_indices": teacher_feature_indices,
        "student_layer_indices": student_layer_indices,
        "layer_weights": layer_weights,
        "alpha": alpha,
        "log_interval": log_interval,
    }


def normalize_fpd_cfg(fpd_cfg: dict, cfg: dict) -> dict:
    if not fpd_cfg or not fpd_cfg.get("enabled", False):
        return {"enabled": False}

    teacher_ckpt_path = str(fpd_cfg.get("teacher_ckpt_path", "")).strip()
    if not teacher_ckpt_path:
        teacher_ckpt_path = str(cfg.get("stage1_init", {}).get("ckpt_path", "")).strip()
    if not teacher_ckpt_path:
        raise ValueError(
            "fpd.enabled=true, but neither fpd.teacher_ckpt_path "
            "nor stage1_init.ckpt_path is provided."
        )

    teacher_ssl_config = str(fpd_cfg.get("teacher_ssl_config", "./configs/stage1_ablation_generated/use_pos_mask.yaml")).strip()
    teacher_branch = str(fpd_cfg.get("teacher_branch", "pos_feats")).strip()
    teacher_feature_indices = [int(x) for x in (fpd_cfg.get("teacher_feature_indices", [0, 1, 2]) or [])]
    student_layer_indices = [int(x) for x in (fpd_cfg.get("student_layer_indices", [4, 6, 8]) or [])]
    layer_weights = [float(x) for x in (fpd_cfg.get("layer_weights", [0.4, 1.0, 0.8]) or [])]
    alpha = float(fpd_cfg.get("alpha", 0.05))
    log_interval = int(fpd_cfg.get("log_interval", 20))

    if len(teacher_feature_indices) == 0:
        raise ValueError("fpd.teacher_feature_indices cannot be empty")
    if len(teacher_feature_indices) != len(student_layer_indices):
        raise ValueError(
            "fpd.teacher_feature_indices and student_layer_indices must have the same length. "
            f"Got {teacher_feature_indices} vs {student_layer_indices}"
        )
    if len(layer_weights) != len(teacher_feature_indices):
        raise ValueError(
            "fpd.layer_weights must match the number of aligned layers. "
            f"Got layer_weights={layer_weights}, layer_count={len(teacher_feature_indices)}"
        )
    if alpha <= 0:
        raise ValueError(f"fpd.alpha must be > 0, but got {alpha}")

    return {
        "enabled": True,
        "teacher_ssl_config": teacher_ssl_config,
        "teacher_ckpt_path": teacher_ckpt_path,
        "teacher_branch": teacher_branch,
        "teacher_feature_indices": teacher_feature_indices,
        "student_layer_indices": student_layer_indices,
        "layer_weights": layer_weights,
        "alpha": alpha,
        "log_interval": log_interval,
        "fg_quantile": float(fpd_cfg.get("fg_quantile", 0.7)),
        "bg_quantile": float(fpd_cfg.get("bg_quantile", 0.3)),
        "lambda_fg": float(fpd_cfg.get("lambda_fg", 1.0)),
        "lambda_bg": float(fpd_cfg.get("lambda_bg", 0.5)),
    }



def normalize_leafaux_cfg(leaf_cfg: dict, cfg: dict) -> dict:
    if not leaf_cfg or not leaf_cfg.get("enabled", False):
        return {"enabled": False}

    teacher_ckpt_path = str(leaf_cfg.get("teacher_ckpt_path", "")).strip()
    if not teacher_ckpt_path:
        teacher_ckpt_path = str(cfg.get("stage1_init", {}).get("ckpt_path", "")).strip()
    if not teacher_ckpt_path:
        raise ValueError(
            "leaf_prior_auxiliary.enabled=true, but neither leaf_prior_auxiliary.teacher_ckpt_path "
            "nor stage1_init.ckpt_path is provided."
        )

    teacher_ssl_config = str(leaf_cfg.get("teacher_ssl_config", "./configs/stage1_ablation_generated/use_pos_mask.yaml")).strip()
    teacher_branch = str(leaf_cfg.get("teacher_branch", "pos_feats")).strip()
    teacher_feature_indices = [int(x) for x in (leaf_cfg.get("teacher_feature_indices", [0, 1, 2]) or [])]
    student_layer_indices = [int(x) for x in (leaf_cfg.get("student_layer_indices", [4, 6, 8]) or [])]
    layer_weights = [float(x) for x in (leaf_cfg.get("layer_weights", [0.4, 1.0, 0.8]) or [])]
    alpha = float(leaf_cfg.get("alpha", 0.03))

    if len(teacher_feature_indices) == 0:
        raise ValueError("leaf_prior_auxiliary.teacher_feature_indices cannot be empty")
    if len(teacher_feature_indices) != len(student_layer_indices):
        raise ValueError(
            "leaf_prior_auxiliary.teacher_feature_indices and student_layer_indices must have the same length. "
            f"Got {teacher_feature_indices} vs {student_layer_indices}"
        )
    if len(layer_weights) != len(teacher_feature_indices):
        raise ValueError(
            "leaf_prior_auxiliary.layer_weights must match the number of aligned layers. "
            f"Got layer_weights={layer_weights}, layer_count={len(teacher_feature_indices)}"
        )
    if alpha <= 0:
        raise ValueError(f"leaf_prior_auxiliary.alpha must be > 0, but got {alpha}")

    return {
        "enabled": True,
        "teacher_ssl_config": teacher_ssl_config,
        "teacher_ckpt_path": teacher_ckpt_path,
        "teacher_branch": teacher_branch,
        "teacher_feature_indices": teacher_feature_indices,
        "student_layer_indices": student_layer_indices,
        "layer_weights": layer_weights,
        "alpha": alpha,
        "gate_init_beta": float(leaf_cfg.get("gate_init_beta", 0.2)),
        "enable_gate": bool(leaf_cfg.get("enable_gate", True)),
        "gamma": float(leaf_cfg.get("gamma", 1.0)),
        "bg_quantile": float(leaf_cfg.get("bg_quantile", 0.3)),
        "lambda_bg": float(leaf_cfg.get("lambda_bg", 0.25)),
        "log_interval": int(leaf_cfg.get("log_interval", 20)),
    }



def normalize_lpot_cfg(lpot_cfg: dict, cfg: dict) -> dict:
    if not lpot_cfg or not lpot_cfg.get("enabled", False):
        return {"enabled": False}

    implementation = str(lpot_cfg.get("implementation", "")).strip()
    route_variant = str(lpot_cfg.get("route_variant", "")).strip()
    resolved_impl = implementation or route_variant or "proposal_support_prior_v1"
    valid_impls = {
        "proposal_support_prior_v1",
        "legacy_lpot_v2",
        "legacy",
        "psp",
        "psp_v1",
        "lpot_v2_legacy",
        "lpotv4_psp_light",
        "teacher_anchored_proxy_support_v1",
        "lpotv5_teacher_anchor",
        "score_level_proposal_support_v1",
        "lpotv41_score_level",
    }
    if resolved_impl not in valid_impls:
        raise ValueError(
            f"Unknown leaf_prior_objectness_transfer implementation/route_variant: {resolved_impl}. "
            f"Supported: {sorted(valid_impls)}"
        )

    teacher_ckpt_path = str(lpot_cfg.get("teacher_ckpt_path", "")).strip()
    if not teacher_ckpt_path:
        teacher_ckpt_path = str(cfg.get("stage1_init", {}).get("ckpt_path", "")).strip()
    if not teacher_ckpt_path:
        raise ValueError(
            "leaf_prior_objectness_transfer.enabled=true, but neither leaf_prior_objectness_transfer.teacher_ckpt_path "
            "nor stage1_init.ckpt_path is provided."
        )

    teacher_ssl_config = str(lpot_cfg.get("teacher_ssl_config", "./configs/stage1_ablation_generated/use_pos_mask.yaml")).strip()
    teacher_branch = str(lpot_cfg.get("teacher_branch", "pos_feats")).strip()
    teacher_feature_indices = [int(x) for x in (lpot_cfg.get("teacher_feature_indices", [0, 1, 2]) or [])]
    student_layer_indices = [int(x) for x in (lpot_cfg.get("student_layer_indices", [4, 6, 8]) or [])]
    proxy_layer_indices = [int(x) for x in (lpot_cfg.get("proxy_layer_indices", [15, 18, 21]) or [])]
    layer_weights = [float(x) for x in (lpot_cfg.get("layer_weights", [0.4, 1.0, 0.8]) or [])]
    proxy_layer_weights = [float(x) for x in (lpot_cfg.get("proxy_layer_weights", [1.0, 1.0, 1.0]) or [])]
    teacher_fuse_weights = [float(x) for x in (lpot_cfg.get("teacher_fuse_weights", layer_weights) or [])]
    alpha = float(lpot_cfg.get("alpha", 1.0))

    if len(teacher_feature_indices) == 0:
        raise ValueError("leaf_prior_objectness_transfer.teacher_feature_indices cannot be empty")
    if len(teacher_feature_indices) != len(student_layer_indices):
        raise ValueError(
            "leaf_prior_objectness_transfer.teacher_feature_indices and student_layer_indices must have the same length. "
            f"Got {teacher_feature_indices} vs {student_layer_indices}"
        )
    if len(layer_weights) != len(teacher_feature_indices):
        raise ValueError(
            "leaf_prior_objectness_transfer.layer_weights must match the number of aligned prior layers. "
            f"Got layer_weights={layer_weights}, layer_count={len(teacher_feature_indices)}"
        )
    if len(proxy_layer_weights) != len(proxy_layer_indices):
        raise ValueError(
            "leaf_prior_objectness_transfer.proxy_layer_weights must match proxy_layer_indices length. "
            f"Got proxy_layer_weights={proxy_layer_weights}, proxy_layers={proxy_layer_indices}"
        )
    if len(teacher_fuse_weights) != len(teacher_feature_indices):
        raise ValueError(
            "leaf_prior_objectness_transfer.teacher_fuse_weights must match teacher_feature_indices length. "
            f"Got teacher_fuse_weights={teacher_fuse_weights}, teacher_features={teacher_feature_indices}"
        )
    if alpha <= 0:
        raise ValueError(f"leaf_prior_objectness_transfer.alpha must be > 0, but got {alpha}")

    return {
        "enabled": True,
        "implementation": resolved_impl,
        "route_variant": resolved_impl,
        "teacher_ssl_config": teacher_ssl_config,
        "teacher_ckpt_path": teacher_ckpt_path,
        "teacher_branch": teacher_branch,
        "teacher_feature_indices": teacher_feature_indices,
        "student_layer_indices": student_layer_indices,
        "proxy_layer_indices": proxy_layer_indices,
        "layer_weights": layer_weights,
        "proxy_layer_weights": proxy_layer_weights,
        "teacher_fuse_weights": teacher_fuse_weights,
        "alpha": alpha,
        "gate_init_beta": float(lpot_cfg.get("gate_init_beta", 0.2)),
        "proxy_gate_init_beta": float(lpot_cfg.get("proxy_gate_init_beta", lpot_cfg.get("gate_init_beta", 0.2))),
        "support_gate_gain": float(lpot_cfg.get("support_gate_gain", 1.0)),
        "proxy_support_gate_gain": float(lpot_cfg.get("proxy_support_gate_gain", lpot_cfg.get("support_gate_gain", 1.0))),
        "enable_feature_gate": bool(lpot_cfg.get("enable_feature_gate", True)),
        "enable_proxy_gate": bool(lpot_cfg.get("enable_proxy_gate", True)),
        "gamma": float(lpot_cfg.get("gamma", 1.0)),
        "fg_quantile": float(lpot_cfg.get("fg_quantile", 0.7)),
        "bg_quantile": float(lpot_cfg.get("bg_quantile", 0.3)),
        "lambda_bg": float(lpot_cfg.get("lambda_bg", 0.25)),
        "lambda_rank": float(lpot_cfg.get("lambda_rank", 0.20)),
        "lambda_align": float(lpot_cfg.get("lambda_align", 1.0)),
        "support_margin": float(lpot_cfg.get("support_margin", 0.10)),
        "lambda_prior": float(lpot_cfg.get("lambda_prior", 1.0)),
        "lambda_proxy": float(lpot_cfg.get("lambda_proxy", 0.5)),
        "lambda_bridge": float(lpot_cfg.get("lambda_bridge", 0.4)),
        "objectness_student_blend": float(lpot_cfg.get("objectness_student_blend", 0.60)),
        "objectness_smooth_kernel": int(lpot_cfg.get("objectness_smooth_kernel", 7)),
        "objectness_smooth_iters": int(lpot_cfg.get("objectness_smooth_iters", 2)),
        "lambda_teacher_rank": float(lpot_cfg.get("lambda_teacher_rank", 0.0)),
        "lambda_teacher_bg": float(lpot_cfg.get("lambda_teacher_bg", 0.0)),
        "teacher_anchor_margin": (
            float(lpot_cfg["teacher_anchor_margin"])
            if lpot_cfg.get("teacher_anchor_margin", None) is not None
            else None
        ),
        "teacher_fg_quantile": (
            float(lpot_cfg["teacher_fg_quantile"])
            if lpot_cfg.get("teacher_fg_quantile", None) is not None
            else None
        ),
        "teacher_bg_quantile": (
            float(lpot_cfg["teacher_bg_quantile"])
            if lpot_cfg.get("teacher_bg_quantile", None) is not None
            else None
        ),
        "teacher_anchor_layer_weights": [float(x) for x in (lpot_cfg.get("teacher_anchor_layer_weights", []) or [])],
        "lambda_score_rank": float(lpot_cfg.get("lambda_score_rank", 0.0)),
        "lambda_score_bg": float(lpot_cfg.get("lambda_score_bg", 0.0)),
        "score_rank_margin": float(lpot_cfg.get("score_rank_margin", 0.10)),
        "score_fg_quantile": float(lpot_cfg.get("score_fg_quantile", lpot_cfg.get("fg_quantile", 0.70))),
        "score_bg_quantile": float(lpot_cfg.get("score_bg_quantile", lpot_cfg.get("bg_quantile", 0.30))),
        "score_rank_layer_weights": [float(x) for x in (lpot_cfg.get("score_rank_layer_weights", []) or [])],
        "score_gt_expand_ratio": float(lpot_cfg.get("score_gt_expand_ratio", 1.5)),
        "score_use_gt_pos": bool(lpot_cfg.get("score_use_gt_pos", True)),
        "score_use_teacher_fg_pos": bool(lpot_cfg.get("score_use_teacher_fg_pos", False)),
        "log_interval": int(lpot_cfg.get("log_interval", 20)),
    }


def _iter_named_backbone_params(det_model, max_layer_idx: int = 8):
    """Yield backbone-like YOLO parameters up to `max_layer_idx`.

    Ultralytics parameter names are usually like `model.0.conv.weight`.
    This helper is intentionally conservative and only checks layer indices
    under `model.<idx>.*`, which matches the current YOLOv8/YOLO-family layout.
    """
    for name, p in det_model.named_parameters():
        if not torch.is_tensor(p):
            continue
        if not name.startswith("model."):
            continue
        parts = name.split(".")
        if len(parts) < 2:
            continue
        try:
            layer_idx = int(parts[1])
        except Exception:
            continue
        if layer_idx <= int(max_layer_idx):
            yield name, p


def collect_model_weight_fingerprint(det_model) -> Dict[str, float | int]:
    """Collect a lightweight numeric fingerprint of all trainable tensors."""
    total_abs = 0.0
    total_sq = 0.0
    total_numel = 0
    tensor_count = 0
    for _name, p in det_model.named_parameters():
        if not torch.is_tensor(p):
            continue
        t = p.detach().float().cpu()
        total_abs += float(t.abs().sum())
        total_sq += float((t * t).sum())
        total_numel += int(t.numel())
        tensor_count += 1
    return {
        "total_abs_sum": total_abs,
        "total_l2_sum": total_sq ** 0.5,
        "total_numel": total_numel,
        "tensor_count": tensor_count,
    }


def collect_backbone_weight_fingerprint(det_model, max_layer_idx: int = 8) -> Dict[str, float | int]:
    """Collect a numeric fingerprint for YOLO backbone-like layers."""
    total_abs = 0.0
    total_sq = 0.0
    total_numel = 0
    tensor_count = 0
    for _name, p in _iter_named_backbone_params(det_model, max_layer_idx=max_layer_idx):
        t = p.detach().float().cpu()
        total_abs += float(t.abs().sum())
        total_sq += float((t * t).sum())
        total_numel += int(t.numel())
        tensor_count += 1
    return {
        "backbone_abs_sum": total_abs,
        "backbone_l2_sum": total_sq ** 0.5,
        "backbone_numel": total_numel,
        "backbone_tensor_count": tensor_count,
        "max_layer_idx": int(max_layer_idx),
    }


def snapshot_backbone_params(det_model, max_layer_idx: int = 8) -> Dict[str, torch.Tensor]:
    """Clone backbone-like parameters before Stage1 transfer for exact diff checking."""
    return {
        name: p.detach().cpu().clone()
        for name, p in _iter_named_backbone_params(det_model, max_layer_idx=max_layer_idx)
    }


def compare_param_snapshot(before: Dict[str, torch.Tensor], det_model) -> Dict:
    """Compare current detector parameters with a previous snapshot."""
    current = dict(det_model.named_parameters())
    changed_count = 0
    same_count = 0
    missing_count = 0
    total_abs_diff = 0.0
    max_abs_diff = 0.0
    rows = []

    for name, before_tensor in before.items():
        now = current.get(name, None)
        if now is None:
            missing_count += 1
            rows.append({
                "name": name,
                "status": "missing_after_transfer",
                "abs_diff_sum": None,
                "abs_diff_max": None,
                "numel": int(before_tensor.numel()),
            })
            continue
        diff = (now.detach().cpu() - before_tensor).float().abs()
        abs_sum = float(diff.sum())
        abs_max = float(diff.max()) if diff.numel() else 0.0
        total_abs_diff += abs_sum
        max_abs_diff = max(max_abs_diff, abs_max)
        if abs_sum > 0.0:
            changed_count += 1
            status = "changed"
        else:
            same_count += 1
            status = "same"
        rows.append({
            "name": name,
            "status": status,
            "abs_diff_sum": abs_sum,
            "abs_diff_max": abs_max,
            "numel": int(diff.numel()),
        })

    rows = sorted(rows, key=lambda x: (-1.0 if x["abs_diff_sum"] is None else x["abs_diff_sum"]), reverse=True)
    return {
        "changed_param_tensors": int(changed_count),
        "same_param_tensors": int(same_count),
        "missing_param_tensors": int(missing_count),
        "total_abs_diff": float(total_abs_diff),
        "max_abs_diff": float(max_abs_diff),
        "top_changed": [r for r in rows if r.get("status") == "changed"][:30],
        "top_all": rows[:30],
    }


def write_json(path: str | Path, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def maybe_transfer_stage1_weights(yolo: YOLO, cfg: dict, save_dir: str, base_dir: Path):
    stage1_cfg = cfg.get("stage1_init", {})
    use_stage1 = bool(stage1_cfg.get("enabled", False))

    if not use_stage1:
        print("[Stage2] Stage1 initialization disabled. Using detector default init.")
        return None

    ckpt_path_raw = stage1_cfg.get("ckpt_path", "")
    if not ckpt_path_raw:
        raise ValueError("stage1_init.enabled=true, but stage1_init.ckpt_path is empty.")

    ckpt_path = resolve_path(ckpt_path_raw, base_dir=base_dir)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"[Stage2] Stage1 checkpoint not found: {ckpt_path}")
    
    trained_layer_max = max(tuple(stage1_cfg.get("layer_indices", [4, 6, 8])))

    _, report = transfer_stage1_backbone_to_yolo(
        det_model=yolo.model,
        stage1_ckpt_path=ckpt_path,
        map_location="cpu",
        verbose=True,
        trained_layer_max=trained_layer_max
    )

    report_path = os.path.join(save_dir, "stage1_transfer_report.json")
    save_transfer_report(report, report_path)
    print(f"[Stage2] Transfer report saved to: {report_path}")
    return report


def build_train_kwargs(cfg: dict, run_name: str, base_dir: Path):
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    freeze_cfg = cfg.get("freeze", {})

    project_abs = resolve_path(train_cfg.get("project", "runs/glcp_stage2_yolo_det"), base_dir=base_dir)
    data_yaml_abs = resolve_path(data_cfg["data_yaml"], base_dir=base_dir)
    exp_dir = resolve_path(train_cfg.get("name", "baseline"), base_dir=Path(project_abs))
    save_runtime_config(cfg, save_dir=exp_dir)

    if not os.path.isfile(data_yaml_abs):
        raise FileNotFoundError(f"[Stage2] data_yaml not found: {data_yaml_abs}")

    kwargs = dict(
        data=data_yaml_abs,
        epochs=train_cfg["epochs"],
        imgsz=data_cfg.get("imgsz", 640),
        batch=train_cfg.get("batch_size", 16),
        device=train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        workers=data_cfg.get("workers", 4),
        project=project_abs,
        name=run_name,
        exist_ok=train_cfg.get("exist_ok", True),
        pretrained=False,
        optimizer=train_cfg.get("optimizer", "AdamW"),
        lr0=train_cfg.get("lr0", 1e-3),
        lrf=train_cfg.get("lrf", 1e-2),
        momentum=train_cfg.get("momentum", 0.937),
        weight_decay=train_cfg.get("weight_decay", 5e-4),
        warmup_epochs=train_cfg.get("warmup_epochs", 3.0),
        warmup_momentum=train_cfg.get("warmup_momentum", 0.8),
        warmup_bias_lr=train_cfg.get("warmup_bias_lr", 0.1),
        cos_lr=train_cfg.get("cos_lr", True),
        close_mosaic=train_cfg.get("close_mosaic", 10),
        hsv_h=train_cfg.get("hsv_h", 0.015),
        hsv_s=train_cfg.get("hsv_s", 0.7),
        hsv_v=train_cfg.get("hsv_v", 0.4),
        degrees=train_cfg.get("degrees", 0.0),
        translate=train_cfg.get("translate", 0.1),
        scale=train_cfg.get("scale", 0.5),
        shear=train_cfg.get("shear", 0.0),
        perspective=train_cfg.get("perspective", 0.0),
        flipud=train_cfg.get("flipud", 0.0),
        fliplr=train_cfg.get("fliplr", 0.5),
        mosaic=train_cfg.get("mosaic", 1.0),
        mixup=train_cfg.get("mixup", 0.0),
        copy_paste=train_cfg.get("copy_paste", 0.0),
        amp=train_cfg.get("amp", True),
        patience=train_cfg.get("patience", 50),
        val=train_cfg.get("val", True),
        save=train_cfg.get("save", True),
        save_period=train_cfg.get("save_period", -1),
        single_cls=model_cfg.get("single_cls", False),
        rect=train_cfg.get("rect", False),
        plots=train_cfg.get("plots", True),
        verbose=True,
        seed=train_cfg.get("seed", 42),
    )

    if freeze_cfg.get("enabled", False):
        freeze_indices = normalize_freeze_indices(freeze_cfg)
        kwargs["freeze"] = freeze_indices
        print(f"[Stage2] Freeze enabled via YAML -> train_kwargs['freeze'] = {freeze_indices}")
    else:
        print("[Stage2] Freeze disabled in YAML.")

    return kwargs


def save_runtime_config(config: Dict, save_dir: str) -> None:
    config_path = os.path.join(save_dir, "config_used.yaml")
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/det_config.yaml")
    args = parser.parse_args()

    cfg, config_abs = load_config(args.config)

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    freeze_cfg = cfg.get("freeze", {})
    spatial_cfg = normalize_spatial_alignment_cfg(cfg.get("spatial_alignment", {}), cfg)
    fpd_cfg = normalize_fpd_cfg(cfg.get("foreground_prior_distillation", {}), cfg)
    leafaux_cfg = normalize_leafaux_cfg(cfg.get("leaf_prior_auxiliary", {}), cfg)
    lpot_cfg = normalize_lpot_cfg(cfg.get("leaf_prior_objectness_transfer", {}), cfg)
    dc_cfg = normalize_detector_coupled_cfg(cfg.get("detector_coupled", {}), cfg)
    # Save normalized DC config into config_used.yaml
    cfg["detector_coupled"] = dc_cfg

    run_name = train_cfg.get("name", "stage2_det")
    project_dir = resolve_path(train_cfg.get("project", "runs/glcp_stage2_yolo_det"), base_dir=PROJECT_ROOT)
    save_dir = os.path.join(project_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    freeze_indices = normalize_freeze_indices(freeze_cfg) if freeze_cfg.get("enabled", False) else []
    ckpt_path = cfg.get("stage1_init", {}).get("ckpt_path", None)
    if ckpt_path: ckpt_path = Path(ckpt_path).resolve()

    print("=" * 80)
    print("[Stage2] Config loaded:", config_abs)
    print("[Stage2] Project dir  :", project_dir)
    print("[Stage2] Run name     :", run_name)
    print("[Stage2] Save dir     :", save_dir)
    print("[Stage2] ckpt path    :", ckpt_path)
    print("[Stage2] freeze       :", freeze_cfg.get("enabled", False))
    print("[Stage2] freeze names :", freeze_cfg.get("stage_names", []))
    print("[Stage2] freeze idx   :", freeze_indices)
    print("[Stage2] spatial algn:", spatial_cfg)
    print("[Stage2] fpd          :", fpd_cfg)
    print("[Stage2] leafaux      :", leafaux_cfg)
    print("[Stage2] lpot         :", lpot_cfg)
    print("[Stage2] detector DC  :", dc_cfg)
    print("[Stage2] stage groups :", BACKBONE_STAGE_GROUPS)
    print("=" * 80)

    enabled_aux = sum([
        1 if spatial_cfg.get("enabled", False) else 0,
        1 if fpd_cfg.get("enabled", False) else 0,
        1 if leafaux_cfg.get("enabled", False) else 0,
        1 if lpot_cfg.get("enabled", False) else 0,
    ])
    if enabled_aux > 1:
        raise ValueError(
            "Only one auxiliary stage2 feature-transfer method should be enabled at a time: "
            "spatial_alignment / fpd / leafaux / lpot."
        )
    
    if dc_cfg.get("enabled", False) and freeze_cfg.get("enabled", False) and dc_cfg.get("strict_no_yaml_freeze", True):
        raise ValueError(
            "detector_coupled.enabled=true conflicts with freeze.enabled=true. "
            "DC uses layer-wise LR and controlled trainability, so disable YAML freeze for DC experiments."
        )

    yolo = YOLO(model_cfg["yolo_model"])

    backbone_max_layer_idx = int(cfg.get("stage1_init", {}).get("backbone_max_layer_idx", 8))
    before_all = collect_model_weight_fingerprint(yolo.model)
    before_backbone = collect_backbone_weight_fingerprint(yolo.model, max_layer_idx=backbone_max_layer_idx)
    before_backbone_snapshot = snapshot_backbone_params(yolo.model, max_layer_idx=backbone_max_layer_idx)

    transfer_report = maybe_transfer_stage1_weights(yolo, cfg, save_dir, PROJECT_ROOT)

    after_all = collect_model_weight_fingerprint(yolo.model)
    after_backbone = collect_backbone_weight_fingerprint(yolo.model, max_layer_idx=backbone_max_layer_idx)
    backbone_diff = compare_param_snapshot(before_backbone_snapshot, yolo.model)

    stage1_init_cfg = cfg.get("stage1_init", {}) or {}
    force_train_from_transferred_model = bool(
        stage1_init_cfg.get("enabled", False)
        and stage1_init_cfg.get("force_train_from_transferred_model", True)
        and transfer_report is not None
    )
    init_after_transfer_path = ""
    reloaded_all = None
    reloaded_backbone = None
    reloaded_backbone_diff = None

    if force_train_from_transferred_model:
        init_after_transfer_path = os.path.join(save_dir, "init_after_stage1_transfer.pt")
        print(f"[Stage2] Saving transferred initialization model to: {init_after_transfer_path}")
        yolo.save(init_after_transfer_path)

        # Important: reload the transferred model and train from this .pt file.
        # This avoids any Ultralytics-side reinitialization from the original model path.
        yolo = YOLO(init_after_transfer_path)
        reloaded_all = collect_model_weight_fingerprint(yolo.model)
        reloaded_backbone = collect_backbone_weight_fingerprint(yolo.model, max_layer_idx=backbone_max_layer_idx)
        reloaded_backbone_diff = compare_param_snapshot(before_backbone_snapshot, yolo.model)
        print(
            "[Stage2] Reloaded transferred init summary: "
            f"changed={reloaded_backbone_diff['changed_param_tensors']}, "
            f"total_abs_diff={reloaded_backbone_diff['total_abs_diff']:.6f}, "
            f"max_abs_diff={reloaded_backbone_diff['max_abs_diff']:.6f}"
        )

    init_check = {
        "model_yolo_model": model_cfg.get("yolo_model", ""),
        "stage1_init_enabled": bool(stage1_init_cfg.get("enabled", False)),
        "stage1_ckpt_path_raw": stage1_init_cfg.get("ckpt_path", ""),
        "stage1_ckpt_path_resolved": resolve_path(stage1_init_cfg.get("ckpt_path", ""), base_dir=PROJECT_ROOT),
        "force_train_from_transferred_model": force_train_from_transferred_model,
        "init_after_transfer_path": init_after_transfer_path,
        "train_start_model": init_after_transfer_path if force_train_from_transferred_model else model_cfg.get("yolo_model", ""),
        "backbone_max_layer_idx": backbone_max_layer_idx,
        "transfer_report": transfer_report,
        "before_all": before_all,
        "after_all": after_all,
        "before_backbone": before_backbone,
        "after_backbone": after_backbone,
        "reloaded_all": reloaded_all,
        "reloaded_backbone": reloaded_backbone,
        "delta_all_abs_sum": float(after_all["total_abs_sum"] - before_all["total_abs_sum"]),
        "delta_backbone_abs_sum": float(after_backbone["backbone_abs_sum"] - before_backbone["backbone_abs_sum"]),
        "backbone_diff_summary": {
            "changed_param_tensors": backbone_diff["changed_param_tensors"],
            "same_param_tensors": backbone_diff["same_param_tensors"],
            "missing_param_tensors": backbone_diff["missing_param_tensors"],
            "total_abs_diff": backbone_diff["total_abs_diff"],
            "max_abs_diff": backbone_diff["max_abs_diff"],
        },
        "reloaded_backbone_diff_summary": (
            {
                "changed_param_tensors": reloaded_backbone_diff["changed_param_tensors"],
                "same_param_tensors": reloaded_backbone_diff["same_param_tensors"],
                "missing_param_tensors": reloaded_backbone_diff["missing_param_tensors"],
                "total_abs_diff": reloaded_backbone_diff["total_abs_diff"],
                "max_abs_diff": reloaded_backbone_diff["max_abs_diff"],
            }
            if reloaded_backbone_diff is not None else None
        ),
    }
    init_check_path = os.path.join(save_dir, "stage2_init_check.json")
    diff_check_path = os.path.join(save_dir, "stage2_transfer_diff_check.json")
    write_json(init_check_path, init_check)
    write_json(diff_check_path, backbone_diff)
    print(f"[Stage2] Init check saved to: {init_check_path}")
    print(f"[Stage2] Transfer diff check saved to: {diff_check_path}")
    print(
        "[Stage2] Transfer diff summary: "
        f"changed={backbone_diff['changed_param_tensors']}, "
        f"same={backbone_diff['same_param_tensors']}, "
        f"total_abs_diff={backbone_diff['total_abs_diff']:.6f}, "
        f"max_abs_diff={backbone_diff['max_abs_diff']:.6f}"
    )

    train_kwargs = build_train_kwargs(cfg, run_name=run_name, base_dir=PROJECT_ROOT)
    trainer_det = None

    if dc_cfg.get("enabled", False):
        trainer_det = build_detector_coupled_trainer(
            dc_cfg=dc_cfg,
            project_root=PROJECT_ROOT,
            base_trainer_det=trainer_det,
        )
        print("[Stage2] Detector-coupled fine-tuning trainer enabled.")

    if spatial_cfg.get("enabled", False):
        if trainer_det is None:
            trainer_det = build_spatial_alignment_trainer(
                spatial_cfg=spatial_cfg,
                project_root=PROJECT_ROOT,
            )
        else:
            trainer_det = build_spatial_alignment_trainer(
                spatial_cfg=spatial_cfg,
                project_root=PROJECT_ROOT,
                base_trainer_det=trainer_det,
            )
        print("[Stage2] Spatial map alignment trainer enabled.")

    if fpd_cfg.get("enabled", False):
        if trainer_det is None:
            trainer_det = build_fpd_trainer(
                fpd_cfg=fpd_cfg,
                project_root=PROJECT_ROOT,
            )
        else:
            trainer_det = build_fpd_trainer(
                fpd_cfg=fpd_cfg,
                project_root=PROJECT_ROOT,
                base_trainer_cls=trainer_det,
            )
        print("[Stage2] Foreground Prior Distillation trainer enabled.")

    if leafaux_cfg.get("enabled", False):
        if trainer_det is None:
            trainer_det = build_leafaux_trainer(
                leafaux_cfg=leafaux_cfg,
                project_root=PROJECT_ROOT,
            )
        else:
            trainer_det = build_leafaux_trainer(
                leafaux_cfg=leafaux_cfg,
                project_root=PROJECT_ROOT,
                base_trainer_det=trainer_det,
            )
        print("[Stage2] Leaf-Prior Auxiliary Head trainer enabled.")


    if lpot_cfg.get("enabled", False):
        if trainer_det is None:
            trainer_det = build_lpot_trainer(
                lpot_cfg=lpot_cfg,
                project_root=PROJECT_ROOT,
            )
        else:
            trainer_det = build_lpot_trainer(
                lpot_cfg=lpot_cfg,
                project_root=PROJECT_ROOT,
                base_trainer_det=trainer_det,
            )
        print("[Stage2] LPOT trainer enabled.")

    if trainer_det is not None:
        train_kwargs["trainer"] = trainer_det

    print("[Stage2] Training kwargs summary:")
    for k in ["data", "epochs", "imgsz", "batch", "device", "project", "name", "optimizer", "lr0", "weight_decay"]:
        print(f"  {k}: {train_kwargs[k]}")
    if "freeze" in train_kwargs:
        print(f"  freeze: {train_kwargs['freeze']}")
    if "trainer" in train_kwargs:
        print(f"  trainer: {train_kwargs['trainer']}")
    print("=" * 80)

    train_kwargs["pretrained"]=True
    train_kwargs["resume"] = False
    train_kwargs["plots"] = False
    print("="*80)
    print("[Stage2-DEBUG] run_name:", run_name, flush=True)
    print("[Stage2-DEBUG] epochs  :", train_kwargs.get("epochs"), flush=True)
    print("[Stage2-DEBUG] patience:", train_kwargs.get("patience"), flush=True)
    print("[Stage2-DEBUG] resume  :", train_kwargs.get("resume"), flush=True)
    print("[Stage2-DEBUG] time    :", train_kwargs.get("time"), flush=True)
    print("[Stage2-DEBUG] project :", train_kwargs.get("project"), flush=True)
    print("[Stage2-DEBUG] name    :", train_kwargs.get("name"), flush=True)
    print("[Stage2-DEBUG] plots   :", train_kwargs.get("plots"), flush=True)
    print("="*80)
    try:
        yolo.train(**train_kwargs)
        print("[Stage2-DEBUG] after yolo.train(): normal return", flush=True)
    except BaseException as e:
        import traceback
        print("[Stage2-DEBUG] yolo.train() crashed:", repr(e), flush=True)
        traceback.print_exc()
        raise e
    finally:
        print("[Stage2-DEBUG] yolo.train() finally reached", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        from utils.utils import cleanup_memory
        cleanup_memory()