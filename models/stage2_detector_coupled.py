"""
Detector-coupled fine-tuning for Stage2 YOLO training.

Place this file in your project as:
    models/stage2_detector_coupled.py

Design goal:
    Keep the existing train_stage2.py structure intact. This module only provides
    (1) config normalization and (2) a custom Trainer class whose optimizer uses
    layer-wise learning-rate ratios.

Main idea:
    Do NOT full-freeze the FGCRL-pretrained backbone. Instead, use controlled,
    layer-wise LR so that Stage1 foreground-aware features can be coupled to
    Stage2 detection outputs through detection gradients.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from ultralytics.models.yolo.detect import DetectionTrainer  # type: ignore


DEFAULT_LAYER_GROUPS_BY_PROFILE = {
    "yolov8_det": {
        "shallow": [0, 1, 2, 3, 4],
        "middle": [5, 6],
        "high": [7, 8],
        "neck": list(range(9, 22)),
        "head": [22],
    },

    "yolov9_det": {
        "shallow": [0, 1, 2, 3, 4],
        "middle": [5, 6],
        "high": [7, 8],
        "neck": list(range(9, 22)),
        "head": [22],
    },

    "yolov10_det": {
        "shallow": [0, 1, 2, 3, 4],
        "middle": [5, 6],
        "high": [7, 8],
        "neck": list(range(9, 23)),
        "head": [23],
    },

    "yolov11_det": {
        "shallow": [0, 1, 2, 3, 4],
        "middle": [5, 6],
        "high": [7, 8],
        "neck": list(range(9, 23)),
        "head": [23],
    },
}


def _as_int_list(value: Any, default: Sequence[int]) -> List[int]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        if not value.strip():
            return []
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


def infer_yolo_profile(cfg: dict) -> str:
    model_path = str(cfg.get("model", {}).get("yolo_model", "")).lower()

    if "yolov8" in model_path or "yolo8" in model_path:
        return "yolov8_det"

    if "yolov9" in model_path or "yolo9" in model_path:
        return "yolov9_det"

    if "yolov10" in model_path or "yolo10" in model_path:
        return "yolov10_det"

    if "yolov11" in model_path or "yolo11" in model_path:
        return "yolov11_det"

    return None


def normalize_detector_coupled_cfg(dc_cfg: dict, cfg: dict) -> dict:
    if not dc_cfg or not dc_cfg.get("enabled", False):
        return {"enabled": False}

    preset = str(dc_cfg.get("preset", "default")).strip().lower()
    model_profile = str(dc_cfg.get("model_profile", "auto")).strip().lower()
    if model_profile == "auto": model_profile = infer_yolo_profile(cfg)
    if model_profile is None:
        raise ValueError(
            f"Unknown detector_coupled.model_profile={model_profile}. "
            f"Available model_profile: yolov8_det/yolov9_det/yolov10_det/yolov11_det."
        ) 

    preset_ratios = {
        "default": {
            "shallow": 0.10,
            "middle": 0.10,
            "high": 0.10,
            "neck": 1.00,
            "head": 1.00,
            "others": 1.00,
        },
        "conservative": {
            "shallow": 0.05,
            "middle": 0.05,
            "high": 0.05,
            "neck": 1.00,
            "head": 1.00,
            "others": 1.00,
        },
        "stronger": {
            "shallow": 0.20,
            "middle": 0.20,
            "high": 0.20,
            "neck": 1.00,
            "head": 1.00,
            "others": 1.00,
        },
    }

    if preset not in preset_ratios:
        raise ValueError(
            f"Unknown detector_coupled.preset={preset}. "
            f"Available presets: {list(preset_ratios.keys())}"
        )

    default_layer_groups = DEFAULT_LAYER_GROUPS_BY_PROFILE.get(model_profile, None)
    layer_groups = dc_cfg.get("layer_groups", default_layer_groups)
    if layer_groups is None:
        raise ValueError(
            f"Unknown detector_coupled.layer_groups={layer_groups}. "
            f"Please check you config yaml."
        )

    return {
        "enabled": True,
        "preset": preset,
        "model_profile": model_profile,
        "optimizer": str(dc_cfg.get("optimizer", "AdamW")),
        "base_lr_from_train": bool(dc_cfg.get("base_lr_from_train", True)),
        "weight_decay_from_train": bool(dc_cfg.get("weight_decay_from_train", True)),
        "momentum_from_train": bool(dc_cfg.get("momentum_from_train", True)),
        "train_shallow": bool(dc_cfg.get("train_shallow", False)),
        "layer_groups": layer_groups,
        "lr_ratios": dc_cfg.get("lr_ratios", preset_ratios[preset]),
        "log_param_groups": bool(dc_cfg.get("log_param_groups", True)),
        "save_param_group_report": bool(dc_cfg.get("save_param_group_report", True)),
        "strict_no_yaml_freeze": bool(dc_cfg.get("strict_no_yaml_freeze", True)),
    }


def _get_yolo_layer_list(model: torch.nn.Module):
    """Return the Ultralytics YOLO layer list from a trainer model."""
    if hasattr(model, "model"):
        layers = model.model
    else:
        raise AttributeError("[DC] Cannot find model.model layer list in YOLO model.")
    return layers


def _get_layer_id_from_param_name(name: str) -> int | None:
    """Parse top-level YOLO layer id from names like 'model.7.cv1.conv.weight'."""
    if not name.startswith("model."):
        return None
    parts = name.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except Exception:
        return None


def _assign_group(layer_id: int | None, layer_groups: Dict[str, List[int]]) -> str:
    if layer_id is None:
        return "others"
    for group_name in ("shallow", "middle", "high", "neck", "head"):
        if layer_id in set(layer_groups.get(group_name, [])):
            return group_name
    return "others"


def _set_requires_grad_by_groups(model: torch.nn.Module, dc_cfg: Dict[str, Any]) -> None:
    layer_groups = dc_cfg["layer_groups"]
    train_shallow = bool(dc_cfg.get("train_shallow", False))
    shallow_ids = set(layer_groups.get("shallow", []))

    # Start with all params trainable except explicitly frozen shallow group.
    for name, p in model.named_parameters():
        layer_id = _get_layer_id_from_param_name(name)
        if layer_id in shallow_ids and not train_shallow:
            p.requires_grad = False
        else:
            p.requires_grad = True


def _make_optimizer(param_groups: List[Dict[str, Any]], optimizer_name: str, lr: float, momentum: float, weight_decay: float):
    name = str(optimizer_name).lower()
    if name == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(param_groups, lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    raise ValueError(f"[DC] Unsupported optimizer: {optimizer_name}. Use AdamW, Adam, or SGD.")


def _build_param_groups(model: torch.nn.Module, dc_cfg: Dict[str, Any], lr: float, decay: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    _set_requires_grad_by_groups(model, dc_cfg)

    layer_groups = dc_cfg["layer_groups"]
    ratios = dc_cfg["lr_ratios"]

    grouped: Dict[str, List[torch.nn.Parameter]] = {
        "shallow": [],
        "middle": [],
        "high": [],
        "neck": [],
        "head": [],
        "others": [],
    }

    # Parameter-level grouping is more robust than layer.modules() grouping because
    # it works even when Ultralytics wraps modules internally.
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        layer_id = _get_layer_id_from_param_name(name)
        group_name = _assign_group(layer_id, layer_groups)
        grouped[group_name].append(p)

    param_groups: List[Dict[str, Any]] = []
    report = {
        "enabled": True,
        "base_lr": float(lr),
        "weight_decay": float(decay),
        "optimizer_groups": [],
        "frozen_layers": layer_groups.get("shallow", []) if not dc_cfg.get("train_shallow", False) else [],
        "layer_groups": layer_groups,
        "lr_ratios": ratios,
    }

    for group_name in ["shallow", "middle", "high", "neck", "head", "others"]:
        params = grouped[group_name]
        if not params:
            continue
        group_lr = float(lr) * float(ratios.get(group_name, 1.0))
        group = {
            "params": params,
            "lr": group_lr,
            "initial_lr": group_lr,  # important for Ultralytics/LambdaLR to preserve ratios
            "weight_decay": float(decay),
            "name": group_name,
            "dc_lr_ratio": float(ratios.get(group_name, 1.0)),
        }
        param_groups.append(group)
        report["optimizer_groups"].append({
            "name": group_name,
            "lr": group_lr,
            "lr_ratio": float(ratios.get(group_name, 1.0)),
            "param_tensors": len(params),
            "numel": int(sum(p.numel() for p in params)),
        })

    return param_groups, report


def _save_report(trainer: Any, report: Dict[str, Any]) -> None:
    try:
        save_dir = Path(getattr(trainer, "save_dir", ""))
        if not str(save_dir):
            return
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / "detector_coupled_optimizer_report.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[DC] Optimizer report saved to: {path}")
    except Exception as exc:  # pragma: no cover
        print(f"[DC] Warning: failed to save optimizer report: {exc}")


def _save_lr_snapshot(trainer: Any, tag: str = "") -> None:
    """
    Save current optimizer param-group learning rates to a jsonl file.
    This is used to verify whether Ultralytics scheduler/warmup changes
    DC layer-wise LR ratios during training.
    """
    try:
        if not hasattr(trainer, "optimizer") or trainer.optimizer is None:
            return

        save_dir = Path(getattr(trainer, "save_dir", ""))
        if not str(save_dir):
            return

        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / "detector_coupled_lr_trace.jsonl"

        epoch = getattr(trainer, "epoch", None)

        rows = []
        for idx, group in enumerate(trainer.optimizer.param_groups):
            rows.append({
                "group_index": idx,
                "name": group.get("name", f"pg{idx}"),
                "lr": float(group.get("lr", -1.0)),
                "initial_lr": float(group.get("initial_lr", -1.0)),
                "dc_lr_ratio": float(group.get("dc_lr_ratio", -1.0)),
                "num_params": len(group.get("params", [])),
            })

        record = {
            "tag": tag,
            "epoch": int(epoch) if epoch is not None else None,
            "groups": rows,
        }

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Also print occasionally so the console shows whether groups are preserved.
        if tag in {"build_optimizer", "epoch_start", "epoch_end"}:
            msg = " | ".join(
                [
                    f"{r['name']}={r['lr']:.8f}"
                    for r in rows
                ]
            )
            print(f"[DC-LR] tag={tag} epoch={record['epoch']} | {msg}")

    except Exception as exc:
        print(f"[DC] Warning: failed to save LR snapshot: {exc}")


def build_detector_coupled_trainer(
    dc_cfg: Dict[str, Any],
    project_root: str | Path | None = None,
    base_trainer_det: type | None = None,
):
    """Return a Trainer class with detector-coupled optimizer.

    This mirrors your existing Stage2 design where stage2 modules provide
    build_*_trainer(...) and train_stage2.py passes the resulting class into
    train_kwargs['trainer'].

    If base_trainer_det is provided, DC is composed with that trainer; otherwise
    it inherits from Ultralytics DetectionTrainer.
    """
    if not dc_cfg or not dc_cfg.get("enabled", False):
        return base_trainer_det if base_trainer_det is not None else DetectionTrainer

    if base_trainer_det is None:
        if DetectionTrainer is None:
            raise ImportError("[DC] Cannot import ultralytics DetectionTrainer.")
        BaseTrainer = DetectionTrainer
    else:
        BaseTrainer = base_trainer_det

    class DetectorCoupledTrainer(BaseTrainer):
        def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.937, decay=5e-4, iterations=1e5):
            effective_lr = float(lr)
            effective_decay = float(decay)
            effective_momentum = float(momentum)

            if not dc_cfg.get("base_lr_from_train", True):
                effective_lr = float(dc_cfg.get("base_lr", effective_lr))
            if not dc_cfg.get("weight_decay_from_train", True):
                effective_decay = float(dc_cfg.get("weight_decay", effective_decay))
            if not dc_cfg.get("momentum_from_train", True):
                effective_momentum = float(dc_cfg.get("momentum", effective_momentum))

            optimizer_name = str(dc_cfg.get("optimizer", name if name != "auto" else "AdamW"))

            print("\n" + "=" * 80)
            print("[DC] Detector-coupled fine-tuning optimizer enabled")
            print(f"[DC] optimizer     : {optimizer_name}")
            print(f"[DC] base lr       : {effective_lr}")
            print(f"[DC] weight_decay  : {effective_decay}")
            print(f"[DC] momentum/beta1: {effective_momentum}")
            print(f"[DC] train_shallow : {dc_cfg.get('train_shallow', False)}")

            param_groups, report = _build_param_groups(
                model=model,
                dc_cfg=dc_cfg,
                lr=effective_lr,
                decay=effective_decay,
            )

            for row in report["optimizer_groups"]:
                print(
                    f"[DC] group={row['name']:<8} "
                    f"lr={row['lr']:.8f} "
                    f"ratio={row['lr_ratio']:<5} "
                    f"tensors={row['param_tensors']:<5} "
                    f"numel={row['numel']}"
                )
            print("=" * 80 + "\n")

            if dc_cfg.get("save_param_group_report", True):
                _save_report(self, report)

            optimizer = _make_optimizer(
                param_groups=param_groups,
                optimizer_name=optimizer_name,
                lr=effective_lr,
                momentum=effective_momentum,
                weight_decay=effective_decay,
            )

            self.optimizer = optimizer
            _save_lr_snapshot(self, tag="build_optimizer")

            return optimizer

    DetectorCoupledTrainer.__name__ = "DetectorCoupledTrainer"
    return DetectorCoupledTrainer
