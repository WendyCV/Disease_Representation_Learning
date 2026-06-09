from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import yaml


class ConfigError(ValueError):
    """Raised when a stage-1 configuration is missing required keys or contains inconsistent switches."""


REQUIRED_STAGE1_PATHS = [
    ("data", "train_dir"),
    ("data", "image_size"),
    ("train", "batch_size"),
    ("train", "epochs"),
    ("train", "lr"),
    ("model", "yolo_model"),
    ("model", "layer_indices"),
    ("model", "proj_dim"),
    ("model", "local_dim"),
    ("loss", "base", "global"),
    ("loss", "base", "local"),
    ("loss", "base", "position"),
]


DEFAULT_STAGE1_CONFIG: Dict[str, Any] = {
    "experiment": {
        "name": None,
        "family": "stage1_ssl",
        "notes": "",
    },
    "model": {
        "sppf_indice": 9,
        "nc": None,
        "queue_size": 4096,
        "momentum": 0.999,
        "temperature": 0.2,
        "pos_pe_channels": 64,
        "pos_init_scales": [0.1, 0.5, 1.0],
        "pos_enable_fg_guidance": True,
        "pos_fg_gate_init": 1.0,
        "separate_projector": False,
        "snapshot_teacher": {
            "enabled": True,
            "freeze_after_epoch": 5,
        },
    },
    "loss": {
        "aux_embedding": {
            "enabled": True,
            "teacher_source": "snapshot",
            "weight": 0.10,
            "local_weight": 0.20,
            "detach_teacher": True,
            "scale_weights": [0.10, 0.45, 0.45],
        },
        "raw_spatial": {
            "enabled": True,
            "mask_weight": 0.25,
            "consistency_weight": 0.15,
            "mask_scale_weights": [0.10, 0.45, 0.45],
            "consistency_scale_weights": [0.10, 0.45, 0.45],
            "foreground_background_margin": 0.15,
        },
    },
    "ablation": {
        "use_pos": True,
        "use_mask": True,
        "use_aux_embedding": None,
        "use_raw_spatial": None,
        "freeze_after_epoch_override": None,
    },
}


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError("Top-level configuration must be a mapping.")
    return config


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def with_stage1_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    return _deep_merge(DEFAULT_STAGE1_CONFIG, config)


def require_config_paths(config: Dict[str, Any], paths: Iterable[Tuple[str, ...]]) -> None:
    for path in paths:
        cursor: Any = config
        for part in path:
            if not isinstance(cursor, dict) or part not in cursor:
                joined = ".".join(path)
                raise ConfigError(f"Missing required config key: {joined}")
            cursor = cursor[part]


def _resolve_bool_override(default_value: bool, override_value):
    if override_value is None:
        return bool(default_value)
    return bool(override_value)


def build_stage1_experiment_name(config: Dict[str, Any]) -> str:
    explicit_name = config.get("experiment", {}).get("name")
    if explicit_name:
        return str(explicit_name)

    use_pos = config["ablation"]["use_pos"]
    use_mask = config["ablation"]["use_mask"]
    use_aux = config["loss"]["aux_embedding"]["enabled"]
    use_raw = config["loss"]["raw_spatial"]["enabled"]
    freeze_after_epoch = config["model"]["snapshot_teacher"]["freeze_after_epoch"]

    if use_pos and use_mask:
        base = "use_pos_mask"
    elif (not use_pos) and use_mask:
        base = "wo_pos"
    elif use_pos and (not use_mask):
        base = "wo_mask"
    else:
        base = "wo_pos_wo_mask"

    suffix = []
    if not use_aux:
        suffix.append("noaux")
    if not use_raw:
        suffix.append("noraw")
    if config["model"]["snapshot_teacher"]["enabled"]:
        suffix.append(f"f{int(freeze_after_epoch):02d}")

    return base if not suffix else base + "__" + "_".join(suffix)


def validate_stage1_config(config: Dict[str, Any]) -> Dict[str, Any]:
    validated = with_stage1_defaults(config)
    require_config_paths(validated, REQUIRED_STAGE1_PATHS)

    layer_indices = validated["model"]["layer_indices"]
    if not isinstance(layer_indices, (list, tuple)) or len(layer_indices) == 0:
        raise ConfigError("model.layer_indices must be a non-empty list.")

    ablation_cfg = validated["ablation"]
    use_pos = bool(ablation_cfg.get("use_pos", True))
    use_mask = bool(ablation_cfg.get("use_mask", True))
    use_aux_embedding = _resolve_bool_override(
        validated["loss"]["aux_embedding"]["enabled"],
        ablation_cfg.get("use_aux_embedding"),
    )
    use_raw_spatial = _resolve_bool_override(
        validated["loss"]["raw_spatial"]["enabled"],
        ablation_cfg.get("use_raw_spatial"),
    )

    default_freeze_after_epoch = int(validated["model"]["snapshot_teacher"]["freeze_after_epoch"])
    freeze_after_epoch_override = ablation_cfg.get("freeze_after_epoch_override")
    freeze_after_epoch = default_freeze_after_epoch if freeze_after_epoch_override is None else int(freeze_after_epoch_override)

    snapshot_teacher_enabled = bool(validated["model"]["snapshot_teacher"]["enabled"])
    aux_teacher_source = validated["loss"]["aux_embedding"].get("teacher_source", "snapshot")

    total_epochs = int(validated["train"]["epochs"])
    if freeze_after_epoch < 0 or freeze_after_epoch > total_epochs:
        raise ConfigError(
            f"model.snapshot_teacher.freeze_after_epoch must be in [0, {total_epochs}], but got {freeze_after_epoch}."
        )

    if use_raw_spatial and not use_mask:
        raise ConfigError("raw_spatial requires masks. Set ablation.use_raw_spatial=false when ablation.use_mask=false.")

    if use_aux_embedding and aux_teacher_source not in {"snapshot", "momentum", "online"}:
        raise ConfigError("loss.aux_embedding.teacher_source must be one of: snapshot, momentum, online.")

    if use_aux_embedding and aux_teacher_source == "snapshot" and not snapshot_teacher_enabled:
        raise ConfigError("aux_embedding uses snapshot teacher, but model.snapshot_teacher.enabled=false.")

    validated["ablation"]["use_pos"] = use_pos
    validated["ablation"]["use_mask"] = use_mask
    validated["ablation"]["use_aux_embedding"] = use_aux_embedding
    validated["ablation"]["use_raw_spatial"] = use_raw_spatial
    validated["ablation"]["freeze_after_epoch_override"] = freeze_after_epoch_override

    validated["loss"]["aux_embedding"]["enabled"] = use_aux_embedding
    validated["loss"]["raw_spatial"]["enabled"] = use_raw_spatial
    validated["model"]["snapshot_teacher"]["freeze_after_epoch"] = freeze_after_epoch

    validated.setdefault("runtime", {})
    validated["runtime"].update({
        "use_pos": use_pos,
        "use_mask": use_mask,
        "use_aux_embedding": use_aux_embedding,
        "use_raw_spatial": use_raw_spatial,
        "snapshot_teacher_enabled": snapshot_teacher_enabled,
        "snapshot_freeze_after_epoch": freeze_after_epoch,
        "needs_snapshot_teacher": snapshot_teacher_enabled and (
            use_aux_embedding and aux_teacher_source == "snapshot"
        ),
    })

    validated["experiment"]["name"] = build_stage1_experiment_name(validated)
    return validated


def summarize_stage1_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "experiment_name": config["experiment"]["name"],
        "train_dir": config["data"]["train_dir"],
        "mask_mode": config["data"].get("mask_mode", "sam2"),
        "image_size": config["data"]["image_size"],
        "epochs": config["train"]["epochs"],
        "batch_size": config["train"]["batch_size"],
        "learning_rate": config["train"]["lr"],
        "use_pos": config["runtime"]["use_pos"],
        "use_mask": config["runtime"]["use_mask"],
        "use_aux_embedding": config["runtime"]["use_aux_embedding"],
        "use_raw_spatial": config["runtime"]["use_raw_spatial"],
        "snapshot_teacher_enabled": config["runtime"]["snapshot_teacher_enabled"],
        "snapshot_freeze_after_epoch": config["runtime"]["snapshot_freeze_after_epoch"],
    }
