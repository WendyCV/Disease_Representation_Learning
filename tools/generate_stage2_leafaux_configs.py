from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CFG = PROJECT_ROOT / "configs" / "det_config.yaml"
DEFAULT_OUT_DIR = PROJECT_ROOT / "configs" / "stage2_leafaux_generated"


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def disable_other_routes(cfg: dict):
    for key in [
        "spatial_alignment",
        "foreground_prior_distillation",
        "leaf_prior_objectness_transfer",
    ]:
        cfg.setdefault(key, {})
        cfg[key]["enabled"] = False


def build_cfg(base: dict, name: str, *, stage1_enabled: bool, leaf_cfg: dict | None):
    cfg = copy.deepcopy(base)
    cfg.setdefault("train", {})
    cfg["train"]["name"] = name

    cfg.setdefault("stage1_init", {})
    cfg["stage1_init"]["enabled"] = bool(stage1_enabled)

    disable_other_routes(cfg)

    cfg.setdefault("leaf_prior_auxiliary", {})
    if leaf_cfg is None:
        cfg["leaf_prior_auxiliary"]["enabled"] = False
    else:
        cfg["leaf_prior_auxiliary"] = copy.deepcopy(leaf_cfg)
        cfg["leaf_prior_auxiliary"]["enabled"] = True
    return cfg


def resolve_levels(levels: str):
    if levels == "l1":
        return [0], [4]
    if levels == "l2":
        return [1], [6]
    if levels == "l3":
        return [2], [8]
    if levels == "l1l2":
        return [0, 1], [4, 6]
    if levels == "l2l3":
        return [1, 2], [6, 8]
    if levels == "l1l2l3":
        return [0, 1, 2], [4, 6, 8]
    raise ValueError(f"Unknown levels: {levels}")


def make_leaf_cfg(
    teacher_ckpt: str,
    teacher_ssl_cfg: str,
    lpa_base: dict,
    *,
    levels: str,
    alpha: float,
    weights: list[float],
    lambda_bg: float | None = None,
    enable_gate: bool | None = None,
):
    t_idx, s_idx = resolve_levels(levels)
    cfg = {
        "teacher_ckpt_path": teacher_ckpt,
        "teacher_ssl_config": teacher_ssl_cfg,
        "teacher_branch": "pos_feats",
        "teacher_feature_indices": t_idx,
        "student_layer_indices": s_idx,
        "layer_weights": [float(x) for x in weights],
        "alpha": float(alpha),
        "log_interval": int(lpa_base.get("log_interval", 20)),
        "gate_init_beta": float(lpa_base.get("gate_init_beta", 0.2)),
        "gamma": float(lpa_base.get("gamma", 1.0)),
        "bg_quantile": float(lpa_base.get("bg_quantile", 0.3)),
        "lambda_bg": float(lpa_base.get("lambda_bg", 0.25) if lambda_bg is None else lambda_bg),
    }
    if enable_gate is not None:
        cfg["enable_gate"] = bool(enable_gate)
    elif "enable_gate" in lpa_base:
        cfg["enable_gate"] = bool(lpa_base.get("enable_gate", True))
    return cfg


def add_leaf_exp(exps: dict, base: dict, teacher_ckpt: str, teacher_ssl_cfg: str, lpa_base: dict, name: str, *, levels: str, alpha: float, weights: list[float], lambda_bg: float | None = None, enable_gate: bool | None = None):
    exps[name] = build_cfg(
        base,
        name,
        stage1_enabled=True,
        leaf_cfg=make_leaf_cfg(
            teacher_ckpt,
            teacher_ssl_cfg,
            lpa_base,
            levels=levels,
            alpha=alpha,
            weights=weights,
            lambda_bg=lambda_bg,
            enable_gate=enable_gate,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate Stage2 LeafAux YAML configs")
    parser.add_argument("--base_config", type=str, default=str(DEFAULT_BASE_CFG))
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--teacher_ssl_config", type=str, default="./configs/stage1_ablation_generated/use_pos_mask.yaml")
    args = parser.parse_args()

    base_path = Path(args.base_config).resolve()
    out_dir = Path(args.out_dir).resolve()
    base = load_yaml(base_path)
    teacher_ckpt = base.get("stage1_init", {}).get("ckpt_path", "./runs/glcp_stage1_yolo_det/use_pos_mask/best.pth")
    lpa_base = base.get("leaf_prior_auxiliary", {})

    exps: dict[str, dict] = {
        "baseline": build_cfg(base, "baseline", stage1_enabled=False, leaf_cfg=None),
        "full_no_freeze": build_cfg(base, "full_no_freeze", stage1_enabled=True, leaf_cfg=None),
    }

    # Layer-combination references.
    add_leaf_exp(exps, base, teacher_ckpt, args.teacher_ssl_config, lpa_base, "leafaux_l1_a050_w10", levels="l1", alpha=0.5, weights=[1.0])
    add_leaf_exp(exps, base, teacher_ckpt, args.teacher_ssl_config, lpa_base, "leafaux_l1l2_a100_w05_10", levels="l1l2", alpha=1.0, weights=[0.5, 1.0])
    add_leaf_exp(exps, base, teacher_ckpt, args.teacher_ssl_config, lpa_base, "leafaux_l2l3_a100_w11", levels="l2l3", alpha=1.0, weights=[1.0, 1.0])
    add_leaf_exp(exps, base, teacher_ckpt, args.teacher_ssl_config, lpa_base, "leafaux_l2l3_a150_w11", levels="l2l3", alpha=1.5, weights=[1.0, 1.0])

    # Best L1/L2/L3 route and layer-weight variants.
    add_leaf_exp(exps, base, teacher_ckpt, args.teacher_ssl_config, lpa_base, "leafaux_l1l2l3_a100_w04_08_10", levels="l1l2l3", alpha=1.0, weights=[0.4, 0.8, 1.0])
    add_leaf_exp(exps, base, teacher_ckpt, args.teacher_ssl_config, lpa_base, "leafaux_l1l2l3_a100_w04_10_08", levels="l1l2l3", alpha=1.0, weights=[0.4, 1.0, 0.8])
    add_leaf_exp(exps, base, teacher_ckpt, args.teacher_ssl_config, lpa_base, "leafaux_l1l2l3_a100_w06_10_08", levels="l1l2l3", alpha=1.0, weights=[0.6, 1.0, 0.8])

    # Alpha sensitivity around best weights.
    add_leaf_exp(exps, base, teacher_ckpt, args.teacher_ssl_config, lpa_base, "leafaux_l1l2l3_a075_w04_08_10", levels="l1l2l3", alpha=0.75, weights=[0.4, 0.8, 1.0])
    add_leaf_exp(exps, base, teacher_ckpt, args.teacher_ssl_config, lpa_base, "leafaux_l1l2l3_a125_w04_08_10", levels="l1l2l3", alpha=1.25, weights=[0.4, 0.8, 1.0])

    # Mechanism ablations.
    add_leaf_exp(exps, base, teacher_ckpt, args.teacher_ssl_config, lpa_base, "leafaux_l1l2l3_a100_w04_08_10_nobg", levels="l1l2l3", alpha=1.0, weights=[0.4, 0.8, 1.0], lambda_bg=0.0)
    add_leaf_exp(exps, base, teacher_ckpt, args.teacher_ssl_config, lpa_base, "leafaux_l1l2l3_a100_w04_08_10_lossonly", levels="l1l2l3", alpha=1.0, weights=[0.4, 0.8, 1.0], enable_gate=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "base_config": str(base_path),
        "output_dir": str(out_dir),
        "purpose": "LeafAux configs: layer combination, L1/L2/L3 weights, alpha sensitivity, bg/gate mechanism.",
        "experiments": [],
    }
    for name, cfg in exps.items():
        out_path = out_dir / f"{name}.yaml"
        save_yaml(cfg, out_path)
        manifest["experiments"].append({"name": name, "file": str(out_path)})
        print(f"[OK] {out_path}")

    manifest_path = out_dir / "manifest_stage2_leafaux.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("=" * 80)
    print("Generated Stage2 LeafAux configs.")
    print("Manifest:", manifest_path)
    print("=" * 80)


if __name__ == "__main__":
    main()
