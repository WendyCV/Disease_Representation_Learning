from __future__ import annotations

import copy
from pathlib import Path
import yaml
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CFG = PROJECT_ROOT / "configs" / "det_config.yaml"
OUT_DIR = PROJECT_ROOT / "configs" / "stage2_lpot_generated"
TEACHER_SSL_CFG = "./configs/stage1_ablation_generated/use_pos_mask.yaml"


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def build_cfg(base: dict, name: str, *, stage1_enabled: bool, lpot_cfg: dict | None):
    cfg = copy.deepcopy(base)
    cfg.setdefault("train", {})
    cfg["train"]["name"] = name

    cfg.setdefault("stage1_init", {})
    cfg["stage1_init"]["enabled"] = bool(stage1_enabled)

    # keep generated YAML as single-route LPOT experiments
    for key in ["spatial_alignment", "foreground_prior_distillation", "leaf_prior_auxiliary"]:
        cfg.setdefault(key, {})
        cfg[key]["enabled"] = False

    cfg.setdefault("leaf_prior_objectness_transfer", {})
    if lpot_cfg is None:
        cfg["leaf_prior_objectness_transfer"]["enabled"] = False
    else:
        cfg["leaf_prior_objectness_transfer"] = copy.deepcopy(lpot_cfg)
        cfg["leaf_prior_objectness_transfer"]["enabled"] = True
    return cfg


def make_lpot_l1l2l3(
    teacher_ckpt: str,
    teacher_cfg: str,
    base_cfg: dict,
    *,
    name_impl: str,
    alpha: float,
    prior_weights: list[float],
    proxy_weights: list[float],
    lambda_prior: float,
    lambda_proxy: float,
    lambda_bridge: float,
    lambda_rank: float,
    lambda_align: float,
    enable_feature_gate: bool,
    enable_proxy_gate: bool,
    proxy_support_gate_gain: float,
    lambda_bg: float,
    lambda_score_rank: float = 0.0,
    lambda_score_bg: float = 0.0,
    score_rank_layer_weights: list[float] | None = None,
):
    if len(prior_weights) != 3:
        raise ValueError("LPOT L1/L2/L3 prior branch requires three prior weights.")
    if len(proxy_weights) != 3:
        raise ValueError("LPOT proxy branch requires P3/P4/P5 three weights.")
    if score_rank_layer_weights is None:
        score_rank_layer_weights = []

    return {
        "implementation": name_impl,
        "route_variant": name_impl,
        "teacher_ckpt_path": teacher_ckpt,
        "teacher_ssl_config": teacher_cfg,
        "teacher_branch": "pos_feats",
        "teacher_feature_indices": [0, 1, 2],
        "student_layer_indices": [4, 6, 8],
        "layer_weights": [float(x) for x in prior_weights],
        "proxy_layer_indices": [15, 18, 21],
        "proxy_layer_weights": [float(x) for x in proxy_weights],
        "teacher_fuse_weights": [float(x) for x in prior_weights],
        "alpha": float(alpha),
        "lambda_prior": float(lambda_prior),
        "lambda_proxy": float(lambda_proxy),
        "lambda_bridge": float(lambda_bridge),
        "lambda_rank": float(lambda_rank),
        "lambda_align": float(lambda_align),
        "support_margin": float(base_cfg.get("support_margin", 0.10)),
        "objectness_student_blend": float(base_cfg.get("objectness_student_blend", 0.75)),
        "objectness_smooth_kernel": int(base_cfg.get("objectness_smooth_kernel", 7)),
        "objectness_smooth_iters": int(base_cfg.get("objectness_smooth_iters", 2)),
        "enable_feature_gate": bool(enable_feature_gate),
        "enable_proxy_gate": bool(enable_proxy_gate),
        "gate_init_beta": float(base_cfg.get("gate_init_beta", 0.2)),
        "proxy_gate_init_beta": float(base_cfg.get("proxy_gate_init_beta", 0.05)),
        "support_gate_gain": 0.0,
        "proxy_support_gate_gain": float(proxy_support_gate_gain),
        "fg_quantile": float(base_cfg.get("fg_quantile", 0.7)),
        "bg_quantile": float(base_cfg.get("bg_quantile", 0.3)),
        "lambda_bg": float(lambda_bg),
        "gamma": float(base_cfg.get("gamma", 1.0)),
        "lambda_teacher_rank": 0.0,
        "lambda_teacher_bg": 0.0,
        "teacher_anchor_margin": float(base_cfg.get("teacher_anchor_margin", 0.10)),
        "teacher_fg_quantile": float(base_cfg.get("teacher_fg_quantile", 0.70)),
        "teacher_bg_quantile": float(base_cfg.get("teacher_bg_quantile", 0.30)),
        "teacher_anchor_layer_weights": [],
        "lambda_score_rank": float(lambda_score_rank),
        "lambda_score_bg": float(lambda_score_bg),
        "score_rank_margin": float(base_cfg.get("score_rank_margin", 0.10)),
        "score_fg_quantile": float(base_cfg.get("score_fg_quantile", 0.70)),
        "score_bg_quantile": float(base_cfg.get("score_bg_quantile", 0.30)),
        "score_rank_layer_weights": [float(x) for x in score_rank_layer_weights],
        "score_gt_expand_ratio": float(base_cfg.get("score_gt_expand_ratio", 1.5)),
        "score_use_gt_pos": bool(base_cfg.get("score_use_gt_pos", True)),
        "score_use_teacher_fg_pos": bool(base_cfg.get("score_use_teacher_fg_pos", False)),
        "log_interval": int(base_cfg.get("log_interval", 20)),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate stage2 ablation YAML configs from base det_config.yaml")
    parser.add_argument("--base_config", type=str, default="./configs/det_config.yaml", help="Base det config yaml")
    parser.add_argument("--out_dir", type=str, default="./configs/stage2_lpot_generated", help="Output dir")
    parser.add_argument("--stage1_ckpt", type=str, default="./runs/glcp_stage1_yolo_det/use_pos_mask/best.pth", help="Stage1 checkpoint path")
    parser.add_argument("--stage1_cfg", type=str, default="./configs/stage1_ablation_generated/use_pos_mask.yaml", help="Stage1 config path")
    args = parser.parse_args()

    base = load_yaml(args.base_config if args.base_config else BASE_CFG)
    teacher_ckpt = args.stage1_ckpt if args.stage1_ckpt else base.get("stage1_init", {}).get("ckpt_path", "")
    teacher_cfg = args.stage1_cfg if args.stage1_cfg else TEACHER_SSL_CFG
    lpot_base = base.get("leaf_prior_objectness_transfer", {})

    exps = {
        "baseline": build_cfg(base, "baseline", stage1_enabled=False, lpot_cfg=None),
        "full_no_freeze": build_cfg(base, "full_no_freeze", stage1_enabled=True, lpot_cfg=None),

        # conservative L1/L2/L3 LPOT route; not recommended as main paper route yet
        "lpot_l1l2l3_psp_light_a040_w04_10_08_lossonly": build_cfg(
            base,
            "lpot_l1l2l3_psp_light_a040_w04_10_08_lossonly",
            stage1_enabled=True,
            lpot_cfg=make_lpot_l1l2l3(
                teacher_ckpt,
                teacher_cfg,
                lpot_base,
                name_impl="proposal_support_prior_v1",
                alpha=0.4,
                prior_weights=[0.4, 1.0, 0.8],
                proxy_weights=[0.3, 1.0, 1.0],
                lambda_prior=0.3,
                lambda_proxy=0.20,
                lambda_bridge=0.0,
                lambda_rank=0.03,
                lambda_align=0.0,
                enable_feature_gate=False,
                enable_proxy_gate=False,
                proxy_support_gate_gain=0.0,
                lambda_bg=0.25,
            ),
        ),
        "lpot_l1l2l3_psp_proxyonly_gate01_nobridge": build_cfg(
            base,
            "lpot_l1l2l3_psp_proxyonly_gate01_nobridge",
            stage1_enabled=True,
            lpot_cfg=make_lpot_l1l2l3(
                teacher_ckpt,
                teacher_cfg,
                lpot_base,
                name_impl="proposal_support_prior_v1",
                alpha=0.4,
                prior_weights=[0.4, 1.0, 0.8],
                proxy_weights=[0.3, 1.0, 1.0],
                lambda_prior=0.0,
                lambda_proxy=0.20,
                lambda_bridge=0.0,
                lambda_rank=0.03,
                lambda_align=0.0,
                enable_feature_gate=False,
                enable_proxy_gate=True,
                proxy_support_gate_gain=0.1,
                lambda_bg=0.25,
            ),
        ),
        "lpot41_l1l2l3_proxyonly_gate01_scorerank002_bg001": build_cfg(
            base,
            "lpot41_l1l2l3_proxyonly_gate01_scorerank002_bg001",
            stage1_enabled=True,
            lpot_cfg=make_lpot_l1l2l3(
                teacher_ckpt,
                teacher_cfg,
                lpot_base,
                name_impl="score_level_proposal_support_v1",
                alpha=0.4,
                prior_weights=[0.4, 1.0, 0.8],
                proxy_weights=[0.3, 1.0, 1.0],
                lambda_prior=0.0,
                lambda_proxy=0.20,
                lambda_bridge=0.0,
                lambda_rank=0.03,
                lambda_align=0.0,
                enable_feature_gate=False,
                enable_proxy_gate=True,
                proxy_support_gate_gain=0.1,
                lambda_bg=0.25,
                lambda_score_rank=0.02,
                lambda_score_bg=0.01,
                score_rank_layer_weights=[1.2, 1.0, 0.5],
            ),
        ),
    }

    cfg_gen_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    cfg_gen_dir.mkdir(parents=True, exist_ok=True)
    for name, cfg in exps.items():
        save_yaml(cfg, cfg_gen_dir / f"{name}.yaml")
        print(f"[OK] {cfg_gen_dir / f'{name}.yaml'}")


if __name__ == "__main__":
    main()
