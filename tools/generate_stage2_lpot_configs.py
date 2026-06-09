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

    # keep every generated YAML as a clean single-route experiment
    for key in [
        "spatial_alignment",
        "foreground_prior_distillation",
        "leaf_prior_auxiliary",
    ]:
        cfg.setdefault(key, {})
        cfg[key]["enabled"] = False

    cfg.setdefault("leaf_prior_objectness_transfer", {})
    if lpot_cfg is None:
        cfg["leaf_prior_objectness_transfer"]["enabled"] = False
    else:
        cfg["leaf_prior_objectness_transfer"] = copy.deepcopy(lpot_cfg)
        cfg["leaf_prior_objectness_transfer"]["enabled"] = True

    return cfg


def make_lpot_cfg(
    teacher_ckpt: str,
    teacher_ssl_cfg: str,
    base_cfg: dict,
    *,
    name_impl: str,
    alpha: float,
    prior_weights,
    proxy_weights,
    lambda_prior: float,
    lambda_proxy: float,
    lambda_bridge: float,
    lambda_rank: float | None = None,
    lambda_align: float | None = None,
    support_margin: float | None = None,
    objectness_student_blend: float | None = None,
    objectness_smooth_kernel: int | None = None,
    objectness_smooth_iters: int | None = None,
    enable_feature_gate: bool | None = None,
    enable_proxy_gate: bool | None = None,
    gate_init_beta: float | None = None,
    proxy_gate_init_beta: float | None = None,
    support_gate_gain: float | None = None,
    proxy_support_gate_gain: float | None = None,
    fg_quantile: float | None = None,
    bg_quantile: float | None = None,
    lambda_bg: float | None = None,
    gamma: float | None = None,
    lambda_teacher_rank: float | None = None,
    lambda_teacher_bg: float | None = None,
    teacher_anchor_margin: float | None = None,
    teacher_fg_quantile: float | None = None,
    teacher_bg_quantile: float | None = None,
    teacher_anchor_layer_weights=None,
    lambda_score_rank: float | None = None,
    lambda_score_bg: float | None = None,
    score_rank_margin: float | None = None,
    score_fg_quantile: float | None = None,
    score_bg_quantile: float | None = None,
    score_rank_layer_weights=None,
    score_gt_expand_ratio: float | None = None,
    score_use_gt_pos: bool | None = None,
    score_use_teacher_fg_pos: bool | None = None,
):
    if objectness_student_blend is None:
        objectness_student_blend = base_cfg.get("objectness_student_blend", 0.75)
    if objectness_smooth_kernel is None:
        objectness_smooth_kernel = base_cfg.get("objectness_smooth_kernel", 7)
    if objectness_smooth_iters is None:
        objectness_smooth_iters = base_cfg.get("objectness_smooth_iters", 2)
    if gate_init_beta is None:
        gate_init_beta = base_cfg.get("gate_init_beta", 0.2)
    if proxy_gate_init_beta is None:
        proxy_gate_init_beta = base_cfg.get("proxy_gate_init_beta", 0.05)
    if support_gate_gain is None:
        support_gate_gain = base_cfg.get("support_gate_gain", 1.0)
    if proxy_support_gate_gain is None:
        proxy_support_gate_gain = base_cfg.get("proxy_support_gate_gain", support_gate_gain)
    if fg_quantile is None:
        fg_quantile = base_cfg.get("fg_quantile", 0.7)
    if bg_quantile is None:
        bg_quantile = base_cfg.get("bg_quantile", 0.3)
    if lambda_bg is None:
        lambda_bg = base_cfg.get("lambda_bg", 0.25)
    if gamma is None:
        gamma = base_cfg.get("gamma", 1.0)
    if support_margin is None:
        support_margin = base_cfg.get("support_margin", 0.10)
    if lambda_rank is None:
        lambda_rank = base_cfg.get("lambda_rank", 0.20)
    if lambda_align is None:
        lambda_align = base_cfg.get("lambda_align", 1.0)
    if lambda_teacher_rank is None:
        lambda_teacher_rank = base_cfg.get("lambda_teacher_rank", 0.0)
    if lambda_teacher_bg is None:
        lambda_teacher_bg = base_cfg.get("lambda_teacher_bg", 0.0)
    if teacher_anchor_margin is None:
        teacher_anchor_margin = base_cfg.get("teacher_anchor_margin", support_margin)
    if teacher_fg_quantile is None:
        teacher_fg_quantile = base_cfg.get("teacher_fg_quantile", fg_quantile)
    if teacher_bg_quantile is None:
        teacher_bg_quantile = base_cfg.get("teacher_bg_quantile", bg_quantile)
    if teacher_anchor_layer_weights is None:
        teacher_anchor_layer_weights = base_cfg.get("teacher_anchor_layer_weights", [])
    if lambda_score_rank is None:
        lambda_score_rank = base_cfg.get("lambda_score_rank", 0.0)
    if lambda_score_bg is None:
        lambda_score_bg = base_cfg.get("lambda_score_bg", 0.0)
    if score_rank_margin is None:
        score_rank_margin = base_cfg.get("score_rank_margin", 0.10)
    if score_fg_quantile is None:
        score_fg_quantile = base_cfg.get("score_fg_quantile", fg_quantile)
    if score_bg_quantile is None:
        score_bg_quantile = base_cfg.get("score_bg_quantile", bg_quantile)
    if score_rank_layer_weights is None:
        score_rank_layer_weights = base_cfg.get("score_rank_layer_weights", [])
    if score_gt_expand_ratio is None:
        score_gt_expand_ratio = base_cfg.get("score_gt_expand_ratio", 1.5)
    if score_use_gt_pos is None:
        score_use_gt_pos = base_cfg.get("score_use_gt_pos", True)
    if score_use_teacher_fg_pos is None:
        score_use_teacher_fg_pos = base_cfg.get("score_use_teacher_fg_pos", False)
    if enable_feature_gate is None:
        enable_feature_gate = bool(base_cfg.get("enable_feature_gate", True))
    if enable_proxy_gate is None:
        enable_proxy_gate = bool(base_cfg.get("enable_proxy_gate", True))

    return {
        "implementation": name_impl,
        "route_variant": name_impl,

        "teacher_ckpt_path": teacher_ckpt,
        "teacher_ssl_config": teacher_ssl_cfg,
        "teacher_branch": "pos_feats",
        "teacher_feature_indices": [0, 1, 2],

        "student_layer_indices": [4, 6, 8],
        "layer_weights": [float(x) for x in prior_weights],

        "proxy_layer_indices": [15, 18, 21],
        "proxy_layer_weights": [float(x) for x in proxy_weights],

        "teacher_fuse_weights": [1.0, 1.0, 1.0],
        "alpha": float(alpha),

        "lambda_prior": float(lambda_prior),
        "lambda_proxy": float(lambda_proxy),
        "lambda_bridge": float(lambda_bridge),
        "lambda_rank": float(lambda_rank),
        "lambda_align": float(lambda_align),
        "support_margin": float(support_margin),

        "objectness_student_blend": float(objectness_student_blend),
        "objectness_smooth_kernel": int(objectness_smooth_kernel),
        "objectness_smooth_iters": int(objectness_smooth_iters),

        "enable_feature_gate": bool(enable_feature_gate),
        "enable_proxy_gate": bool(enable_proxy_gate),
        "gate_init_beta": float(gate_init_beta),
        "proxy_gate_init_beta": float(proxy_gate_init_beta),
        "support_gate_gain": float(support_gate_gain),
        "proxy_support_gate_gain": float(proxy_support_gate_gain),

        "fg_quantile": float(fg_quantile),
        "bg_quantile": float(bg_quantile),
        "lambda_bg": float(lambda_bg),
        "gamma": float(gamma),

        # LPOT-v5 teacher-anchor fields. Defaults are zero, so old v4 configs
        # are unchanged unless explicitly set.
        "lambda_teacher_rank": float(lambda_teacher_rank),
        "lambda_teacher_bg": float(lambda_teacher_bg),
        "teacher_anchor_margin": float(teacher_anchor_margin),
        "teacher_fg_quantile": float(teacher_fg_quantile),
        "teacher_bg_quantile": float(teacher_bg_quantile),
        "teacher_anchor_layer_weights": [float(x) for x in (teacher_anchor_layer_weights or [])],

        # LPOT-v4.1 score-level proposal-support regularization.
        # Disabled by default; enabled only by v4.1 generated configs.
        "lambda_score_rank": float(lambda_score_rank),
        "lambda_score_bg": float(lambda_score_bg),
        "score_rank_margin": float(score_rank_margin),
        "score_fg_quantile": float(score_fg_quantile),
        "score_bg_quantile": float(score_bg_quantile),
        "score_rank_layer_weights": [float(x) for x in (score_rank_layer_weights or [])],
        "score_gt_expand_ratio": float(score_gt_expand_ratio),
        "score_use_gt_pos": bool(score_use_gt_pos),
        "score_use_teacher_fg_pos": bool(score_use_teacher_fg_pos),

        "log_interval": int(base_cfg.get("log_interval", 20)),
    }


def add_lpot_exp(
    exps: dict,
    base: dict,
    teacher_ckpt: str,
    teacher_cfg: str,
    lpot_base: dict,
    *,
    name: str,
    stage1_enabled: bool = True,
    **kwargs,
):
    exps[name] = build_cfg(
        base,
        name,
        stage1_enabled=stage1_enabled,
        lpot_cfg=make_lpot_cfg(
            teacher_ckpt,
            teacher_cfg,
            lpot_base,
            **kwargs,
        ),
    )


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

    exps: dict[str, dict] = {}

    # Reference baselines
    exps["baseline"] = build_cfg(base, "baseline", stage1_enabled=False, lpot_cfg=None)
    exps["full_no_freeze"] = build_cfg(base, "full_no_freeze", stage1_enabled=True, lpot_cfg=None)

    # Current best LPOT reference from full ablation:
    # low-strength legacy, L2 slightly stronger than L3.
    add_lpot_exp(
        exps,
        base,
        teacher_ckpt,
        teacher_cfg,
        lpot_base,
        name="lpotv2_l2l3_a060_pw12_08",
        name_impl="legacy_lpot_v2",
        alpha=0.6,
        prior_weights=[1.2, 0.8],
        proxy_weights=[1.0, 1.0, 1.0],
        lambda_prior=1.0,
        lambda_proxy=0.5,
        lambda_bridge=0.4,
        lambda_rank=0.0,
        lambda_align=1.0,
        objectness_student_blend=0.75,
        objectness_smooth_kernel=7,
        objectness_smooth_iters=2,
        enable_feature_gate=True,
        enable_proxy_gate=True,
        support_gate_gain=1.0,
        proxy_support_gate_gain=1.0,
    )

    # LPOT-v4 light PSP: loss-only, ranking/background-oriented.
    # This directly tests the corrected narrative:
    # Stage1 foreground prior -> soft proposal-support ranking -> no feature override.
    add_lpot_exp(
        exps,
        base,
        teacher_ckpt,
        teacher_cfg,
        lpot_base,
        name="lpotv4_psp_light_pw12_08_lossonly",
        name_impl="proposal_support_prior_v1",
        alpha=0.6,
        prior_weights=[1.2, 0.8],
        proxy_weights=[0.3, 1.0, 1.0],
        lambda_prior=0.5,
        lambda_proxy=0.25,
        lambda_bridge=0.2,
        lambda_rank=0.05,
        lambda_align=0.0,
        support_margin=0.10,
        objectness_student_blend=0.75,
        objectness_smooth_kernel=7,
        objectness_smooth_iters=2,
        enable_feature_gate=False,
        enable_proxy_gate=False,
        support_gate_gain=0.0,
        proxy_support_gate_gain=0.0,
        fg_quantile=0.7,
        bg_quantile=0.3,
        lambda_bg=0.25,
        gamma=1.0,
    )

    # LPOT-v4 light PSP + extremely weak proxy gate:
    # The prior still does not touch backbone features, but can softly bias
    # pre-head proxy features as a very weak support signal.
    add_lpot_exp(
        exps,
        base,
        teacher_ckpt,
        teacher_cfg,
        lpot_base,
        name="lpotv4_psp_light_pw12_08_proxygate02",
        name_impl="proposal_support_prior_v1",
        alpha=0.6,
        prior_weights=[1.2, 0.8],
        proxy_weights=[0.3, 1.0, 1.0],
        lambda_prior=0.5,
        lambda_proxy=0.25,
        lambda_bridge=0.2,
        lambda_rank=0.05,
        lambda_align=0.0,
        support_margin=0.10,
        objectness_student_blend=0.75,
        objectness_smooth_kernel=7,
        objectness_smooth_iters=2,
        enable_feature_gate=False,
        enable_proxy_gate=True,
        support_gate_gain=0.0,
        proxy_support_gate_gain=0.2,
        fg_quantile=0.7,
        bg_quantile=0.3,
        lambda_bg=0.25,
        gamma=1.0,
    )

    # ------------------------------------------------------------------
    # NEXT GROUP: v4-light stability refinement experiments
    # Added only; all previous experiments are kept unchanged.
    # These groups test whether weaker proxy support and/or stronger
    # background suppression can reduce over-candidate behavior and improve
    # final-epoch stability.
    # ------------------------------------------------------------------
    add_lpot_exp(
        exps,
        base,
        teacher_ckpt,
        teacher_cfg,
        lpot_base,
        name="lpotv4_psp_light_pw12_08_proxygate01",
        name_impl="proposal_support_prior_v1",
        alpha=0.6,
        prior_weights=[1.2, 0.8],
        proxy_weights=[0.3, 1.0, 1.0],
        lambda_prior=0.5,
        lambda_proxy=0.25,
        lambda_bridge=0.2,
        lambda_rank=0.05,
        lambda_align=0.0,
        support_margin=0.10,
        objectness_student_blend=0.75,
        objectness_smooth_kernel=7,
        objectness_smooth_iters=2,
        enable_feature_gate=False,
        enable_proxy_gate=True,
        support_gate_gain=0.0,
        proxy_support_gate_gain=0.1,
        fg_quantile=0.7,
        bg_quantile=0.3,
        lambda_bg=0.25,
        gamma=1.0,
    )

    add_lpot_exp(
        exps,
        base,
        teacher_ckpt,
        teacher_cfg,
        lpot_base,
        name="lpotv4_psp_light_pw12_08_proxygate015",
        name_impl="proposal_support_prior_v1",
        alpha=0.6,
        prior_weights=[1.2, 0.8],
        proxy_weights=[0.3, 1.0, 1.0],
        lambda_prior=0.5,
        lambda_proxy=0.25,
        lambda_bridge=0.2,
        lambda_rank=0.05,
        lambda_align=0.0,
        support_margin=0.10,
        objectness_student_blend=0.75,
        objectness_smooth_kernel=7,
        objectness_smooth_iters=2,
        enable_feature_gate=False,
        enable_proxy_gate=True,
        support_gate_gain=0.0,
        proxy_support_gate_gain=0.15,
        fg_quantile=0.7,
        bg_quantile=0.3,
        lambda_bg=0.25,
        gamma=1.0,
    )

    add_lpot_exp(
        exps,
        base,
        teacher_ckpt,
        teacher_cfg,
        lpot_base,
        name="lpotv4_psp_light_pw12_08_proxygate02_bg035",
        name_impl="proposal_support_prior_v1",
        alpha=0.6,
        prior_weights=[1.2, 0.8],
        proxy_weights=[0.3, 1.0, 1.0],
        lambda_prior=0.5,
        lambda_proxy=0.25,
        lambda_bridge=0.2,
        lambda_rank=0.05,
        lambda_align=0.0,
        support_margin=0.10,
        objectness_student_blend=0.75,
        objectness_smooth_kernel=7,
        objectness_smooth_iters=2,
        enable_feature_gate=False,
        enable_proxy_gate=True,
        support_gate_gain=0.0,
        proxy_support_gate_gain=0.2,
        fg_quantile=0.7,
        bg_quantile=0.3,
        lambda_bg=0.35,
        gamma=1.0,
    )

    add_lpot_exp(
        exps,
        base,
        teacher_ckpt,
        teacher_cfg,
        lpot_base,
        name="lpotv4_psp_light_pw12_08_proxygate01_bg035",
        name_impl="proposal_support_prior_v1",
        alpha=0.6,
        prior_weights=[1.2, 0.8],
        proxy_weights=[0.3, 1.0, 1.0],
        lambda_prior=0.5,
        lambda_proxy=0.25,
        lambda_bridge=0.2,
        lambda_rank=0.05,
        lambda_align=0.0,
        support_margin=0.10,
        objectness_student_blend=0.75,
        objectness_smooth_kernel=7,
        objectness_smooth_iters=2,
        enable_feature_gate=False,
        enable_proxy_gate=True,
        support_gate_gain=0.0,
        proxy_support_gate_gain=0.1,
        fg_quantile=0.7,
        bg_quantile=0.3,
        lambda_bg=0.35,
        gamma=1.0,
    )

    add_lpot_exp(
        exps,
        base,
        teacher_ckpt,
        teacher_cfg,
        lpot_base,
        name="lpotv4_psp_light_pw12_08_lossonly_bg035",
        name_impl="proposal_support_prior_v1",
        alpha=0.6,
        prior_weights=[1.2, 0.8],
        proxy_weights=[0.3, 1.0, 1.0],
        lambda_prior=0.5,
        lambda_proxy=0.25,
        lambda_bridge=0.2,
        lambda_rank=0.05,
        lambda_align=0.0,
        support_margin=0.10,
        objectness_student_blend=0.75,
        objectness_smooth_kernel=7,
        objectness_smooth_iters=2,
        enable_feature_gate=False,
        enable_proxy_gate=False,
        support_gate_gain=0.0,
        proxy_support_gate_gain=0.0,
        fg_quantile=0.7,
        bg_quantile=0.3,
        lambda_bg=0.35,
        gamma=1.0,
    )

    # ------------------------------------------------------------------
    # LPOT-v4 mechanism-consistent ablations
    # Purpose:
    #   Verify whether the effective LPOT mechanism mainly comes from
    #   pre-head objectness-like candidate support, rather than backbone
    #   prior alignment or bridge consistency.
    # ------------------------------------------------------------------

    # A. Proxy-only LPOT with weak pre-head gate.
    # Question:
    #   Is pre-head candidate-support supervision the main effective part?
    add_lpot_exp(
        exps,
        base,
        teacher_ckpt,
        teacher_cfg,
        lpot_base,
        name="lpotv4_psp_proxyonly_gate02",
        name_impl="proposal_support_prior_v1",
        alpha=0.6,
        prior_weights=[1.2, 0.8],
        proxy_weights=[0.3, 1.0, 1.0],
        lambda_prior=0.0,
        lambda_proxy=0.25,
        lambda_bridge=0.2,
        lambda_rank=0.05,
        lambda_align=0.0,
        support_margin=0.10,
        objectness_student_blend=0.75,
        objectness_smooth_kernel=7,
        objectness_smooth_iters=2,
        enable_feature_gate=False,
        enable_proxy_gate=True,
        support_gate_gain=0.0,
        proxy_support_gate_gain=0.2,
        fg_quantile=0.7,
        bg_quantile=0.3,
        lambda_bg=0.25,
        gamma=1.0,
    )

    # B. Full light PSP with weak pre-head gate, but without bridge.
    # Question:
    #   Is bridge consistency necessary for the LPOT mechanism?
    add_lpot_exp(
        exps,
        base,
        teacher_ckpt,
        teacher_cfg,
        lpot_base,
        name="lpotv4_psp_gate02_nobridge",
        name_impl="proposal_support_prior_v1",
        alpha=0.6,
        prior_weights=[1.2, 0.8],
        proxy_weights=[0.3, 1.0, 1.0],
        lambda_prior=0.5,
        lambda_proxy=0.25,
        lambda_bridge=0.0,
        lambda_rank=0.05,
        lambda_align=0.0,
        support_margin=0.10,
        objectness_student_blend=0.75,
        objectness_smooth_kernel=7,
        objectness_smooth_iters=2,
        enable_feature_gate=False,
        enable_proxy_gate=True,
        support_gate_gain=0.0,
        proxy_support_gate_gain=0.2,
        fg_quantile=0.7,
        bg_quantile=0.3,
        lambda_bg=0.25,
        gamma=1.0,
    )

    # C. Minimal candidate-support LPOT:
    #    proxy-only + weak pre-head gate + no bridge.
    # Question:
    #   Can Stage1 foreground prior work mainly as pre-head
    #   objectness-like candidate-support supervision?
    add_lpot_exp(
        exps,
        base,
        teacher_ckpt,
        teacher_cfg,
        lpot_base,
        name="lpotv4_psp_proxyonly_gate02_nobridge",
        name_impl="proposal_support_prior_v1",
        alpha=0.6,
        prior_weights=[1.2, 0.8],
        proxy_weights=[0.3, 1.0, 1.0],
        lambda_prior=0.0,
        lambda_proxy=0.25,
        lambda_bridge=0.0,
        lambda_rank=0.05,
        lambda_align=0.0,
        support_margin=0.10,
        objectness_student_blend=0.75,
        objectness_smooth_kernel=7,
        objectness_smooth_iters=2,
        enable_feature_gate=False,
        enable_proxy_gate=True,
        support_gate_gain=0.0,
        proxy_support_gate_gain=0.2,
        fg_quantile=0.7,
        bg_quantile=0.3,
        lambda_bg=0.25,
        gamma=1.0,
    )

    # ------------------------------------------------------------------
    # LPOT-v4 teacher/student blend ablations for minimal pre-head support
    # Purpose:
    #   Based on the current best mechanism setting:
    #     proxy-only + no bridge + weak pre-head gate
    #   test whether the proxy target should rely more on Stage1 teacher prior
    #   rather than student-generated support.
    #
    # Current reference:
    #   lpotv4_psp_proxyonly_gate02_nobridge
    #   objectness_student_blend = 0.75
    #
    # Here:
    #   blend025 means 25% student support + 75% teacher support
    #   blend050 means 50% student support + 50% teacher support
    # ------------------------------------------------------------------

    add_lpot_exp(
        exps,
        base,
        teacher_ckpt,
        teacher_cfg,
        lpot_base,
        name="lpotv4_psp_proxyonly_gate02_nobridge_blend025",
        name_impl="proposal_support_prior_v1",
        alpha=0.6,
        prior_weights=[1.2, 0.8],
        proxy_weights=[0.3, 1.0, 1.0],
        lambda_prior=0.0,
        lambda_proxy=0.25,
        lambda_bridge=0.0,
        lambda_rank=0.05,
        lambda_align=0.0,
        support_margin=0.10,
        objectness_student_blend=0.25,
        objectness_smooth_kernel=7,
        objectness_smooth_iters=2,
        enable_feature_gate=False,
        enable_proxy_gate=True,
        support_gate_gain=0.0,
        proxy_support_gate_gain=0.2,
        fg_quantile=0.7,
        bg_quantile=0.3,
        lambda_bg=0.25,
        gamma=1.0,
    )

    add_lpot_exp(
        exps,
        base,
        teacher_ckpt,
        teacher_cfg,
        lpot_base,
        name="lpotv4_psp_proxyonly_gate02_nobridge_blend050",
        name_impl="proposal_support_prior_v1",
        alpha=0.6,
        prior_weights=[1.2, 0.8],
        proxy_weights=[0.3, 1.0, 1.0],
        lambda_prior=0.0,
        lambda_proxy=0.25,
        lambda_bridge=0.0,
        lambda_rank=0.05,
        lambda_align=0.0,
        support_margin=0.10,
        objectness_student_blend=0.50,
        objectness_smooth_kernel=7,
        objectness_smooth_iters=2,
        enable_feature_gate=False,
        enable_proxy_gate=True,
        support_gate_gain=0.0,
        proxy_support_gate_gain=0.2,
        fg_quantile=0.7,
        bg_quantile=0.3,
        lambda_bg=0.25,
        gamma=1.0,
    )


    # ------------------------------------------------------------------
    # LPOT-v5: Teacher-anchored student-adaptive proxy support
    # Purpose:
    #   Explicitly transfer Stage1 foreground prior into pre-head proxy
    #   support probabilities through teacher-prior ranking, while keeping
    #   the v4-final proxy-only/no-bridge route.
    # ------------------------------------------------------------------

    add_lpot_exp(
        exps,
        base,
        teacher_ckpt,
        teacher_cfg,
        lpot_base,
        name="lpotv5_proxyonly_gate02_nobridge_tanchor005",
        name_impl="teacher_anchored_proxy_support_v1",
        alpha=0.6,
        prior_weights=[1.2, 0.8],
        proxy_weights=[0.3, 1.0, 1.0],
        lambda_prior=0.0,
        lambda_proxy=0.25,
        lambda_bridge=0.0,
        lambda_rank=0.05,
        lambda_align=0.0,
        support_margin=0.10,
        objectness_student_blend=0.75,
        objectness_smooth_kernel=7,
        objectness_smooth_iters=2,
        enable_feature_gate=False,
        enable_proxy_gate=True,
        support_gate_gain=0.0,
        proxy_support_gate_gain=0.2,
        fg_quantile=0.7,
        bg_quantile=0.3,
        lambda_bg=0.25,
        gamma=1.0,
        lambda_teacher_rank=0.05,
        lambda_teacher_bg=0.03,
        teacher_anchor_margin=0.10,
        teacher_fg_quantile=0.70,
        teacher_bg_quantile=0.30,
        teacher_anchor_layer_weights=[1.0, 1.0, 0.5],
    )

    add_lpot_exp(
        exps,
        base,
        teacher_ckpt,
        teacher_cfg,
        lpot_base,
        name="lpotv5_proxyonly_gate02_nobridge_tanchor005_p34",
        name_impl="teacher_anchored_proxy_support_v1",
        alpha=0.6,
        prior_weights=[1.2, 0.8],
        proxy_weights=[0.3, 1.0, 1.0],
        lambda_prior=0.0,
        lambda_proxy=0.25,
        lambda_bridge=0.0,
        lambda_rank=0.05,
        lambda_align=0.0,
        support_margin=0.10,
        objectness_student_blend=0.75,
        objectness_smooth_kernel=7,
        objectness_smooth_iters=2,
        enable_feature_gate=False,
        enable_proxy_gate=True,
        support_gate_gain=0.0,
        proxy_support_gate_gain=0.2,
        fg_quantile=0.7,
        bg_quantile=0.3,
        lambda_bg=0.25,
        gamma=1.0,
        lambda_teacher_rank=0.05,
        lambda_teacher_bg=0.03,
        teacher_anchor_margin=0.10,
        teacher_fg_quantile=0.70,
        teacher_bg_quantile=0.30,
        teacher_anchor_layer_weights=[1.0, 1.0, 0.3],
    )

    add_lpot_exp(
        exps,
        base,
        teacher_ckpt,
        teacher_cfg,
        lpot_base,
        name="lpotv5_proxyonly_gate02_nobridge_tanchor010_p34",
        name_impl="teacher_anchored_proxy_support_v1",
        alpha=0.6,
        prior_weights=[1.2, 0.8],
        proxy_weights=[0.3, 1.0, 1.0],
        lambda_prior=0.0,
        lambda_proxy=0.25,
        lambda_bridge=0.0,
        lambda_rank=0.05,
        lambda_align=0.0,
        support_margin=0.10,
        objectness_student_blend=0.75,
        objectness_smooth_kernel=7,
        objectness_smooth_iters=2,
        enable_feature_gate=False,
        enable_proxy_gate=True,
        support_gate_gain=0.0,
        proxy_support_gate_gain=0.2,
        fg_quantile=0.7,
        bg_quantile=0.3,
        lambda_bg=0.25,
        gamma=1.0,
        lambda_teacher_rank=0.10,
        lambda_teacher_bg=0.03,
        teacher_anchor_margin=0.10,
        teacher_fg_quantile=0.70,
        teacher_bg_quantile=0.30,
        teacher_anchor_layer_weights=[1.0, 1.0, 0.3],
    )

    # ------------------------------------------------------------------
    # LPOT-v4.1: score-level proposal-support regularization
    # Purpose:
    #   Keep the v4-final proxy-only/no-bridge route, but connect Stage1
    #   foreground prior to the detector decision level by lightly ranking
    #   YOLO class-score maps: GT/object regions should score above
    #   low-prior background regions. This is not heatmap imitation.
    # ------------------------------------------------------------------

    add_lpot_exp(
        exps,
        base,
        teacher_ckpt,
        teacher_cfg,
        lpot_base,
        name="lpotv41_proxyonly_gate02_nobridge_scorerank002",
        name_impl="score_level_proposal_support_v1",
        alpha=0.6,
        prior_weights=[1.2, 0.8],
        proxy_weights=[0.3, 1.0, 1.0],
        lambda_prior=0.0,
        lambda_proxy=0.25,
        lambda_bridge=0.0,
        lambda_rank=0.05,
        lambda_align=0.0,
        support_margin=0.10,
        objectness_student_blend=0.75,
        objectness_smooth_kernel=7,
        objectness_smooth_iters=2,
        enable_feature_gate=False,
        enable_proxy_gate=True,
        support_gate_gain=0.0,
        proxy_support_gate_gain=0.2,
        fg_quantile=0.7,
        bg_quantile=0.3,
        lambda_bg=0.25,
        gamma=1.0,
        lambda_teacher_rank=0.0,
        lambda_teacher_bg=0.0,
        teacher_anchor_layer_weights=[],
        lambda_score_rank=0.02,
        lambda_score_bg=0.0,
        score_rank_margin=0.10,
        score_fg_quantile=0.70,
        score_bg_quantile=0.30,
        score_rank_layer_weights=[1.0, 1.0, 0.3],
        score_gt_expand_ratio=1.5,
        score_use_gt_pos=True,
        score_use_teacher_fg_pos=False,
    )

    add_lpot_exp(
        exps,
        base,
        teacher_ckpt,
        teacher_cfg,
        lpot_base,
        name="lpotv41_proxyonly_gate02_nobridge_scorerank002_bg001",
        name_impl="score_level_proposal_support_v1",
        alpha=0.6,
        prior_weights=[1.2, 0.8],
        proxy_weights=[0.3, 1.0, 1.0],
        lambda_prior=0.0,
        lambda_proxy=0.25,
        lambda_bridge=0.0,
        lambda_rank=0.05,
        lambda_align=0.0,
        support_margin=0.10,
        objectness_student_blend=0.75,
        objectness_smooth_kernel=7,
        objectness_smooth_iters=2,
        enable_feature_gate=False,
        enable_proxy_gate=True,
        support_gate_gain=0.0,
        proxy_support_gate_gain=0.2,
        fg_quantile=0.7,
        bg_quantile=0.3,
        lambda_bg=0.25,
        gamma=1.0,
        lambda_teacher_rank=0.0,
        lambda_teacher_bg=0.0,
        teacher_anchor_layer_weights=[],
        lambda_score_rank=0.02,
        lambda_score_bg=0.01,
        score_rank_margin=0.10,
        score_fg_quantile=0.70,
        score_bg_quantile=0.30,
        score_rank_layer_weights=[1.0, 1.0, 0.3],
        score_gt_expand_ratio=1.5,
        score_use_gt_pos=True,
        score_use_teacher_fg_pos=False,
    )

    # ------------------------------------------------------------------
    # LPOT-v4.1 additional ablations: score-level proposal-support tuning
    # Purpose:
    #   Based on the current observations:
    #     - scorerank002 improves peak mAP50-95 but is less stable.
    #     - scorerank002_bg001 gives better final stability and high precision.
    #     - v4.1 tends to make candidates more foreground-oriented, but may
    #       slightly reduce TP count and weaken P3/P5 support.
    #
    # These groups only tune score-level regularization strength and
    # layer weights. The v4-final proxy-only/no-bridge route is unchanged.
    # ------------------------------------------------------------------

    # A. Weaker score-rank only.
    # Question:
    #   Is lambda_score_rank=0.02 too strong? Can 0.01 keep the benefit
    #   while reducing candidate shrinkage?
    add_lpot_exp(
        exps,
        base,
        teacher_ckpt,
        teacher_cfg,
        lpot_base,
        name="lpotv41_proxyonly_gate02_nobridge_scorerank001",
        name_impl="score_level_proposal_support_v1",
        alpha=0.6,
        prior_weights=[1.2, 0.8],
        proxy_weights=[0.3, 1.0, 1.0],
        lambda_prior=0.0,
        lambda_proxy=0.25,
        lambda_bridge=0.0,
        lambda_rank=0.05,
        lambda_align=0.0,
        support_margin=0.10,
        objectness_student_blend=0.75,
        objectness_smooth_kernel=7,
        objectness_smooth_iters=2,
        enable_feature_gate=False,
        enable_proxy_gate=True,
        support_gate_gain=0.0,
        proxy_support_gate_gain=0.2,
        fg_quantile=0.7,
        bg_quantile=0.3,
        lambda_bg=0.25,
        gamma=1.0,
        lambda_teacher_rank=0.0,
        lambda_teacher_bg=0.0,
        teacher_anchor_layer_weights=[],
        lambda_score_rank=0.01,
        lambda_score_bg=0.0,
        score_rank_margin=0.10,
        score_fg_quantile=0.70,
        score_bg_quantile=0.30,
        score_rank_layer_weights=[1.0, 1.0, 0.3],
        score_gt_expand_ratio=1.5,
        score_use_gt_pos=True,
        score_use_teacher_fg_pos=False,
    )

    # B. Weaker score-rank + very weak background score suppression.
    # Question:
    #   Is bg001 still useful when score-rank is reduced to 0.01?
    #   This is the most important follow-up to current scorerank002_bg001.
    add_lpot_exp(
        exps,
        base,
        teacher_ckpt,
        teacher_cfg,
        lpot_base,
        name="lpotv41_proxyonly_gate02_nobridge_scorerank001_bg001",
        name_impl="score_level_proposal_support_v1",
        alpha=0.6,
        prior_weights=[1.2, 0.8],
        proxy_weights=[0.3, 1.0, 1.0],
        lambda_prior=0.0,
        lambda_proxy=0.25,
        lambda_bridge=0.0,
        lambda_rank=0.05,
        lambda_align=0.0,
        support_margin=0.10,
        objectness_student_blend=0.75,
        objectness_smooth_kernel=7,
        objectness_smooth_iters=2,
        enable_feature_gate=False,
        enable_proxy_gate=True,
        support_gate_gain=0.0,
        proxy_support_gate_gain=0.2,
        fg_quantile=0.7,
        bg_quantile=0.3,
        lambda_bg=0.25,
        gamma=1.0,
        lambda_teacher_rank=0.0,
        lambda_teacher_bg=0.0,
        teacher_anchor_layer_weights=[],
        lambda_score_rank=0.01,
        lambda_score_bg=0.01,
        score_rank_margin=0.10,
        score_fg_quantile=0.70,
        score_bg_quantile=0.30,
        score_rank_layer_weights=[1.0, 1.0, 0.3],
        score_gt_expand_ratio=1.5,
        score_use_gt_pos=True,
        score_use_teacher_fg_pos=False,
    )

    # C. Original score-rank strength + weaker background suppression.
    # Question:
    #   Is bg001 slightly too suppressive? Test bg0005 while keeping
    #   the current high-peak score-rank strength.
    add_lpot_exp(
        exps,
        base,
        teacher_ckpt,
        teacher_cfg,
        lpot_base,
        name="lpotv41_proxyonly_gate02_nobridge_scorerank002_bg0005",
        name_impl="score_level_proposal_support_v1",
        alpha=0.6,
        prior_weights=[1.2, 0.8],
        proxy_weights=[0.3, 1.0, 1.0],
        lambda_prior=0.0,
        lambda_proxy=0.25,
        lambda_bridge=0.0,
        lambda_rank=0.05,
        lambda_align=0.0,
        support_margin=0.10,
        objectness_student_blend=0.75,
        objectness_smooth_kernel=7,
        objectness_smooth_iters=2,
        enable_feature_gate=False,
        enable_proxy_gate=True,
        support_gate_gain=0.0,
        proxy_support_gate_gain=0.2,
        fg_quantile=0.7,
        bg_quantile=0.3,
        lambda_bg=0.25,
        gamma=1.0,
        lambda_teacher_rank=0.0,
        lambda_teacher_bg=0.0,
        teacher_anchor_layer_weights=[],
        lambda_score_rank=0.02,
        lambda_score_bg=0.005,
        score_rank_margin=0.10,
        score_fg_quantile=0.70,
        score_bg_quantile=0.30,
        score_rank_layer_weights=[1.0, 1.0, 0.3],
        score_gt_expand_ratio=1.5,
        score_use_gt_pos=True,
        score_use_teacher_fg_pos=False,
    )

    # D. Weaker score-rank + weaker background suppression.
    # Question:
    #   Can a gentler score-level regularizer keep the stability of bg001
    #   while reducing TP shrinkage?
    add_lpot_exp(
        exps,
        base,
        teacher_ckpt,
        teacher_cfg,
        lpot_base,
        name="lpotv41_proxyonly_gate02_nobridge_scorerank001_bg0005",
        name_impl="score_level_proposal_support_v1",
        alpha=0.6,
        prior_weights=[1.2, 0.8],
        proxy_weights=[0.3, 1.0, 1.0],
        lambda_prior=0.0,
        lambda_proxy=0.25,
        lambda_bridge=0.0,
        lambda_rank=0.05,
        lambda_align=0.0,
        support_margin=0.10,
        objectness_student_blend=0.75,
        objectness_smooth_kernel=7,
        objectness_smooth_iters=2,
        enable_feature_gate=False,
        enable_proxy_gate=True,
        support_gate_gain=0.0,
        proxy_support_gate_gain=0.2,
        fg_quantile=0.7,
        bg_quantile=0.3,
        lambda_bg=0.25,
        gamma=1.0,
        lambda_teacher_rank=0.0,
        lambda_teacher_bg=0.0,
        teacher_anchor_layer_weights=[],
        lambda_score_rank=0.01,
        lambda_score_bg=0.005,
        score_rank_margin=0.10,
        score_fg_quantile=0.70,
        score_bg_quantile=0.30,
        score_rank_layer_weights=[1.0, 1.0, 0.3],
        score_gt_expand_ratio=1.5,
        score_use_gt_pos=True,
        score_use_teacher_fg_pos=False,
    )

    # E. Weaker score-rank + bg001 + P3/P5-protected scale weights.
    # Question:
    #   Current v4.1 mainly strengthens P4 and weakens P3/P5.
    #   This group slightly increases P3 and P5 weights to protect
    #   small-object and high-level support while keeping stable bg001.
    add_lpot_exp(
        exps,
        base,
        teacher_ckpt,
        teacher_cfg,
        lpot_base,
        name="lpotv41_proxyonly_gate02_nobridge_scorerank001_bg001_w120105",
        name_impl="score_level_proposal_support_v1",
        alpha=0.6,
        prior_weights=[1.2, 0.8],
        proxy_weights=[0.3, 1.0, 1.0],
        lambda_prior=0.0,
        lambda_proxy=0.25,
        lambda_bridge=0.0,
        lambda_rank=0.05,
        lambda_align=0.0,
        support_margin=0.10,
        objectness_student_blend=0.75,
        objectness_smooth_kernel=7,
        objectness_smooth_iters=2,
        enable_feature_gate=False,
        enable_proxy_gate=True,
        support_gate_gain=0.0,
        proxy_support_gate_gain=0.2,
        fg_quantile=0.7,
        bg_quantile=0.3,
        lambda_bg=0.25,
        gamma=1.0,
        lambda_teacher_rank=0.0,
        lambda_teacher_bg=0.0,
        teacher_anchor_layer_weights=[],
        lambda_score_rank=0.01,
        lambda_score_bg=0.01,
        score_rank_margin=0.10,
        score_fg_quantile=0.70,
        score_bg_quantile=0.30,
        # P3/P4/P5 weights
        score_rank_layer_weights=[1.2, 1.0, 0.5],
        score_gt_expand_ratio=1.5,
        score_use_gt_pos=True,
        score_use_teacher_fg_pos=False,
    )

    cfg_gen_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    cfg_gen_dir.mkdir(parents=True, exist_ok=True)
    for name, cfg in exps.items():
        save_yaml(cfg, cfg_gen_dir / f"{name}.yaml")
        print(f"[OK] {cfg_gen_dir / f'{name}.yaml'}")


if __name__ == "__main__":
    main()
