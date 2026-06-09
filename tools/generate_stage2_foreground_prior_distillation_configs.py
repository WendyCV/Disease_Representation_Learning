from __future__ import annotations

import copy
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CFG = PROJECT_ROOT / "configs" / "det_config.yaml"
OUT_DIR = PROJECT_ROOT / "configs" / "stage2_foreground_prior_distillation_generated"
TEACHER_SSL_CFG = "./configs/stage1_ablation_generated/use_pos_mask.yaml"


LEVEL_MAP = {
    # Stage1 layer_indices=[4,6,8] -> teacher feature indices [0,1,2]
    # Stage2 YOLO backbone hooks: model.4/model.6/model.8
    "l1": ([0], [4], [1.0]),
    "l1l2": ([0, 1], [4, 6], [0.5, 1.0]),
    "l2l3": ([1, 2], [6, 8], [1.2, 0.8]),
    "l1l2l3": ([0, 1, 2], [4, 6, 8], [0.4, 1.0, 0.8]),
}


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def disable_other_routes(cfg: dict):
    for key in ["spatial_alignment", "leaf_prior_auxiliary", "leaf_prior_objectness_transfer"]:
        cfg.setdefault(key, {})
        cfg[key]["enabled"] = False


def build_cfg(base: dict, name: str, *, stage1_enabled: bool, fpd_cfg: dict | None):
    cfg = copy.deepcopy(base)
    cfg.setdefault("train", {})
    cfg["train"]["name"] = name

    cfg.setdefault("stage1_init", {})
    cfg["stage1_init"]["enabled"] = bool(stage1_enabled)

    disable_other_routes(cfg)

    cfg.setdefault("foreground_prior_distillation", {})
    if fpd_cfg is None:
        cfg["foreground_prior_distillation"]["enabled"] = False
    else:
        cfg["foreground_prior_distillation"] = copy.deepcopy(fpd_cfg)
        cfg["foreground_prior_distillation"]["enabled"] = True
    return cfg


def make_fpd_cfg(
    teacher_ckpt: str,
    fpd_base: dict,
    *,
    levels: str,
    alpha: float,
    weights: list[float] | None = None,
):
    if levels not in LEVEL_MAP:
        raise ValueError(f"Unknown levels: {levels}")

    teacher_indices, student_indices, default_weights = LEVEL_MAP[levels]
    if weights is None:
        weights = default_weights

    if not (len(teacher_indices) == len(student_indices) == len(weights)):
        raise ValueError(
            f"Length mismatch for {levels}: "
            f"teacher={teacher_indices}, student={student_indices}, weights={weights}"
        )

    return {
        "teacher_ckpt_path": teacher_ckpt,
        "teacher_ssl_config": TEACHER_SSL_CFG,
        "teacher_branch": "pos_feats",
        "teacher_feature_indices": teacher_indices,
        "student_layer_indices": student_indices,
        "layer_weights": [float(x) for x in weights],
        "alpha": float(alpha),
        "log_interval": int(fpd_base.get("log_interval", 20)),
        "fg_quantile": float(fpd_base.get("fg_quantile", 0.7)),
        "bg_quantile": float(fpd_base.get("bg_quantile", 0.3)),
        "lambda_fg": float(fpd_base.get("lambda_fg", 1.0)),
        "lambda_bg": float(fpd_base.get("lambda_bg", 0.5)),
    }


def main():
    base = load_yaml(BASE_CFG)
    teacher_ckpt = base.get("stage1_init", {}).get("ckpt_path", "")
    fpd_base = base.get("foreground_prior_distillation", {})

    exps = {
        # fixed references
        "baseline": build_cfg(base, "baseline", stage1_enabled=False, fpd_cfg=None),
        "full_no_freeze": build_cfg(base, "full_no_freeze", stage1_enabled=True, fpd_cfg=None),

        # old best/reference route: keep it for fair comparison with previous results
        "full_fpd_l2l3_a003_w12_08": build_cfg(
            base,
            "full_fpd_l2l3_a003_w12_08",
            stage1_enabled=True,
            fpd_cfg=make_fpd_cfg(teacher_ckpt, fpd_base, levels="l2l3", alpha=0.03, weights=[1.2, 0.8]),
        ),

        # new main route: match Stage1's L1/L2/L3 design
        "full_fpd_l1l2l3_a002_w04_10_08": build_cfg(
            base,
            "full_fpd_l1l2l3_a002_w04_10_08",
            stage1_enabled=True,
            fpd_cfg=make_fpd_cfg(teacher_ckpt, fpd_base, levels="l1l2l3", alpha=0.02, weights=[0.4, 1.0, 0.8]),
        ),

        # key scientific controls for L1 lesion-detail contribution
        "full_fpd_l1l2_a002_w05_10": build_cfg(
            base,
            "full_fpd_l1l2_a002_w05_10",
            stage1_enabled=True,
            fpd_cfg=make_fpd_cfg(teacher_ckpt, fpd_base, levels="l1l2", alpha=0.02, weights=[0.5, 1.0]),
        ),
        "full_fpd_l1_a001_w10": build_cfg(
            base,
            "full_fpd_l1_a001_w10",
            stage1_enabled=True,
            fpd_cfg=make_fpd_cfg(teacher_ckpt, fpd_base, levels="l1", alpha=0.01, weights=[1.0]),
        ),

        # conservative variants around the new main route
        "full_fpd_l1l2l3_a0015_w04_10_08": build_cfg(
            base,
            "full_fpd_l1l2l3_a0015_w04_10_08",
            stage1_enabled=True,
            fpd_cfg=make_fpd_cfg(teacher_ckpt, fpd_base, levels="l1l2l3", alpha=0.015, weights=[0.4, 1.0, 0.8]),
        ),
        "full_fpd_l1l2l3_a002_w06_10_08": build_cfg(
            base,
            "full_fpd_l1l2l3_a002_w06_10_08",
            stage1_enabled=True,
            fpd_cfg=make_fpd_cfg(teacher_ckpt, fpd_base, levels="l1l2l3", alpha=0.02, weights=[0.6, 1.0, 0.8]),
        ),
        "full_fpd_l1l2l3_a002_w04_08_10": build_cfg(
            base,
            "full_fpd_l1l2l3_a002_w04_08_10",
            stage1_enabled=True,
            fpd_cfg=make_fpd_cfg(teacher_ckpt, fpd_base, levels="l1l2l3", alpha=0.02, weights=[0.4, 0.8, 1.0]),
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, cfg in exps.items():
        save_yaml(cfg, OUT_DIR / f"{name}.yaml")
        print(f"[OK] {OUT_DIR / f'{name}.yaml'}")


if __name__ == "__main__":
    main()
