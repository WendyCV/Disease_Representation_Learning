from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import yaml


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def resolve_path(p: str) -> Path:
    return Path(p).expanduser().resolve()


def infer_stage1_ckpt(stage1_root: Path, variant_name: str, ckpt_name: str) -> Path:
    return stage1_root / variant_name / ckpt_name


def infer_stage1_ssl_cfg(stage1_cfg_dirs: list[Path], variant_name: str) -> str:
    for d in stage1_cfg_dirs:
        p = d / f"{variant_name}.yaml"
        if p.is_file():
            return str(p)
    return ""


def disable_all_aux_routes(cfg: dict):
    cfg.setdefault("freeze", {})
    cfg["freeze"]["enabled"] = False
    cfg["freeze"]["layer_indices"] = [0, 1, 2, 3, 4]

    cfg.setdefault("spatial_alignment", {})
    cfg["spatial_alignment"]["enabled"] = False

    cfg.setdefault("foreground_prior_distillation", {})
    cfg["foreground_prior_distillation"]["enabled"] = False

    cfg.setdefault("leaf_prior_auxiliary", {})
    cfg["leaf_prior_auxiliary"]["enabled"] = False

    cfg.setdefault("leaf_prior_objectness_transfer", {})
    cfg["leaf_prior_objectness_transfer"]["enabled"] = False


def apply_leafaux_best(cfg: dict):
    lpa = cfg.setdefault("leaf_prior_auxiliary", {})
    lpa["enabled"] = True
    lpa["teacher_feature_indices"] = [0, 1, 2]
    lpa["student_layer_indices"] = [4, 6, 8]
    lpa["layer_weights"] = [0.4, 0.8, 1.0]
    lpa["alpha"] = 1.0
    lpa["teacher_branch"] = "pos_feats"
    lpa["loss_type"] = "bce"
    lpa["enable_gate"] = True
    lpa["detach_teacher"] = True
    lpa.setdefault("gate_init_beta", 2.0)
    lpa.setdefault("gate_gain", 1.0)
    lpa.setdefault("lambda_bg", 0.25)
    lpa.setdefault("bg_quantile", 0.20)


def build_cfg(
    base_cfg: dict,
    *,
    exp_name: str,
    project: str,
    stage1_enabled: bool,
    stage1_ckpt: str = "",
    stage1_ssl_cfg: str = "",
    enable_leafaux: bool = False,
    mode: str = "direct_transfer",
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("train", {})
    cfg.setdefault("stage1_init", {})

    cfg["train"]["name"] = exp_name
    cfg["train"]["project"] = project

    cfg["stage1_init"]["enabled"] = bool(stage1_enabled)
    cfg["stage1_init"]["ckpt_path"] = stage1_ckpt if stage1_enabled else ""

    disable_all_aux_routes(cfg)

    if enable_leafaux:
        apply_leafaux_best(cfg)

    cfg.setdefault("metadata", {})
    cfg["metadata"]["generator"] = "generate_stage2_rpd_minval_configs.py"
    cfg["metadata"]["mode"] = mode
    cfg["metadata"]["stage1_variant"] = exp_name.replace("full_no_freeze_", "").replace("leafaux_best_", "") if stage1_enabled else "baseline"
    cfg["metadata"]["stage1_ssl_config"] = stage1_ssl_cfg

    return cfg


def build_experiments(args, mode: str, stage1_root: Path, stage1_cfg_dirs: list[Path]):
    experiments = []

    if not args.skip_baseline:
        experiments.append(
            dict(
                file="baseline.yaml",
                name="baseline",
                stage1_enabled=False,
                stage1_ckpt="",
                stage1_ssl_cfg="",
                enable_leafaux=False,
                desc="Pure detector baseline without Stage1 initialization.",
            )
        )

    def add_variant(variant_name: str, prefix: str, enable_leafaux: bool, desc_prefix: str):
        ckpt_path = infer_stage1_ckpt(stage1_root, variant_name, args.ckpt_name)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Stage1 checkpoint not found: {ckpt_path}")

        ssl_cfg = infer_stage1_ssl_cfg(stage1_cfg_dirs, variant_name)
        experiments.append(
            dict(
                file=f"{prefix}_{variant_name}.yaml",
                name=f"{prefix}_{variant_name}",
                stage1_enabled=True,
                stage1_ckpt=str(ckpt_path),
                stage1_ssl_cfg=ssl_cfg,
                enable_leafaux=enable_leafaux,
                desc=f"{desc_prefix}: {variant_name}",
            )
        )

    if mode == "direct_transfer":
        for variant_name in [args.lesion_sensitive_name, args.rpd_w010_name, args.rpd_w020_name]:
            add_variant(
                variant_name=variant_name,
                prefix="full_no_freeze",
                enable_leafaux=False,
                desc_prefix="Direct Stage2 transfer using Stage1 checkpoint",
            )
    elif mode == "leafaux_compare":
        for variant_name in [args.lesion_sensitive_name, args.rpd_w020_name]:
            add_variant(
                variant_name=variant_name,
                prefix="leafaux_best",
                enable_leafaux=True,
                desc_prefix="LeafAux-best using Stage1 checkpoint",
            )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return experiments


def main():
    parser = argparse.ArgumentParser(
        description="Generate Stage2 validation configs for direct-transfer or LeafAux-vs-RPD comparison."
    )
    parser.add_argument("--mode", type=str, default="direct_transfer",
                        choices=["direct_transfer", "leafaux_compare"],
                        help="Generation mode.")
    parser.add_argument("--base_config", type=str, default="./configs/det_config.yaml", help="Base det config yaml")
    parser.add_argument("--out_dir", type=str, default="./configs/stage2_rpd_minval_generated", help="Output dir")
    parser.add_argument("--project", type=str, default="./runs/glcp_stage2_yolo_det", help="train.project")

    parser.add_argument(
        "--stage1_root",
        type=str,
        default="./runs/glcp_stage1_yolo_det",
        help="Directory containing Stage1 experiment folders",
    )
    parser.add_argument(
        "--stage1_cfg_dirs",
        type=str,
        nargs="*",
        default=[
            "./configs/stage1_ablation_generated",
            "./configs/stage1_layer_weights_generated",
            "./configs/stage1_rpd_generated",
        ],
        help="Directories used to infer stage1 yaml config files for bookkeeping",
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="best.pth",
        help="Checkpoint filename inside each Stage1 experiment directory",
    )

    parser.add_argument(
        "--lesion_sensitive_name",
        type=str,
        default="use_pos_mask_sw_lesion_sensitive",
        help="Stage1 lesion_sensitive experiment folder name",
    )
    parser.add_argument(
        "--rpd_w010_name",
        type=str,
        default="use_pos_mask_rpd_hybrid_w010",
        help="Stage1 RPD hybrid w010 experiment folder name",
    )
    parser.add_argument(
        "--rpd_w020_name",
        type=str,
        default="use_pos_mask_rpd_hybrid_w020",
        help="Stage1 RPD hybrid w020 experiment folder name",
    )

    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Do not generate the pure baseline config",
    )

    args = parser.parse_args()

    base_path = resolve_path(args.base_config)
    out_dir = resolve_path(args.out_dir)
    stage1_root = resolve_path(args.stage1_root)
    stage1_cfg_dirs = [resolve_path(p) for p in args.stage1_cfg_dirs]

    if not base_path.is_file():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    base_cfg = load_yaml(base_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = build_experiments(args, args.mode, stage1_root, stage1_cfg_dirs)

    manifest = {
        "mode": args.mode,
        "base_config": str(base_path),
        "output_dir": str(out_dir),
        "project": args.project,
        "stage1_root": str(stage1_root),
        "ckpt_name": args.ckpt_name,
        "experiments": [],
    }

    for exp in experiments:
        cfg = build_cfg(
            base_cfg,
            exp_name=exp["name"],
            project=args.project,
            stage1_enabled=exp["stage1_enabled"],
            stage1_ckpt=exp["stage1_ckpt"],
            stage1_ssl_cfg=exp["stage1_ssl_cfg"],
            enable_leafaux=exp["enable_leafaux"],
            mode=args.mode,
        )
        out_path = out_dir / exp["file"]
        save_yaml(cfg, out_path)

        manifest["experiments"].append(
            {
                "file": str(out_path),
                "name": exp["name"],
                "desc": exp["desc"],
                "stage1_enabled": exp["stage1_enabled"],
                "stage1_ckpt": exp["stage1_ckpt"],
                "stage1_ssl_config": exp["stage1_ssl_cfg"],
                "enable_leafaux": exp["enable_leafaux"],
            }
        )
        print(f"[OK] Generated: {out_path}")

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"Stage2 config generation finished. mode={args.mode}")
    print("Base config :", base_path)
    print("Output dir  :", out_dir)
    print("Manifest    :", manifest_path)
    print("=" * 80)


if __name__ == "__main__":
    main()
