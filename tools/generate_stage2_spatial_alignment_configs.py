import argparse
import copy
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LEVEL_MAP = {
    "l1": ([0], [4]),
    "l1l2": ([0, 1], [4, 6]),
    "l2l3": ([1, 2], [6, 8]),
    "l1l2l3": ([0, 1, 2], [4, 6, 8]),
}

WEIGHTS = {
    "l1": [1.0],
    "l1l2": [0.5, 1.0],
    "l2l3": [1.2, 0.8],
    "l1l2l3": [0.4, 1.0, 0.8],
}


def resolve_path(path_str: str, base_dir: Path | None = None) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    if base_dir is not None:
        return (base_dir / p).resolve()
    return (PROJECT_ROOT / p).resolve()


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def build_experiments(args):
    return [
        {
            "file": "baseline.yaml",
            "name": "baseline",
            "stage1_enabled": False,
            "ckpt": "",
            "spatial_enabled": False,
            "levels": "l1l2l3",
            "alpha": args.alpha,
            "desc": "Pure detector baseline.",
        },
        {
            "file": "full_no_freeze.yaml",
            "name": "full_no_freeze",
            "stage1_enabled": True,
            "ckpt": args.full_ckpt,
            "spatial_enabled": False,
            "levels": "l1l2l3",
            "alpha": args.alpha,
            "desc": "Stage1 full initialization without spatial alignment.",
        },
        {
            "file": "full_spatial_l2l3.yaml",
            "name": "full_spatial_l2l3",
            "stage1_enabled": True,
            "ckpt": args.full_ckpt,
            "spatial_enabled": True,
            "levels": "l2l3",
            "alpha": args.alpha,
            "desc": "Reference L2/L3 spatial map alignment.",
        },
        {
            "file": "full_spatial_l1l2l3.yaml",
            "name": "full_spatial_l1l2l3",
            "stage1_enabled": True,
            "ckpt": args.full_ckpt,
            "spatial_enabled": True,
            "levels": "l1l2l3",
            "alpha": args.alpha,
            "desc": "Three-level L1/L2/L3 spatial map alignment.",
        },
        {
            "file": "full_spatial_l1l2.yaml",
            "name": "full_spatial_l1l2",
            "stage1_enabled": True,
            "ckpt": args.full_ckpt,
            "spatial_enabled": True,
            "levels": "l1l2",
            "alpha": args.alpha,
            "desc": "L1/L2 lesion-detail and lesion-region spatial alignment.",
        },
    ]


def main():
    parser = argparse.ArgumentParser(description="Generate Stage2 L1/L2/L3 spatial-alignment configs from det_config.yaml")
    parser.add_argument("--base_config", type=str, default="./configs/det_config.yaml")
    parser.add_argument("--out_dir", type=str, default="./configs/stage2_spatial_alignment_generated")
    parser.add_argument("--project", type=str, default="runs/glcp_stage2_yolo_det")
    parser.add_argument("--full_ckpt", type=str, default="./runs/glcp_stage1_yolo_det/use_pos_mask/best.pth")
    parser.add_argument("--teacher_ssl_config", type=str, default="./configs/stage1_ablation_generated/use_pos_mask.yaml")
    parser.add_argument("--alpha", type=float, default=0.03)
    args = parser.parse_args()

    base_path = resolve_path(args.base_config)
    out_dir = resolve_path(args.out_dir)
    base_cfg = load_yaml(base_path)
    experiments = build_experiments(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"base_config": str(base_path), "output_dir": str(out_dir), "experiments": []}

    for exp in experiments:
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("train", {})
        cfg.setdefault("stage1_init", {})
        cfg.setdefault("spatial_alignment", {})

        for key in ["foreground_prior_distillation", "leaf_prior_auxiliary", "leaf_prior_objectness_transfer"]:
            cfg.setdefault(key, {})
            cfg[key]["enabled"] = False

        cfg["train"]["name"] = exp["name"]
        cfg["train"]["project"] = args.project
        cfg["stage1_init"]["enabled"] = exp["stage1_enabled"]
        cfg["stage1_init"]["ckpt_path"] = exp["ckpt"]

        teacher_indices, student_indices = LEVEL_MAP[exp["levels"]]
        cfg["spatial_alignment"]["enabled"] = exp["spatial_enabled"]
        cfg["spatial_alignment"]["teacher_ckpt_path"] = exp["ckpt"]
        cfg["spatial_alignment"]["teacher_ssl_config"] = args.teacher_ssl_config
        cfg["spatial_alignment"]["teacher_branch"] = "pos_feats"
        cfg["spatial_alignment"]["teacher_feature_indices"] = teacher_indices
        cfg["spatial_alignment"]["student_layer_indices"] = student_indices
        cfg["spatial_alignment"]["layer_weights"] = WEIGHTS[exp["levels"]]
        cfg["spatial_alignment"]["alpha"] = exp["alpha"]

        out_path = out_dir / exp["file"]
        save_yaml(cfg, out_path)
        manifest["experiments"].append({
            "file": str(out_path),
            "name": exp["name"],
            "desc": exp["desc"],
            "spatial_enabled": exp["spatial_enabled"],
            "levels": exp["levels"],
            "teacher_indices": teacher_indices,
            "student_indices": student_indices,
            "alpha": exp["alpha"],
        })
        print(f"[OK] Generated: {out_path}")

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Stage2 L1/L2/L3 spatial-alignment configs generated successfully.")
    print("Base config :", base_path)
    print("Output dir  :", out_dir)
    print("Manifest    :", manifest_path)
    print("=" * 80)


if __name__ == "__main__":
    main()
