import argparse
import copy
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRESETS = [
    {
        "file": "full_no_freeze.yaml",
        "name": "full_no_freeze",
        "desc": "Reference full fine-tuning after Stage1 initialization.",
        "control": {"enabled": False},
    },
    {
        "file": "full_ctrl_relaxed_b1.yaml",
        "name": "full_ctrl_relaxed_b1",
        "desc": "Only shallow stage is slowed mildly; head is slightly boosted.",
        "control": {
            "enabled": True,
            "strategy_name": "relaxed_b1",
            "group_lr_scales": {"stage_b1": 0.5, "stage_b2": 1.0, "stage_b3": 1.0, "head": 1.25},
        },
    },
    {
        "file": "full_ctrl_relaxed_pyramid.yaml",
        "name": "full_ctrl_relaxed_pyramid",
        "desc": "Mild discriminative fine-tuning with boosted head learning.",
        "control": {
            "enabled": True,
            "strategy_name": "relaxed_pyramid",
            "group_lr_scales": {"stage_b1": 0.5, "stage_b2": 0.8, "stage_b3": 1.0, "head": 1.25},
        },
    },
    {
        "file": "full_ctrl_two_phase_release.yaml",
        "name": "full_ctrl_two_phase_release",
        "desc": "Protect shallow transfer early, then release for full detector adaptation.",
        "control": {
            "enabled": True,
            "strategy_name": "two_phase_release",
            "phase_epochs": [15],
            "phase_group_lr_scales": [
                {"stage_b1": 0.35, "stage_b2": 0.70, "stage_b3": 1.00, "head": 1.25},
                {"stage_b1": 0.75, "stage_b2": 1.00, "stage_b3": 1.00, "head": 1.00},
            ],
        },
    },
    {
        "file": "full_ctrl_two_phase_headboost.yaml",
        "name": "full_ctrl_two_phase_headboost",
        "desc": "Stronger early head adaptation while preserving transferred shallow cues.",
        "control": {
            "enabled": True,
            "strategy_name": "two_phase_headboost",
            "phase_epochs": [15],
            "phase_group_lr_scales": [
                {"stage_b1": 0.35, "stage_b2": 0.70, "stage_b3": 1.00, "head": 1.50},
                {"stage_b1": 0.75, "stage_b2": 1.00, "stage_b3": 1.00, "head": 1.10},
            ],
        },
    },
]


def resolve_path(path_str: str, base_dir: Path = None) -> Path:
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


def main():
    parser = argparse.ArgumentParser(description="Generate Stage2 transfer-attack YAML configs")
    parser.add_argument("--base_config", type=str, default="./configs/det_config.yaml")
    parser.add_argument("--out_dir", type=str, default="./configs/stage2_transfer_attack_generated")
    parser.add_argument("--project", type=str, default="runs/glcp_stage2_yolo_det")
    parser.add_argument("--full_ckpt", type=str, default="./runs/glcp_stage1_yolo_det/use_pos_mask/best.pth")
    args = parser.parse_args()

    base_path = resolve_path(args.base_config)
    out_dir = resolve_path(args.out_dir)
    if not base_path.is_file():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    base_cfg = load_yaml(base_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"base_config": str(base_path), "output_dir": str(out_dir), "experiments": []}
    for exp in PRESETS:
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("train", {})
        cfg.setdefault("stage1_init", {})
        cfg.setdefault("freeze", {})

        cfg["train"]["name"] = exp["name"]
        cfg["train"]["project"] = args.project
        cfg["stage1_init"]["enabled"] = True
        cfg["stage1_init"]["ckpt_path"] = args.full_ckpt
        cfg["freeze"]["enabled"] = False
        cfg["freeze"]["stage_names"] = []
        cfg["freeze"]["layer_indices"] = []

        out_path = out_dir / exp["file"]
        save_yaml(cfg, out_path)
        manifest["experiments"].append({
            "file": str(out_path),
            "name": exp["name"],
            "desc": exp["desc"],
        })
        print(f"[OK] Generated: {out_path}")

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Stage2 transfer-attack configs generated successfully.")
    print("Base config :", base_path)
    print("Output dir  :", out_dir)
    print("Manifest    :", manifest_path)
    print("=" * 80)


if __name__ == "__main__":
    main()
