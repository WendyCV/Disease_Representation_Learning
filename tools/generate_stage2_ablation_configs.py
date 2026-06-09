import argparse
import copy
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def build_experiments(args):
    return [
        {
            "file": "baseline.yaml",
            "name": "baseline",
            "stage1_enabled": False,
            "ckpt": "",
            "freeze_enabled": False,
            "freeze_layers": [0, 1, 2, 3, 4],
            "desc": "No stage1 initialization. Pure detector baseline.",
        },
        {
            "file": "wo_mask.yaml",
            "name": "wo_mask",
            "stage1_enabled": True,
            "ckpt": args.wo_mask_ckpt,
            "freeze_enabled": False,
            "freeze_layers": [0, 1, 2, 3, 4],
            "desc": "Initialize detector with stage1 wo_mask checkpoint.",
        },
        {
            "file": "wo_pos.yaml",
            "name": "wo_pos",
            "stage1_enabled": True,
            "ckpt": args.wo_pos_ckpt,
            "freeze_enabled": False,
            "freeze_layers": [0, 1, 2, 3, 4],
            "desc": "Initialize detector with stage1 wo_pos checkpoint.",
        },
        {
            "file": "full_no_freeze.yaml",
            "name": "full_no_freeze",
            "stage1_enabled": True,
            "ckpt": args.full_ckpt,
            "freeze_enabled": False,
            "freeze_layers": [0, 1, 2, 3, 4],
            "desc": "Initialize detector with stage1 full checkpoint.",
        },
        {
            "file": "full_freeze.yaml",
            "name": "full_freeze",
            "stage1_enabled": True,
            "ckpt": args.full_ckpt,
            "freeze_enabled": True,
            "freeze_layers": [0, 1, 2, 3, 4],
            "desc": "Initialize detector with stage1 full checkpoint and freeze shallow layers.",
        },
    ]


def main():
    parser = argparse.ArgumentParser(description="Generate stage2 ablation YAML configs from base det_config.yaml")
    parser.add_argument("--base_config", type=str, default="./configs/det_config.yaml", help="Base det config yaml")
    parser.add_argument("--out_dir", type=str, default="./configs/stage2_ablation_generated", help="Output dir")

    parser.add_argument(
        "--project",
        type=str,
        default="runs/glcp_stage2_yolo_det",
        help="train.project used in generated configs",
    )

    parser.add_argument(
        "--full_ckpt",
        type=str,
        default="./runs/glcp_stage1_yolo_det/use_pos_mask/best.pth",
        help="Stage1 full checkpoint path",
    )
    parser.add_argument(
        "--wo_mask_ckpt",
        type=str,
        default="./runs/glcp_stage1_yolo_det/wo_mask/best.pth",
        help="Stage1 wo_mask checkpoint path",
    )
    parser.add_argument(
        "--wo_pos_ckpt",
        type=str,
        default="./runs/glcp_stage1_yolo_det/wo_pos/best.pth",
        help="Stage1 wo_pos checkpoint path",
    )

    args = parser.parse_args()

    base_path = resolve_path(args.base_config)
    out_dir = resolve_path(args.out_dir)

    if not base_path.is_file():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    base_cfg = load_yaml(base_path)
    experiments = build_experiments(args)

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "base_config": str(base_path),
        "output_dir": str(out_dir),
        "experiments": [],
    }

    for exp in experiments:
        cfg = copy.deepcopy(base_cfg)

        cfg.setdefault("train", {})
        cfg.setdefault("stage1_init", {})
        cfg.setdefault("freeze", {})

        cfg["train"]["name"] = exp["name"]
        cfg["train"]["project"] = args.project

        cfg["stage1_init"]["enabled"] = exp["stage1_enabled"]
        cfg["stage1_init"]["ckpt_path"] = exp["ckpt"]

        cfg["freeze"]["enabled"] = exp["freeze_enabled"]
        cfg["freeze"]["layer_indices"] = exp["freeze_layers"]

        out_path = out_dir / exp["file"]
        save_yaml(cfg, out_path)

        manifest["experiments"].append(
            {
                "file": str(out_path),
                "name": exp["name"],
                "desc": exp["desc"],
                "stage1_enabled": exp["stage1_enabled"],
                "ckpt": exp["ckpt"],
                "freeze_enabled": exp["freeze_enabled"],
                "freeze_layers": exp["freeze_layers"],
            }
        )

        print(f"[OK] Generated: {out_path}")

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Stage2 ablation configs generated successfully.")
    print("Base config :", base_path)
    print("Output dir  :", out_dir)
    print("Manifest    :", manifest_path)
    print("=" * 80)


if __name__ == "__main__":
    main()