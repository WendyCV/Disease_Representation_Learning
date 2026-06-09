import argparse
import copy
import json
import os
from pathlib import Path

import yaml


"""
Generate Stage-1 L1/L2/L3 layer-weight ablation configs.

This script is intentionally separated from generate_stage1_ablation_configs.py.
It keeps the full use_pos_mask mechanism fixed:

    use_pos = True
    use_mask = True
    use_raw_spatial = True
    use_aux_embedding = True

and only changes the multi-scale weighting strategy across L1/L2/L3:

    model.pos_init_scales
    loss.raw_spatial.mask_scale_weights
    loss.raw_spatial.consistency_scale_weights
    loss.aux_embedding.scale_weights

The purpose is to test whether Stage-1 should emphasize:
    - old L2/L3 foreground/context prior,
    - current lesion-sensitive balanced prior,
    - stronger L1 lesion-detail prior,
    - stronger L2 lesion-region prior,
    - stronger L3 leaf-context prior.
"""


def load_yaml(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def set_nested(cfg: dict, path: list[str], value):
    cur = cfg
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


def build_use_pos_mask_base(
    base_cfg: dict,
    *,
    name: str,
    notes: str,
    freeze_after_epoch: int,
    pos_init_scales: list[float],
    raw_mask_scale_weights: list[float],
    raw_consistency_scale_weights: list[float],
    aux_embedding_scale_weights: list[float],
):
    cfg = copy.deepcopy(base_cfg)

    cfg.setdefault("experiment", {})
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("ablation", {})

    cfg["experiment"]["name"] = name
    cfg["experiment"]["family"] = "stage1_layer_weights"
    cfg["experiment"]["notes"] = notes

    # Full use_pos_mask mechanism. This script only changes L1/L2/L3 weights.
    cfg["data"]["mask_mode"] = "sam2"
    cfg["ablation"]["use_pos"] = True
    cfg["ablation"]["use_mask"] = True
    cfg["ablation"]["use_raw_spatial"] = True
    cfg["ablation"]["use_aux_embedding"] = True
    cfg["ablation"]["freeze_after_epoch_override"] = int(freeze_after_epoch)

    # L1/L2/L3 scale-weight settings.
    cfg["model"]["pos_init_scales"] = [float(x) for x in pos_init_scales]

    cfg.setdefault("loss", {})
    cfg["loss"].setdefault("raw_spatial", {})
    cfg["loss"].setdefault("aux_embedding", {})

    cfg["loss"]["raw_spatial"]["enabled"] = True
    cfg["loss"]["raw_spatial"]["mask_scale_weights"] = [float(x) for x in raw_mask_scale_weights]
    cfg["loss"]["raw_spatial"]["consistency_scale_weights"] = [float(x) for x in raw_consistency_scale_weights]

    cfg["loss"]["aux_embedding"]["enabled"] = True
    cfg["loss"]["aux_embedding"]["scale_weights"] = [float(x) for x in aux_embedding_scale_weights]

    return cfg


def get_variants():
    """
    Return a compact, hypothesis-driven Stage-1 scale-weight ablation set.

    Naming convention:
      sw = scale weights
      L1 = lesion-detail layer
      L2 = lesion-region layer
      L3 = leaf-context layer
    """
    return [
        {
            "name": "use_pos_mask_sw_old_l2l3",
            "notes": (
                "Old L2/L3-dominant setting. L1 is weak. Used as the historical "
                "reference to test whether insufficient L1 lesion-detail weighting limits transfer."
            ),
            "pos_init_scales": [0.10, 0.50, 1.00],
            "raw_mask_scale_weights": [0.10, 0.45, 0.45],
            "raw_consistency_scale_weights": [0.10, 0.45, 0.45],
            "aux_embedding_scale_weights": [0.10, 0.45, 0.45],
        },
        {
            "name": "use_pos_mask_sw_lesion_sensitive",
            "notes": (
                "Current lesion-sensitive setting. L1 is strengthened but does not dominate; "
                "L2 remains the lesion-region bridge and L3 provides leaf-context prior."
            ),
            "pos_init_scales": [0.30, 0.50, 0.80],
            "raw_mask_scale_weights": [0.25, 0.40, 0.35],
            "raw_consistency_scale_weights": [0.35, 0.35, 0.30],
            "aux_embedding_scale_weights": [0.30, 0.35, 0.35],
        },
        {
            "name": "use_pos_mask_sw_l1_stronger",
            "notes": (
                "Stronger L1 lesion-detail setting. Tests whether increasing fine-grained "
                "texture/color/edge supervision introduces useful lesion cues or noisy texture bias."
            ),
            "pos_init_scales": [0.40, 0.50, 0.70],
            "raw_mask_scale_weights": [0.35, 0.35, 0.30],
            "raw_consistency_scale_weights": [0.40, 0.35, 0.25],
            "aux_embedding_scale_weights": [0.40, 0.30, 0.30],
        },
        {
            "name": "use_pos_mask_sw_l2_centered",
            "notes": (
                "L2-centered lesion-region setting. Tests whether the middle layer is the most "
                "effective bridge between L1 lesion details and L3 leaf context."
            ),
            "pos_init_scales": [0.25, 0.70, 0.70],
            "raw_mask_scale_weights": [0.20, 0.50, 0.30],
            "raw_consistency_scale_weights": [0.30, 0.45, 0.25],
            "aux_embedding_scale_weights": [0.25, 0.45, 0.30],
        },
        {
            "name": "use_pos_mask_sw_l3_context",
            "notes": (
                "L3-context stronger setting. Tests whether stronger high-level leaf-context prior "
                "improves background suppression or makes the detector too conservative downstream."
            ),
            "pos_init_scales": [0.25, 0.45, 0.90],
            "raw_mask_scale_weights": [0.20, 0.35, 0.45],
            "raw_consistency_scale_weights": [0.25, 0.35, 0.40],
            "aux_embedding_scale_weights": [0.25, 0.30, 0.45],
        },
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Generate Stage-1 L1/L2/L3 layer-weight ablation YAML configs."
    )
    parser.add_argument(
        "--base_config",
        type=str,
        required=True,
        help="Base ssl_config.yaml. The script keeps full use_pos_mask switches and only changes L1/L2/L3 weights.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for generated YAML files.",
    )
    parser.add_argument(
        "--freeze_after_epoch",
        type=int,
        default=5,
        help="Snapshot teacher freeze epoch used by all generated configs.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of experiment names to generate.",
    )
    args = parser.parse_args()

    base_cfg = load_yaml(args.base_config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = get_variants()
    if args.only:
        only = set(args.only)
        variants = [v for v in variants if v["name"] in only]
        missing = sorted(only - {v["name"] for v in variants})
        if missing:
            raise ValueError(f"Unknown experiment names in --only: {missing}")

    print("=" * 80)
    print("Generate Stage-1 L1/L2/L3 layer-weight ablation configs")
    print("Base config        :", args.base_config)
    print("Output dir         :", out_dir)
    print("freeze_after_epoch :", args.freeze_after_epoch)
    print("Num variants       :", len(variants))
    print("=" * 80)

    manifest = {
        "base_config": str(args.base_config),
        "output_dir": str(out_dir),
        "family": "stage1_layer_weights",
        "freeze_after_epoch": int(args.freeze_after_epoch),
        "experiments": [],
    }

    for v in variants:
        cfg = build_use_pos_mask_base(
            base_cfg,
            name=v["name"],
            notes=v["notes"],
            freeze_after_epoch=args.freeze_after_epoch,
            pos_init_scales=v["pos_init_scales"],
            raw_mask_scale_weights=v["raw_mask_scale_weights"],
            raw_consistency_scale_weights=v["raw_consistency_scale_weights"],
            aux_embedding_scale_weights=v["aux_embedding_scale_weights"],
        )
        out_path = out_dir / f"{v['name']}.yaml"
        save_yaml(cfg, out_path)
        manifest["experiments"].append(
            {
                "name": v["name"],
                "file": str(out_path),
                "notes": v["notes"],
                "pos_init_scales": v["pos_init_scales"],
                "raw_mask_scale_weights": v["raw_mask_scale_weights"],
                "raw_consistency_scale_weights": v["raw_consistency_scale_weights"],
                "aux_embedding_scale_weights": v["aux_embedding_scale_weights"],
            }
        )
        print(f"[OK] {v['name']:36s} -> {out_path}")

    manifest_path = out_dir / "manifest_stage1_layer_weights.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Done.")
    print("Manifest:", manifest_path)
    print("=" * 80)


if __name__ == "__main__":
    main()
