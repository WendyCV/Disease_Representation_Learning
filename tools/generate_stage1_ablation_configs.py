import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def _set_common_fields(
    cfg: Dict[str, Any],
    *,
    name: str,
    family: str,
    notes: str,
    mask_mode: str,
    use_pos: bool,
    use_mask: bool,
    use_raw_spatial: bool,
    use_aux_embedding: bool,
    freeze_after_epoch_override: int,
) -> Dict[str, Any]:
    cfg.setdefault("experiment", {})
    cfg.setdefault("data", {})
    cfg.setdefault("ablation", {})

    cfg["experiment"]["name"] = name
    cfg["experiment"]["family"] = family
    cfg["experiment"]["notes"] = notes

    cfg["data"]["mask_mode"] = mask_mode

    # Keep every switch explicit. This avoids relying on base-config defaults
    # and makes each generated YAML easier to read in the paper appendix.
    cfg["ablation"]["use_pos"] = bool(use_pos)
    cfg["ablation"]["use_mask"] = bool(use_mask)
    cfg["ablation"]["use_raw_spatial"] = bool(use_raw_spatial)
    cfg["ablation"]["use_aux_embedding"] = bool(use_aux_embedding)
    cfg["ablation"]["freeze_after_epoch_override"] = int(freeze_after_epoch_override)
    return cfg


def build_variant(base_cfg: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    return _set_common_fields(copy.deepcopy(base_cfg), **kwargs)


def main_ablation_variants(freeze_after_epoch: int) -> List[Dict[str, Any]]:
    """Main paper Stage-1 ablations: mask and position necessity."""
    return [
        dict(
            name="use_pos_mask",
            family="stage1_main_ablation",
            mask_mode="sam2",
            use_pos=True,
            use_mask=True,
            use_raw_spatial=True,
            use_aux_embedding=True,
            freeze_after_epoch_override=freeze_after_epoch,
            notes=(
                "Full Stage-1 method: position-aware modeling + foreground masks + "
                "raw-spatial supervision + auxiliary embedding transfer."
            ),
        ),
        dict(
            name="wo_mask",
            family="stage1_main_ablation",
            mask_mode="none",
            use_pos=True,
            use_mask=False,
            use_raw_spatial=False,
            use_aux_embedding=True,
            freeze_after_epoch_override=freeze_after_epoch,
            notes=(
                "Remove foreground masks while keeping the position-aware module. "
                "Raw-spatial supervision is disabled because it requires masks."
            ),
        ),
        dict(
            name="wo_pos",
            family="stage1_main_ablation",
            mask_mode="sam2",
            use_pos=False,
            use_mask=True,
            use_raw_spatial=True,
            use_aux_embedding=True,
            freeze_after_epoch_override=freeze_after_epoch,
            notes=(
                "Remove the position-aware module while keeping mask-guided raw-spatial "
                "supervision and auxiliary embedding transfer."
            ),
        ),
        dict(
            name="wo_pos_wo_mask",
            family="stage1_main_ablation",
            mask_mode="none",
            use_pos=False,
            use_mask=False,
            use_raw_spatial=False,
            use_aux_embedding=True,
            freeze_after_epoch_override=freeze_after_epoch,
            notes=(
                "Remove both position-aware modeling and foreground masks. "
                "Included as a lowest-control appendix baseline."
            ),
        ),
    ]


def mechanism_variants(freeze_after_epoch: int, include_full_alias: bool = True) -> List[Dict[str, Any]]:
    """Appendix mechanism ablations for raw-spatial and auxiliary embedding routes."""
    variants: List[Dict[str, Any]] = []

    if include_full_alias:
        variants.append(
            dict(
                name="use_pos_mask_full",
                family="stage1_mechanism_ablation",
                mask_mode="sam2",
                use_pos=True,
                use_mask=True,
                use_raw_spatial=True,
                use_aux_embedding=True,
                freeze_after_epoch_override=freeze_after_epoch,
                notes=(
                    "Alias of use_pos_mask grouped with mechanism ablations. It is kept for "
                    "backward compatibility with previous scripts and appendix naming."
                ),
            )
        )

    variants.extend(
        [
            dict(
                name="use_pos_mask_no_raw_spatial",
                family="stage1_mechanism_ablation",
                mask_mode="sam2",
                use_pos=True,
                use_mask=True,
                use_raw_spatial=False,
                use_aux_embedding=True,
                freeze_after_epoch_override=freeze_after_epoch,
                notes=(
                    "Remove direct raw-backbone spatial supervision to test whether the "
                    "foreground prior is compressed into the raw backbone."
                ),
            ),
            dict(
                name="use_pos_mask_no_aux_embedding",
                family="stage1_mechanism_ablation",
                mask_mode="sam2",
                use_pos=True,
                use_mask=True,
                use_raw_spatial=True,
                use_aux_embedding=False,
                freeze_after_epoch_override=freeze_after_epoch,
                notes=(
                    "Remove auxiliary embedding transfer to test the role of local/global "
                    "embedding transfer from the snapshot teacher."
                ),
            ),
            dict(
                name="use_pos_mask_no_raw_no_aux",
                family="stage1_mechanism_ablation",
                mask_mode="sam2",
                use_pos=True,
                use_mask=True,
                use_raw_spatial=False,
                use_aux_embedding=False,
                freeze_after_epoch_override=freeze_after_epoch,
                notes=(
                    "Remove both raw-spatial supervision and auxiliary embedding transfer "
                    "while keeping position+mask; appendix stress-test."
                ),
            ),
        ]
    )
    return variants


def freeze_variants(freeze_epochs: List[int]) -> List[Dict[str, Any]]:
    """Appendix-only snapshot-teacher freeze timing sensitivity."""
    variants: List[Dict[str, Any]] = []
    for epoch in freeze_epochs:
        variants.append(
            dict(
                name=f"use_pos_mask_f{int(epoch):02d}",
                family="stage1_freeze_after_epoch",
                mask_mode="sam2",
                use_pos=True,
                use_mask=True,
                use_raw_spatial=True,
                use_aux_embedding=True,
                freeze_after_epoch_override=int(epoch),
                notes=(
                    f"Appendix-only sensitivity check. Freeze snapshot teacher after epoch {int(epoch)}. "
                    "Not part of the main method comparison."
                ),
            )
        )
    return variants


def build_all_variants(args: argparse.Namespace) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []
    requested = set(args.families)

    if "main" in requested or "all" in requested:
        variants.extend(main_ablation_variants(args.freeze_after_epoch))
    if "mechanism" in requested or "all" in requested:
        variants.extend(
            mechanism_variants(
                freeze_after_epoch=args.freeze_after_epoch,
                include_full_alias=not args.no_full_alias,
            )
        )
    if "freeze" in requested or "all" in requested:
        variants.extend(freeze_variants(args.freeze_epochs))

    # Deduplicate by experiment name while preserving order.
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for variant in variants:
        name = variant["name"]
        if name in seen:
            continue
        seen.add(name)
        deduped.append(variant)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate all Stage-1 ablation YAML configs from one script. This file replaces "
            "generate_stage1_mechanism_ablation_configs.py and "
            "generate_stage1_freeze_ablation_configs.py."
        )
    )
    parser.add_argument("--base_config", type=str, required=True, help="Base SSL config, e.g. ./configs/ssl_config.yaml")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for generated Stage-1 YAMLs")
    parser.add_argument("--freeze_after_epoch", type=int, default=5, help="Default snapshot-teacher freeze epoch for non-freeze-sweep variants")
    parser.add_argument("--freeze_epochs", type=int, nargs="+", default=[3, 5, 8, 10], help="Freeze-epoch sweep values for appendix configs")
    parser.add_argument(
        "--families",
        nargs="+",
        default=["all"],
        choices=["all", "main", "mechanism", "freeze"],
        help="Which groups to generate. Default: all.",
    )
    parser.add_argument(
        "--no_full_alias",
        action="store_true",
        help="Do not generate use_pos_mask_full alias in mechanism ablations.",
    )
    args = parser.parse_args()

    base_cfg = load_yaml(args.base_config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = build_all_variants(args)
    manifest = {
        "base_config": str(Path(args.base_config).resolve()),
        "output_dir": str(out_dir.resolve()),
        "freeze_after_epoch": args.freeze_after_epoch,
        "freeze_epochs": args.freeze_epochs,
        "families": args.families,
        "experiments": [],
    }

    print("=" * 80)
    print("Generate Stage-1 ablation configs")
    print("Base config :", args.base_config)
    print("Output dir  :", args.out_dir)
    print("Families    :", ", ".join(args.families))
    print("Default f   :", args.freeze_after_epoch)
    print("Freeze list :", args.freeze_epochs)
    print("=" * 80)

    for variant in variants:
        cfg = build_variant(base_cfg, **variant)
        out_path = out_dir / f"{variant['name']}.yaml"
        save_yaml(cfg, out_path)
        manifest["experiments"].append(
            {
                "name": variant["name"],
                "family": variant["family"],
                "file": str(out_path),
                "mask_mode": variant["mask_mode"],
                "use_pos": variant["use_pos"],
                "use_mask": variant["use_mask"],
                "use_raw_spatial": variant["use_raw_spatial"],
                "use_aux_embedding": variant["use_aux_embedding"],
                "freeze_after_epoch_override": variant["freeze_after_epoch_override"],
                "notes": variant["notes"],
            }
        )
        print(f"[OK] {variant['name']:<36s} ({variant['family']}) -> {out_path}")

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"Done. Generated {len(variants)} config(s).")
    print("Manifest:", manifest_path)
    print("=" * 80)


if __name__ == "__main__":
    main()
