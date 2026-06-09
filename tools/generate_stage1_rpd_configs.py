import os
import copy
import json
import yaml
import argparse


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _normalize_branch(branch):
    if isinstance(branch, str):
        return branch
    if isinstance(branch, (list, tuple)):
        return list(branch)
    raise TypeError(f"Unsupported teacher branch type: {type(branch)}")


def build_variant(
    base_cfg,
    *,
    name,
    teacher_branch,
    weight,
    scale_weights,
    loss_type,
    start_epoch=6,
    freeze_after_epoch=5,
    teacher_source="online",
    detach_teacher=True,
    notes="",
):
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("experiment", {})
    cfg.setdefault("data", {})
    cfg.setdefault("ablation", {})
    cfg.setdefault("loss", {})

    cfg["experiment"]["name"] = name
    cfg["experiment"]["family"] = "stage1_rpd"
    cfg["experiment"]["notes"] = notes

    # Force the full Stage-1 setting as the RPD base
    cfg["data"]["mask_mode"] = "sam2"
    cfg["data"]["missing_mask_policy"] = "error"

    cfg["ablation"]["use_pos"] = True
    cfg["ablation"]["use_mask"] = True
    cfg["ablation"]["use_raw_spatial"] = True
    cfg["ablation"]["use_aux_embedding"] = True
    cfg["ablation"]["freeze_after_epoch_override"] = int(freeze_after_epoch)

    rpd = cfg["loss"].setdefault("raw_prior_distillation", {})
    rpd["enabled"] = True
    rpd["start_epoch"] = int(start_epoch)
    rpd["teacher_source"] = str(teacher_source)
    rpd["teacher_branch"] = _normalize_branch(teacher_branch)
    rpd["loss_type"] = str(loss_type)
    rpd["weight"] = float(weight)
    rpd["scale_weights"] = [float(x) for x in scale_weights]
    rpd["detach_teacher"] = bool(detach_teacher)
    rpd.setdefault("student_temperature", 0.10)
    rpd.setdefault("teacher_temperature", 0.07)

    return cfg


def build_variant_specs():
    variants = []

    def add(name, teacher_branch, weight, scale_weights, loss_type, notes):
        variants.append(
            dict(
                name=name,
                teacher_branch=teacher_branch,
                weight=weight,
                scale_weights=scale_weights,
                loss_type=loss_type,
                notes=notes,
            )
        )

    # ------------------------------------------------------------------
    # Keep old reference experiments so historical comparison remains possible.
    # ------------------------------------------------------------------
    add(
        "use_pos_mask_rpd_pos_s1_w003",
        "pos_feats", 0.03, [0.30, 0.40, 0.30], "smooth_l1",
        "Legacy reference. Pos-teacher, smooth_l1, balanced scale weights."
    )
    add(
        "use_pos_mask_rpd_pos_s1_w005",
        "pos_feats", 0.05, [0.30, 0.40, 0.30], "smooth_l1",
        "RPD weight sweep (mild). Teacher=pos_feats, loss=smooth_l1, balanced scale weights."
    )
    add(
        "use_pos_mask_rpd_pos_s1_w010",
        "pos_feats", 0.10, [0.30, 0.40, 0.30], "smooth_l1",
        "RPD weight sweep (stronger). Teacher=pos_feats, loss=smooth_l1, balanced scale weights."
    )
    add(
        "use_pos_mask_rpd_pos_s1_w020",
        "pos_feats", 0.20, [0.30, 0.40, 0.30], "smooth_l1",
        "RPD weight sweep (aggressive). Teacher=pos_feats, loss=smooth_l1, balanced scale weights."
    )
    add(
        "use_pos_mask_rpd_pos_s1_bal_w010",
        "pos_feats", 0.10, [0.30, 0.40, 0.30], "smooth_l1",
        "Scale-weight reference. Mild L2 emphasis."
    )
    add(
        "use_pos_mask_rpd_pos_s1_l2main_w010",
        "pos_feats", 0.10, [0.20, 0.50, 0.30], "smooth_l1",
        "Scale-weight ablation. Emphasize L2 lesion-region compression."
    )
    add(
        "use_pos_mask_rpd_pos_s1_l3ctx_w010",
        "pos_feats", 0.10, [0.20, 0.40, 0.40], "smooth_l1",
        "Scale-weight ablation. Stronger L3 context compression."
    )
    add(
        "use_pos_mask_rpd_pos_s1_l1lite_w010",
        "pos_feats", 0.10, [0.15, 0.50, 0.35], "smooth_l1",
        "Scale-weight ablation. Keep L1 very light, let L2 dominate."
    )

    add(
        "use_pos_mask_rpd_hybrid_s1_w003",
        ["local_embs", "pos_feats", "pos_feats"], 0.03, [0.30, 0.40, 0.30], "smooth_l1",
        "Hybrid baseline reference. L1=local_embs, L2/L3=pos_feats."
    )
    add(
        "use_pos_mask_rpd_hybrid_s1_w010",
        ["local_embs", "pos_feats", "pos_feats"], 0.10, [0.30, 0.40, 0.30], "smooth_l1",
        "Hybrid teacher. L1 from local_embs, L2/L3 from pos_feats. Stronger weight."
    )
    add(
        "use_pos_mask_rpd_hybrid_s1_l2main_w010",
        ["local_embs", "pos_feats", "pos_feats"], 0.10, [0.20, 0.50, 0.30], "smooth_l1",
        "Hybrid teacher + L2-main scale weights."
    )

    add(
        "use_pos_mask_rpd_pos_cos_w010",
        "pos_feats", 0.10, [0.20, 0.50, 0.30], "cosine",
        "Loss-type ablation. Cosine alignment with pos teacher, L2-main weights."
    )
    add(
        "use_pos_mask_rpd_hybrid_cos_w010",
        ["local_embs", "pos_feats", "pos_feats"], 0.10, [0.20, 0.50, 0.30], "cosine",
        "Loss-type ablation. Cosine alignment with hybrid teacher, L2-main weights."
    )

    # ------------------------------------------------------------------
    # NEW HYBRID-FOCUSED ABLATIONS
    # All newly added ablations are based on the hybrid baseline:
    #   teacher_branch = [local_embs, pos_feats, pos_feats]
    #   weight         = 0.03
    #   scale_weights  = [0.30, 0.40, 0.30]
    #   loss_type      = smooth_l1
    # ------------------------------------------------------------------

    # A. Weight sweep (same hybrid teacher, same scales, same loss)
    add(
        "use_pos_mask_rpd_hybrid_w003",
        ["local_embs", "pos_feats", "pos_feats"], 0.03, [0.30, 0.40, 0.30], "smooth_l1",
        "Hybrid baseline. Weight sweep anchor."
    )
    add(
        "use_pos_mask_rpd_hybrid_w005",
        ["local_embs", "pos_feats", "pos_feats"], 0.05, [0.30, 0.40, 0.30], "smooth_l1",
        "Hybrid weight sweep. Mildly stronger than baseline."
    )
    add(
        "use_pos_mask_rpd_hybrid_w010",
        ["local_embs", "pos_feats", "pos_feats"], 0.10, [0.30, 0.40, 0.30], "smooth_l1",
        "Hybrid weight sweep. Stronger RPD."
    )
    add(
        "use_pos_mask_rpd_hybrid_w020",
        ["local_embs", "pos_feats", "pos_feats"], 0.20, [0.30, 0.40, 0.30], "smooth_l1",
        "Hybrid weight sweep. Aggressive RPD."
    )

    # B. Scale-weight sweep (same hybrid teacher, weight fixed at baseline 0.03)
    add(
        "use_pos_mask_rpd_hybrid_bal_w003",
        ["local_embs", "pos_feats", "pos_feats"], 0.03, [0.30, 0.40, 0.30], "smooth_l1",
        "Hybrid scale sweep reference. Balanced L1/L2/L3 compression."
    )
    add(
        "use_pos_mask_rpd_hybrid_l2main_w003",
        ["local_embs", "pos_feats", "pos_feats"], 0.03, [0.20, 0.50, 0.30], "smooth_l1",
        "Hybrid scale sweep. Emphasize L2 lesion-region compression."
    )
    add(
        "use_pos_mask_rpd_hybrid_l3ctx_w003",
        ["local_embs", "pos_feats", "pos_feats"], 0.03, [0.20, 0.40, 0.40], "smooth_l1",
        "Hybrid scale sweep. Stronger L3 context compression."
    )
    add(
        "use_pos_mask_rpd_hybrid_l1lite_w003",
        ["local_embs", "pos_feats", "pos_feats"], 0.03, [0.15, 0.50, 0.35], "smooth_l1",
        "Hybrid scale sweep. Keep L1 very light, let L2 dominate."
    )

    # C. Loss-type sweep (same hybrid teacher, baseline weight/scales)
    add(
        "use_pos_mask_rpd_hybrid_s1_w003",
        ["local_embs", "pos_feats", "pos_feats"], 0.03, [0.30, 0.40, 0.30], "smooth_l1",
        "Hybrid loss-type reference (smooth_l1)."
    )
    add(
        "use_pos_mask_rpd_hybrid_cos_w003",
        ["local_embs", "pos_feats", "pos_feats"], 0.03, [0.30, 0.40, 0.30], "cosine",
        "Hybrid loss-type sweep. Cosine alignment under baseline weight/scales."
    )

    # Optional stronger hybrid references retained from previous exploration.
    add(
        "use_pos_mask_rpd_hybrid_s1_w010",
        ["local_embs", "pos_feats", "pos_feats"], 0.10, [0.30, 0.40, 0.30], "smooth_l1",
        "Legacy stronger hybrid reference."
    )
    add(
        "use_pos_mask_rpd_hybrid_s1_l2main_w010",
        ["local_embs", "pos_feats", "pos_feats"], 0.10, [0.20, 0.50, 0.30], "smooth_l1",
        "Legacy stronger hybrid reference with L2-main scales."
    )
    add(
        "use_pos_mask_rpd_hybrid_cos_w010",
        ["local_embs", "pos_feats", "pos_feats"], 0.10, [0.20, 0.50, 0.30], "cosine",
        "Legacy stronger hybrid cosine reference."
    )

    return variants


def resolve_selected_variants(variants, groups, only_names):
    by_name = {v["name"]: v for v in variants}

    group_map = {
        "all": [v["name"] for v in variants],

        # New hybrid-focused groups
        "hybrid_weight": [
            "use_pos_mask_rpd_hybrid_w003",
            "use_pos_mask_rpd_hybrid_w005",
            "use_pos_mask_rpd_hybrid_w010",
            "use_pos_mask_rpd_hybrid_w020",
        ],
        "hybrid_scale_weights": [
            "use_pos_mask_rpd_hybrid_bal_w003",
            "use_pos_mask_rpd_hybrid_l2main_w003",
            "use_pos_mask_rpd_hybrid_l3ctx_w003",
            "use_pos_mask_rpd_hybrid_l1lite_w003",
        ],
        "hybrid_loss_type": [
            "use_pos_mask_rpd_hybrid_s1_w003",
            "use_pos_mask_rpd_hybrid_cos_w003",
        ],
        "hybrid_core": [
            "use_pos_mask_rpd_hybrid_w003",
            "use_pos_mask_rpd_hybrid_w010",
            "use_pos_mask_rpd_hybrid_l2main_w003",
            "use_pos_mask_rpd_hybrid_cos_w003",
        ],

        # Keep legacy groups for comparison/backward compatibility
        "legacy_refs": [
            "use_pos_mask_rpd_pos_s1_w003",
            "use_pos_mask_rpd_hybrid_s1_w003",
            "use_pos_mask_rpd_hybrid_s1_w010",
            "use_pos_mask_rpd_hybrid_s1_l2main_w010",
            "use_pos_mask_rpd_hybrid_cos_w010",
        ],
    }

    selected_names = []
    for g in groups:
        if g not in group_map:
            raise ValueError(f"Unknown group: {g}. Available groups: {sorted(group_map.keys())}")
        selected_names.extend(group_map[g])

    if only_names:
        for name in only_names:
            if name not in by_name:
                raise ValueError(f"Unknown variant name: {name}")
        selected_names.extend(only_names)

    if not selected_names:
        selected_names = group_map["all"]

    dedup = []
    seen = set()
    for name in selected_names:
        if name not in seen:
            dedup.append(name)
            seen.add(name)

    return [by_name[name] for name in dedup], group_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--freeze_after_epoch", type=int, default=5)
    parser.add_argument("--start_epoch", type=int, default=6)
    parser.add_argument(
        "--groups",
        type=str,
        nargs="*",
        default=["all"],
        help="Groups to generate: all / hybrid_core / hybrid_weight / hybrid_scale_weights / hybrid_loss_type / legacy_refs",
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="*",
        default=None,
        help="Generate only the explicitly listed variant names (can be mixed with groups).",
    )
    args = parser.parse_args()

    base_cfg = load_yaml(args.base_config)
    os.makedirs(args.out_dir, exist_ok=True)

    all_variants = build_variant_specs()
    selected_variants, group_map = resolve_selected_variants(all_variants, args.groups, args.only)

    manifest = {
        "base_config": os.path.abspath(args.base_config),
        "out_dir": os.path.abspath(args.out_dir),
        "freeze_after_epoch": int(args.freeze_after_epoch),
        "start_epoch": int(args.start_epoch),
        "requested_groups": args.groups,
        "requested_only": args.only or [],
        "generated": [],
        "available_groups": group_map,
    }

    print("=" * 100)
    print("Generate Stage-1 RPD ablation configs (hybrid-focused)")
    print("Base config        :", args.base_config)
    print("Output dir         :", args.out_dir)
    print("freeze_after_epoch :", args.freeze_after_epoch)
    print("rpd_start_epoch    :", args.start_epoch)
    print("groups             :", args.groups)
    print("only               :", args.only or [])
    print("=" * 100)

    for spec in selected_variants:
        cfg = build_variant(
            base_cfg,
            name=spec["name"],
            teacher_branch=spec["teacher_branch"],
            weight=spec["weight"],
            scale_weights=spec["scale_weights"],
            loss_type=spec["loss_type"],
            start_epoch=args.start_epoch,
            freeze_after_epoch=args.freeze_after_epoch,
            notes=spec["notes"],
        )
        out_path = os.path.join(args.out_dir, f"{spec['name']}.yaml")
        save_yaml(cfg, out_path)

        manifest["generated"].append(
            {
                "name": spec["name"],
                "path": os.path.abspath(out_path),
                "teacher_branch": spec["teacher_branch"],
                "weight": spec["weight"],
                "scale_weights": spec["scale_weights"],
                "loss_type": spec["loss_type"],
                "notes": spec["notes"],
            }
        )
        print(f"[OK] {spec['name']:42s} -> {out_path}")

    save_json(manifest, os.path.join(args.out_dir, "manifest_stage1_rpd.json"))
    print("=" * 100)
    print(f"Generated {len(selected_variants)} config(s).")
    print("Manifest:", os.path.join(args.out_dir, "manifest_stage1_rpd.json"))
    print("=" * 100)


if __name__ == "__main__":
    main()
