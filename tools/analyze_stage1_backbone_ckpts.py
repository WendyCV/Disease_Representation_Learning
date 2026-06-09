import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# =========================================================
# 关键修复：将工程根目录加入 sys.path
# =========================================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.checkpoint import (
    load_torch_checkpoint,
    extract_stage1_state_dict,
    extract_stage1_backbone_state_dict,
    filter_stage1_trained_backbone_state_dict,
)


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def classify_key_type(key: str) -> str:
    """
    将 state_dict key 分类，便于判断差异到底发生在 learnable params 还是 BN buffers
    """
    if key.endswith("conv.weight"):
        return "conv.weight"
    elif key.endswith("bn.weight"):
        return "bn.weight"
    elif key.endswith("bn.bias"):
        return "bn.bias"
    elif key.endswith("bn.running_mean"):
        return "bn.running_mean"
    elif key.endswith("bn.running_var"):
        return "bn.running_var"
    elif key.endswith("bn.num_batches_tracked"):
        return "bn.num_batches_tracked"
    elif key.endswith(".weight"):
        return "other.weight"
    elif key.endswith(".bias"):
        return "other.bias"
    else:
        return "other"


def is_learnable_key(key: str) -> bool:
    """
    定义“真正可学习参数”
    """
    key_type = classify_key_type(key)
    return key_type in {"conv.weight", "bn.weight", "bn.bias", "other.weight", "other.bias"}


def is_buffer_key(key: str) -> bool:
    """
    定义 BN buffer / 非梯度更新项
    """
    key_type = classify_key_type(key)
    return key_type in {"bn.running_mean", "bn.running_var", "bn.num_batches_tracked"}


def tensor_diff_stats(t1: torch.Tensor, t2: torch.Tensor):
    """
    比较两个 tensor 的数值差异
    """
    if t1.shape != t2.shape:
        return {
            "same_shape": False,
            "exact_equal": False,
            "max_abs_diff": None,
            "mean_abs_diff": None,
        }

    diff = (t1.float() - t2.float()).abs()
    max_abs_diff = float(diff.max().item()) if diff.numel() > 0 else 0.0
    mean_abs_diff = float(diff.mean().item()) if diff.numel() > 0 else 0.0
    exact_equal = bool(torch.equal(t1, t2))

    return {
        "same_shape": True,
        "exact_equal": exact_equal,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
    }


def load_stage1_trained_backbone_subset(
    ckpt_path: Path,
    trained_layer_min: int,
    trained_layer_max: int,
) -> Dict[str, torch.Tensor]:
    ckpt = load_torch_checkpoint(str(ckpt_path), map_location="cpu")
    stage1_state = extract_stage1_state_dict(ckpt)
    raw_backbone = extract_stage1_backbone_state_dict(stage1_state)
    filtered_state, _ = filter_stage1_trained_backbone_state_dict(
        raw_backbone,
        trained_layer_min=trained_layer_min,
        trained_layer_max=trained_layer_max,
    )
    return filtered_state


def empty_group_stats():
    return {
        "num_common": 0,
        "num_exact_equal": 0,
        "num_different": 0,
        "num_shape_mismatch": 0,
        "max_abs_diff": 0.0,
        "mean_abs_diff": 0.0,
        "different_keys": [],
    }


def finalize_group_stats(stats_dict: Dict):
    """
    将累积中的 mean_abs_diff_list 收敛成最终 mean_abs_diff
    """
    out = {}
    for group_name, st in stats_dict.items():
        mean_list = st.pop("mean_abs_diff_list")
        st["mean_abs_diff"] = float(sum(mean_list) / len(mean_list)) if mean_list else 0.0
        out[group_name] = st
    return out


def init_group_trackers():
    return {
        "all": {
            "num_common": 0,
            "num_exact_equal": 0,
            "num_different": 0,
            "num_shape_mismatch": 0,
            "max_abs_diff": 0.0,
            "mean_abs_diff_list": [],
            "different_keys": [],
        },
        "learnable": {
            "num_common": 0,
            "num_exact_equal": 0,
            "num_different": 0,
            "num_shape_mismatch": 0,
            "max_abs_diff": 0.0,
            "mean_abs_diff_list": [],
            "different_keys": [],
        },
        "buffer": {
            "num_common": 0,
            "num_exact_equal": 0,
            "num_different": 0,
            "num_shape_mismatch": 0,
            "max_abs_diff": 0.0,
            "mean_abs_diff_list": [],
            "different_keys": [],
        },
    }


def update_group_stats(group, key, stats, sd_a=None, sd_b=None):
    group["num_common"] += 1
    if not stats["same_shape"]:
        group["num_shape_mismatch"] += 1
        group["num_different"] += 1
        group["different_keys"].append({
            "key": key,
            "type": classify_key_type(key),
            "reason": "shape_mismatch",
            "shape_a": list(sd_a[key].shape),
            "shape_b": list(sd_b[key].shape),
        })
        return

    if stats["exact_equal"]:
        group["num_exact_equal"] += 1
    else:
        group["num_different"] += 1
        group["max_abs_diff"] = max(group["max_abs_diff"], stats["max_abs_diff"])
        group["mean_abs_diff_list"].append(stats["mean_abs_diff"])
        group["different_keys"].append({
            "key": key,
            "type": classify_key_type(key),
            "reason": "value_diff",
            "max_abs_diff": stats["max_abs_diff"],
            "mean_abs_diff": stats["mean_abs_diff"],
        })


def summarize_key_type_counts(keys: List[str]) -> Dict[str, int]:
    counts = {}
    for k in keys:
        t = classify_key_type(k)
        counts[t] = counts.get(t, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: x[0]))


def compare_two_state_dicts(
    name_a: str,
    sd_a: Dict[str, torch.Tensor],
    name_b: str,
    sd_b: Dict[str, torch.Tensor],
) -> Dict:
    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())

    common_keys = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    groups = init_group_trackers()

    for k in common_keys:
        stats = tensor_diff_stats(sd_a[k], sd_b[k])

        # all
        update_group_stats(groups["all"], k, stats, sd_a, sd_b)

        # learnable or buffer
        if is_learnable_key(k):
            update_group_stats(groups["learnable"], k, stats, sd_a, sd_b)
        elif is_buffer_key(k):
            update_group_stats(groups["buffer"], k, stats, sd_a, sd_b)

    groups = finalize_group_stats(groups)

    # 各类型计数
    common_key_type_counts = summarize_key_type_counts(common_keys)
    diff_key_type_counts = summarize_key_type_counts([x["key"] for x in groups["all"]["different_keys"]])
    learnable_diff_key_type_counts = summarize_key_type_counts([x["key"] for x in groups["learnable"]["different_keys"]])
    buffer_diff_key_type_counts = summarize_key_type_counts([x["key"] for x in groups["buffer"]["different_keys"]])

    # Top-N 差异最大的 learnable 参数
    top_learnable_diffs = sorted(
        [x for x in groups["learnable"]["different_keys"] if x["reason"] == "value_diff"],
        key=lambda x: x["max_abs_diff"],
        reverse=True,
    )

    top_buffer_diffs = sorted(
        [x for x in groups["buffer"]["different_keys"] if x["reason"] == "value_diff"],
        key=lambda x: x["max_abs_diff"],
        reverse=True,
    )

    return {
        "pair": [name_a, name_b],
        "num_keys_a": len(keys_a),
        "num_keys_b": len(keys_b),
        "num_common_keys": len(common_keys),
        "num_only_a": len(only_a),
        "num_only_b": len(only_b),

        "only_a_keys": only_a,
        "only_b_keys": only_b,

        "common_key_type_counts": common_key_type_counts,
        "different_key_type_counts": diff_key_type_counts,
        "learnable_different_key_type_counts": learnable_diff_key_type_counts,
        "buffer_different_key_type_counts": buffer_diff_key_type_counts,

        "all_stats": groups["all"],
        "learnable_stats": groups["learnable"],
        "buffer_stats": groups["buffer"],

        "top_learnable_diffs": top_learnable_diffs[:50],
        "top_buffer_diffs": top_buffer_diffs[:50],
    }


def print_group_stats(title: str, stats: Dict):
    print(f"  [{title}]")
    print(f"    common            : {stats['num_common']}")
    print(f"    exact equal       : {stats['num_exact_equal']}")
    print(f"    different         : {stats['num_different']}")
    print(f"    shape mismatch    : {stats['num_shape_mismatch']}")
    print(f"    max abs diff      : {stats['max_abs_diff']}")
    print(f"    mean abs diff     : {stats['mean_abs_diff']}")


def print_pair_summary(result: Dict):
    pair = result["pair"]
    print("=" * 110)
    print(f"[Compare] {pair[0]}  <->  {pair[1]}")
    print(f"  num_keys_a         : {result['num_keys_a']}")
    print(f"  num_keys_b         : {result['num_keys_b']}")
    print(f"  common keys        : {result['num_common_keys']}")
    print(f"  only in A          : {result['num_only_a']}")
    print(f"  only in B          : {result['num_only_b']}")
    print("-" * 110)

    print_group_stats("ALL", result["all_stats"])
    print_group_stats("LEARNABLE", result["learnable_stats"])
    print_group_stats("BUFFER", result["buffer_stats"])
    print("-" * 110)

    print("  Common key type counts:")
    for k, v in result["common_key_type_counts"].items():
        print(f"    {k:24s}: {v}")

    print("-" * 110)
    print("  Different key type counts (ALL):")
    if result["different_key_type_counts"]:
        for k, v in result["different_key_type_counts"].items():
            print(f"    {k:24s}: {v}")
    else:
        print("    None")

    print("-" * 110)
    print("  Different key type counts (LEARNABLE):")
    if result["learnable_different_key_type_counts"]:
        for k, v in result["learnable_different_key_type_counts"].items():
            print(f"    {k:24s}: {v}")
    else:
        print("    None")

    print("-" * 110)
    print("  Different key type counts (BUFFER):")
    if result["buffer_different_key_type_counts"]:
        for k, v in result["buffer_different_key_type_counts"].items():
            print(f"    {k:24s}: {v}")
    else:
        print("    None")

    print("-" * 110)
    print("  Top learnable diffs (up to 20):")
    if result["top_learnable_diffs"]:
        for item in result["top_learnable_diffs"][:20]:
            print(
                f"    - {item['key']} [{item['type']}] "
                f"(max_abs_diff={item['max_abs_diff']:.8f}, mean_abs_diff={item['mean_abs_diff']:.8f})"
            )
    else:
        print("    None")

    print("-" * 110)
    print("  Top buffer diffs (up to 20):")
    if result["top_buffer_diffs"]:
        for item in result["top_buffer_diffs"][:20]:
            print(
                f"    - {item['key']} [{item['type']}] "
                f"(max_abs_diff={item['max_abs_diff']:.8f}, mean_abs_diff={item['mean_abs_diff']:.8f})"
            )
    else:
        print("    None")

    print("=" * 110)


def main():
    parser = argparse.ArgumentParser(
        description="Compare stage1 checkpoints on trained backbone(model.0~8) and show learnable/buffer metrics separately"
    )
    parser.add_argument(
        "--full_ckpt",
        type=str,
        default="./runs/glcp_stage1_yolo_det/use_pos_mask/best.pth",
        help="Path to stage1 full checkpoint",
    )
    parser.add_argument(
        "--wo_mask_ckpt",
        type=str,
        default="./runs/glcp_stage1_yolo_det/wo_mask/best.pth",
        help="Path to stage1 wo_mask checkpoint",
    )
    parser.add_argument(
        "--wo_pos_ckpt",
        type=str,
        default="./runs/glcp_stage1_yolo_det/wo_pos/best.pth",
        help="Path to stage1 wo_pos checkpoint",
    )
    parser.add_argument(
        "--trained_layer_min",
        type=int,
        default=0,
        help="Minimum trained layer index in stage1",
    )
    parser.add_argument(
        "--trained_layer_max",
        type=int,
        default=8,
        help="Maximum trained layer index in stage1",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default="./runs/stage1_backbone_compare_report.json",
        help="Path to save comparison report as JSON",
    )

    args = parser.parse_args()

    ckpt_map = {
        "full": resolve_path(args.full_ckpt),
        "wo_mask": resolve_path(args.wo_mask_ckpt),
        "wo_pos": resolve_path(args.wo_pos_ckpt),
    }

    print("=" * 110)
    print("Stage1 backbone checkpoint comparison (trained layers only)")
    print(f"trained layer range: [{args.trained_layer_min}, {args.trained_layer_max}]")
    for name, path in ckpt_map.items():
        print(f"{name:10s}: {path}")
    print("=" * 110)

    state_dicts = {}
    key_counts = {}

    for name, path in ckpt_map.items():
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        sd = load_stage1_trained_backbone_subset(
            ckpt_path=path,
            trained_layer_min=args.trained_layer_min,
            trained_layer_max=args.trained_layer_max,
        )
        state_dicts[name] = sd
        key_counts[name] = len(sd)

    print("[Key counts]")
    for name, count in key_counts.items():
        print(f"  {name:10s}: {count}")
    print("=" * 110)

    pair_results = []
    for name_a, name_b in combinations(state_dicts.keys(), 2):
        result = compare_two_state_dicts(name_a, state_dicts[name_a], name_b, state_dicts[name_b])
        pair_results.append(result)
        print_pair_summary(result)

    report = {
        "trained_layer_range": [args.trained_layer_min, args.trained_layer_max],
        "checkpoints": {k: str(v) for k, v in ckpt_map.items()},
        "key_counts": key_counts,
        "pair_results": pair_results,
    }

    save_json_path = resolve_path(args.save_json)
    save_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[OK] Comparison report saved to: {save_json_path}")


if __name__ == "__main__":
    main()