import json
import os
from typing import Dict, List, Tuple

import torch


STAGE1_BACKBONE_PREFIXES = [
    "online.backbone.",
    "module.online.backbone.",
]


def save_checkpoint(state, save_dir, filename="last.pth"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    return path


def save_json(data: Dict, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_transfer_report(report: Dict, save_path: str):
    save_json(report, save_path)


def load_torch_checkpoint(ckpt_path: str, map_location: str = "cpu"):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location=map_location)


def extract_stage1_state_dict(ckpt) -> Dict[str, torch.Tensor]:
    """
    兼容两种格式：
    1) {"model": state_dict, ...}
    2) 直接保存的纯 state_dict
    """
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError("Unsupported checkpoint format for stage1 checkpoint.")


def extract_stage1_backbone_state_dict(stage1_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    从 stage1 权重中提取 online.backbone 参数。

    原始 key 例如：
      online.backbone.model.0.conv.weight
    转成可直接对齐 detector 的 key：
      model.0.conv.weight
    """
    backbone_state = {}

    for k, v in stage1_state_dict.items():
        matched_prefix = None
        for prefix in STAGE1_BACKBONE_PREFIXES:
            if k.startswith(prefix):
                matched_prefix = prefix
                break
        if matched_prefix is None:
            continue

        new_key = k[len(matched_prefix):]
        backbone_state[new_key] = v

    if not backbone_state:
        raise KeyError(
            "No stage1 online backbone parameters were found. "
            "Expected keys like 'online.backbone.model.*'."
        )

    return backbone_state


def _parse_model_layer_index(key: str):
    """
    解析 state_dict key 中最外层 model.X 的层号
    例如:
      model.8.m.0.cv1.conv.weight -> 8
      model.22.cv2.0.0.conv.weight -> 22
    """
    parts = key.split(".")
    if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
        return int(parts[1])
    return None


def filter_stage1_trained_backbone_state_dict(
    state_dict: Dict[str, torch.Tensor],
    trained_layer_min: int = 0,
    trained_layer_max: int = 8,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """
    只保留 stage1 实际参与训练的 backbone 层。
    根据你当前 stage1 的 forward 逻辑：
      layer_indices = [4, 6, 8]
      if idx >= max(layer_indices): break
    所以实际参与前向/反向的是 model.0 ~ model.8

    返回：
      filtered_state, excluded_keys
    """
    filtered_state = {}
    excluded_keys = []

    for k, v in state_dict.items():
        layer_idx = _parse_model_layer_index(k)

        # 保守策略：如果解析不出层号，默认排除并记录
        if layer_idx is None:
            excluded_keys.append(k)
            continue

        if trained_layer_min <= layer_idx <= trained_layer_max:
            filtered_state[k] = v
        else:
            excluded_keys.append(k)

    return filtered_state, excluded_keys


def transfer_stage1_backbone_to_yolo(
    det_model,
    stage1_ckpt_path: str,
    map_location: str = "cpu",
    verbose: bool = True,
    print_excluded_keys: bool = False,
    trained_layer_min: int = 0,
    trained_layer_max: int = 8,
) -> Tuple[object, Dict]:
    """
    将 stage1 checkpoint 中 online.backbone 的参数迁移到 YOLO detector，
    但只迁移 stage1 实际训练过的层（默认 model.0 ~ model.8）。

    返回：
      det_model, report
    """
    ckpt = load_torch_checkpoint(stage1_ckpt_path, map_location=map_location)
    stage1_state = extract_stage1_state_dict(ckpt)

    # 先提取 online.backbone.model.*
    raw_transfer_state = extract_stage1_backbone_state_dict(stage1_state)

    # 再只保留 stage1 实际训练过的层
    filtered_transfer_state, excluded_keys = filter_stage1_trained_backbone_state_dict(
        raw_transfer_state,
        trained_layer_min=trained_layer_min,
        trained_layer_max=trained_layer_max,
    )

    det_state = det_model.state_dict()
    merged_state = det_state.copy()

    loaded_keys: List[str] = []
    skipped_shape: List[Dict] = []
    missing_in_det: List[str] = []

    for k, v in filtered_transfer_state.items():
        if k not in det_state:
            missing_in_det.append(k)
            continue

        if tuple(det_state[k].shape) != tuple(v.shape):
            skipped_shape.append({
                "key": k,
                "stage1_shape": list(v.shape),
                "det_shape": list(det_state[k].shape),
            })
            continue

        merged_state[k] = v
        loaded_keys.append(k)

    det_model.load_state_dict(merged_state, strict=False)

    report = {
        "stage1_ckpt_path": stage1_ckpt_path,
        "trained_layer_range": [trained_layer_min, trained_layer_max],
        "raw_transfer_count": len(raw_transfer_state),

        "excluded_non_trained_count": len(excluded_keys),
        "excluded_non_trained_keys": excluded_keys,

        "filtered_transfer_count": len(filtered_transfer_state),

        "loaded_count": len(loaded_keys),
        "loaded_keys": loaded_keys,

        "missing_in_det_count": len(missing_in_det),
        "missing_in_det": missing_in_det,

        "skipped_shape_count": len(skipped_shape),
        "skipped_shape": skipped_shape,
    }

    if verbose:
        print("=" * 80)
        print("[Stage2] Stage1 trained-backbone transfer summary")
        print(f"checkpoint              : {stage1_ckpt_path}")
        print(f"trained layer range     : [{trained_layer_min}, {trained_layer_max}]")
        print(f"raw candidate params    : {report['raw_transfer_count']}")
        print(f"excluded non-trained    : {report['excluded_non_trained_count']}")
        print(f"filtered params         : {report['filtered_transfer_count']}")
        print(f"loaded params           : {report['loaded_count']}")
        print(f"missing in det          : {report['missing_in_det_count']}")
        print(f"shape mismatched        : {report['skipped_shape_count']}")
        print("=" * 80)

        if excluded_keys and print_excluded_keys:
            print("[Stage2] Excluded non-trained parameter names:")
            for k in excluded_keys:
                print(f"  - {k}")
            print("=" * 80)

    return det_model, report