import os
import sys
import yaml
import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import torch
import numpy as np
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.stage1_ssl_model import Stage1SslModel
from utils.config_utils import validate_stage1_config
from datasets.augmentations import build_base_transform

import warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# =========================================================
# Basic IO
# =========================================================
def resolve_path(path_str: str | Path, base_dir: Optional[Path] = None) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    if base_dir is not None:
        return (base_dir / p).resolve()
    return (Path(PROJECT_ROOT) / p).resolve()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_images(root_dir):
    root = Path(root_dir)
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS:
            files.append(str(p))
    return files


def evenly_sample(items, num_samples):
    """Evenly sample images; num_samples=-1 means use all images."""
    items = list(items)
    if num_samples is None or int(num_samples) < 0:
        return items
    if int(num_samples) == 0:
        return []
    if len(items) <= int(num_samples):
        return items
    idxs = np.linspace(0, len(items) - 1, int(num_samples)).round().astype(int)
    return [items[i] for i in idxs]


def parse_dataset_splits(dataset_split) -> List[str]:
    """Accept `val test`, `val,test`, or a single split string."""
    if dataset_split is None:
        return ["val"]
    if isinstance(dataset_split, str):
        raw_items = [dataset_split]
    else:
        raw_items = list(dataset_split)

    splits: List[str] = []
    for item in raw_items:
        for part in str(item).split(","):
            part = part.strip()
            if part and part not in splits:
                splits.append(part)
    return splits or ["val"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_rgb(path, rgb):
    Image.fromarray(rgb).save(path)


def save_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================================================
# Preprocess
# =========================================================
def build_infer_preprocess(image_size):
    base_transform = build_base_transform(image_size=image_size)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return base_transform, mean, std


def _make_all_one_mask_pil(img_pil):
    width, height = img_pil.size
    mask_np = np.ones((height, width), dtype=np.uint8) * 255
    return Image.fromarray(mask_np, mode="L")


def _resolve_mask_path(image_path, image_root, mask_root, mask_suffix):
    rel_path = os.path.relpath(image_path, image_root)
    rel_path = os.path.splitext(rel_path)[0] + mask_suffix
    return os.path.join(mask_root, rel_path)


def _load_mask_pil_for_visualize(image_path, img_pil, cfg):
    if not cfg["runtime"]["use_mask"]:
        return _make_all_one_mask_pil(img_pil)

    mask_root = cfg["data"].get("mask_root_dir", None)
    if mask_root is None:
        return _make_all_one_mask_pil(img_pil)

    mask_suffix = cfg["data"].get("mask_suffix", ".png")
    mask_path = _resolve_mask_path(image_path, cfg["data"]["train_dir"], mask_root, mask_suffix)
    if not os.path.exists(mask_path):
        return _make_all_one_mask_pil(img_pil)

    mask_pil = Image.open(mask_path).convert("L")
    if mask_pil.size != img_pil.size:
        mask_pil = mask_pil.resize(img_pil.size, Image.NEAREST)

    threshold = cfg["data"].get("external_mask_threshold", 127)
    mask_np = np.array(mask_pil, dtype=np.uint8)
    mask_np = (mask_np > threshold).astype(np.uint8) * 255
    return Image.fromarray(mask_np, mode="L")


def compute_unpad_crop_box(
    original_w: int,
    original_h: int,
    image_size: int,
) -> Tuple[int, int, int, int]:
    """
    Compute the valid image region inside a square image produced by
    ResizeAndPadToSquare(long_size=image_size).

    The saved visualization after cropping will have:
      long side = image_size
      short side = resized according to the original aspect ratio

    Returns:
        (x1, y1, x2, y2) in square model-input coordinates.
    """
    if original_w <= 0 or original_h <= 0:
        return 0, 0, int(image_size), int(image_size)

    scale = float(image_size) / float(max(original_w, original_h))
    resized_w = int(round(original_w * scale))
    resized_h = int(round(original_h * scale))

    resized_w = max(1, min(int(image_size), resized_w))
    resized_h = max(1, min(int(image_size), resized_h))

    pad_x = int(image_size) - resized_w
    pad_y = int(image_size) - resized_h

    x1 = pad_x // 2
    y1 = pad_y // 2
    x2 = x1 + resized_w
    y2 = y1 + resized_h

    return int(x1), int(y1), int(x2), int(y2)


def crop_rgb_by_box(
    rgb: np.ndarray,
    crop_box: Optional[Tuple[int, int, int, int]],
) -> np.ndarray:
    """
    Crop RGB image by crop_box. If crop_box is None, return input unchanged.
    """
    if crop_box is None:
        return rgb

    x1, y1, x2, y2 = crop_box
    h, w = rgb.shape[:2]

    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(x1 + 1, min(w, int(x2)))
    y2 = max(y1 + 1, min(h, int(y2)))

    return rgb[y1:y2, x1:x2, :].copy()


def crop_gray_by_box(
    gray: np.ndarray,
    crop_box: Optional[Tuple[int, int, int, int]],
) -> np.ndarray:
    """
    Crop single-channel image by crop_box. If crop_box is None, return input unchanged.
    """
    if crop_box is None:
        return gray

    x1, y1, x2, y2 = crop_box
    h, w = gray.shape[:2]

    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(x1 + 1, min(w, int(x2)))
    y2 = max(y1 + 1, min(h, int(y2)))

    return gray[y1:y2, x1:x2].copy()


def load_image_and_mask_for_model(image_path, cfg, crop_padding_visuals: bool = False):
    """
    Load image and mask for the model.

    Important:
      - x and m are always generated from the normal model transform.
        If the transform pads to square, model input remains padded square.
      - vis_rgb_square is the transformed square RGB image.
      - vis_rgb is only for saving visualization. If crop_padding_visuals=True,
        it is cropped to remove black padding.
    """
    image_size = cfg["data"]["image_size"]
    base_transform, mean, std = build_infer_preprocess(image_size)

    img = Image.open(image_path).convert("RGB")
    original_w, original_h = img.size

    mask_pil = _load_mask_pil_for_visualize(image_path, img, cfg)

    rgba = img.copy()
    rgba.putalpha(mask_pil)
    rgba_tensor = base_transform(rgba)

    rgb_tensor = rgba_tensor[:3]
    mask_tensor = (rgba_tensor[3:].float() > 0.5).float()

    vis_rgb_square = (
        rgb_tensor.permute(1, 2, 0).numpy() * 255.0
    ).clip(0, 255).astype(np.uint8)

    crop_box = None
    if crop_padding_visuals:
        crop_box = compute_unpad_crop_box(
            original_w=original_w,
            original_h=original_h,
            image_size=image_size,
        )

    vis_rgb = crop_rgb_by_box(vis_rgb_square, crop_box)

    x = ((rgb_tensor - mean) / std).unsqueeze(0)
    m = mask_tensor.unsqueeze(0)

    vis_info = {
        "crop_padding_visuals": bool(crop_padding_visuals),
        "original_size": {
            "width": int(original_w),
            "height": int(original_h),
        },
        "model_input_size": {
            "width": int(vis_rgb_square.shape[1]),
            "height": int(vis_rgb_square.shape[0]),
        },
        "saved_visual_size": {
            "width": int(vis_rgb.shape[1]),
            "height": int(vis_rgb.shape[0]),
        },
        "crop_box_in_model_input": {
            "x1": int(crop_box[0]) if crop_box else 0,
            "y1": int(crop_box[1]) if crop_box else 0,
            "x2": int(crop_box[2]) if crop_box else int(vis_rgb_square.shape[1]),
            "y2": int(crop_box[3]) if crop_box else int(vis_rgb_square.shape[0]),
        },
    }

    return vis_rgb_square, vis_rgb, crop_box, x, m, vis_info


# =========================================================
# Dataset helpers
# =========================================================
def load_data_yaml(data_yaml_path: Path) -> dict:
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_dataset_root_from_data_yaml(data_yaml_path: Path) -> Path:
    data_yaml = load_data_yaml(data_yaml_path)
    if data_yaml.get("path", None):
        return resolve_path(str(data_yaml["path"]), data_yaml_path.parent)
    return data_yaml_path.parent


def resolve_dataset_split_dir(data_yaml_path: Path, split: str = "val") -> Path:
    data_yaml = load_data_yaml(data_yaml_path)
    split_key = split if split in data_yaml else None
    if split_key is None:
        for candidate in ("val", "test", "train"):
            if candidate in data_yaml:
                split_key = candidate
                break
    if split_key is None:
        raise KeyError(f"No train/val/test field found in data yaml: {data_yaml_path}")

    split_value = data_yaml[split_key]
    base_root = data_yaml_path.parent
    dataset_root = data_yaml.get("path", None)
    dataset_root = resolve_path(str(dataset_root), base_root) if dataset_root else base_root

    split_path = Path(split_value)
    return split_path.resolve() if split_path.is_absolute() else (dataset_root / split_path).resolve()


# =========================================================
# Model / checkpoint
# =========================================================
def build_model_from_cfg(cfg, device):
    """
    当前版本不再涉及 snapshot_bias，但仍保留 snapshot_teacher，
    因为 aux_embedding.teacher_source 仍可能使用 snapshot。
    """
    aux_cfg = cfg["loss"].get("aux_embedding", {})
    runtime_cfg = cfg.get("runtime", {})
    layer_indices = tuple(cfg["model"]["layer_indices"])

    model = Stage1SslModel(
        yolo_model=cfg["model"]["yolo_model"],
        nc=cfg["model"].get("nc", None),
        layer_indices=layer_indices,
        image_size=cfg["data"]["image_size"],
        proj_dim=cfg["model"]["proj_dim"],
        local_dim=cfg["model"]["local_dim"],
        queue_size=cfg["model"]["queue_size"],
        momentum=cfg["model"]["momentum"],
        sppf_indice=cfg["model"].get("sppf_indice", 9),
        use_pos=runtime_cfg["use_pos"],
        pos_pe_channels=cfg["model"].get("pos_pe_channels", 64),
        pos_pe_spans=cfg["model"].get("pos_pe_spans", [1, 1, 1]),
        pos_init_scales=cfg["model"].get("pos_init_scales", [0.1, 0.5, 1.0]),
        pos_enable_fg_guidance=cfg["model"].get("pos_enable_fg_guidance", True),
        pos_fg_gate_init=cfg["model"].get("pos_fg_gate_init", 1.0),
        enable_raw_projection=aux_cfg.get("enabled", False),
        separate_projector=cfg["model"].get("separate_projector", False),
        use_snapshot_teacher=runtime_cfg.get(
            "needs_snapshot_teacher",
            cfg["model"]["snapshot_teacher"].get("enabled", False),
        ),
        verbose=False,
    )
    model = model.to(device)
    model.eval()
    return model


def recover_runtime_flags_from_state_dict(model, checkpoint_payload):
    """
    仅恢复 snapshot_ready。
    当前脚本不再关心 snapshot_bias，但 snapshot_teacher 仍可能被 aux_embedding 使用。
    """
    runtime = checkpoint_payload.get("runtime", {}) if isinstance(checkpoint_payload, dict) else {}
    if hasattr(model, "snapshot_ready"):
        if "snapshot_ready" in runtime:
            model.snapshot_ready = bool(runtime["snapshot_ready"])
        else:
            state_dict = checkpoint_payload.get("model", checkpoint_payload)
            has_snapshot_weights = any(k.startswith("snapshot_teacher.") for k in state_dict.keys())
            if has_snapshot_weights:
                model.snapshot_ready = True


def load_checkpoint_model(ckpt_path, cfg, device):
    model = build_model_from_cfg(cfg, device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    recover_runtime_flags_from_state_dict(model, ckpt if isinstance(ckpt, dict) else {"model": ckpt})
    model.eval()
    return model


def build_cfg_for_experiment(base_cfg, exp_dir):
    """
    优先读取实验目录下的 config_used.yaml，避免依赖外部 base config 猜测运行配置。
    """
    config_used_path = os.path.join(exp_dir, "config_used.yaml")
    if os.path.exists(config_used_path):
        cfg = load_config(config_used_path)
    else:
        cfg = copy.deepcopy(base_cfg)
    return validate_stage1_config(cfg)


# =========================================================
# Heatmap / visualization helpers
# =========================================================
def feature_to_heatmap(feat, out_size):
    """
    Convert a spatial feature tensor to a display heatmap.

    Modified display rule:
      - still uses abs_mean so the original logic is preserved;
      - uses a stricter percentile window (5% - 99.5%) to reduce
        visual saturation caused by very low responses and extreme outliers;
      - applies gamma correction (>1) so weak/mid responses are less likely
        to appear as red regions.

    This changes only the visualization display effect, not model features.
    """
    feat = feat.detach().float().cpu()
    heat = feat.abs().mean(dim=1)[0].numpy()

    q_low = np.percentile(heat, 5.0)
    q_high = np.percentile(heat, 99.5)
    if q_high <= q_low:
        q_low = heat.min()
        q_high = heat.max() + 1e-8

    heat = np.clip((heat - q_low) / (q_high - q_low + 1e-8), 0.0, 1.0)
    heat = np.power(heat, 1.35)
    heat = cv2.resize(heat, out_size, interpolation=cv2.INTER_LINEAR)
    # gaussian模糊平滑
    # heat = cv2.GaussianBlur(heat, ksize=(0, 0), sigmaX=16.0, sigmaY=16.0)
    return (heat * 255.0).astype(np.uint8)


def vector_to_heat_strip(vec, out_size):
    vec = vec.detach().float().cpu()
    if vec.ndim == 2:
        vec = vec[0]

    arr = vec.abs().numpy()
    q_low = np.percentile(arr, 1.0)
    q_high = np.percentile(arr, 99.0)
    if q_high <= q_low:
        q_low = arr.min()
        q_high = arr.max() + 1e-8

    arr = np.clip((arr - q_low) / (q_high - q_low + 1e-8), 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)[None, :]
    return cv2.resize(arr, out_size, interpolation=cv2.INTER_NEAREST)


def apply_colormap_overlay(image_rgb, heat_uint8, alpha=0.45):
    heat_color_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_color_rgb = cv2.cvtColor(heat_color_bgr, cv2.COLOR_BGR2RGB)
    overlay = (
        (1.0 - alpha) * image_rgb.astype(np.float32)
        + alpha * heat_color_rgb.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)
    return heat_color_rgb, overlay


def apply_colormap_to_gray(gray_uint8):
    color_bgr = cv2.applyColorMap(gray_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)


def make_overview_strip(image_rgb, overlays, titles=None, title_height=40):
    h, w, _ = image_rgb.shape
    panels = [image_rgb] + overlays
    strip = np.concatenate(panels, axis=1)

    if titles is None:
        return strip

    canvas = np.ones((h + title_height, strip.shape[1], 3), dtype=np.uint8) * 255
    canvas[title_height:, :, :] = strip
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, title in enumerate(titles):
        cv2.putText(canvas, title, (i * w + 10, 28), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    return canvas


def make_vertical_overview(panels, titles, pad=12, title_height=30, bg=255):
    if len(panels) == 0:
        return None

    widths = [p.shape[1] for p in panels]
    heights = [p.shape[0] for p in panels]
    out_w = max(widths)
    out_h = sum(h + title_height + pad for h in heights) - pad

    canvas = np.ones((out_h, out_w, 3), dtype=np.uint8) * bg
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 0

    for panel, title in zip(panels, titles):
        cv2.putText(canvas, title, (10, y + 22), font, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        y0 = y + title_height
        canvas[y0:y0 + panel.shape[0], 0:panel.shape[1], :] = panel
        y = y0 + panel.shape[0] + pad

    return canvas


def make_pair_strip(left_rgb, right_rgb, left_title, right_title, title_height=40):
    h, w, _ = left_rgb.shape
    strip = np.concatenate([left_rgb, right_rgb], axis=1)

    canvas = np.ones((h + title_height, strip.shape[1], 3), dtype=np.uint8) * 255
    canvas[title_height:, :, :] = strip

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, left_title, (10, 28), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, right_title, (w + 10, 28), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    return canvas


def save_spatial_feature_group(image_rgb, feat_list, save_dir, prefix, crop_box=None):
    """
    Save spatial feature heatmaps and overlays.

    image_rgb:
        Square model-input RGB image. Usually 640x640 if the preprocessing uses
        ResizeAndPadToSquare(long_size=640).

    crop_box:
        If provided, heatmap and overlay are first computed in the square model-input
        coordinate system, then cropped before saving. This removes black padding
        from saved visualization while keeping feature alignment correct.
    """
    ensure_dir(save_dir)

    image_rgb_save = crop_rgb_by_box(image_rgb, crop_box)

    overlays = []
    titles = ["input"]

    for i, feat in enumerate(feat_list, start=1):
        heat = feature_to_heatmap(feat, out_size=(image_rgb.shape[1], image_rgb.shape[0]))
        heat_rgb, overlay = apply_colormap_overlay(image_rgb, heat, alpha=0.45)

        heat_rgb_save = crop_rgb_by_box(heat_rgb, crop_box)
        overlay_save = crop_rgb_by_box(overlay, crop_box)

        save_rgb(os.path.join(save_dir, f"layer{i}_heatmap.jpg"), heat_rgb_save)
        save_rgb(os.path.join(save_dir, f"layer{i}_overlay.jpg"), overlay_save)

        overlays.append(overlay_save)
        titles.append(f"{prefix}_L{i}")

    overview = make_overview_strip(image_rgb_save, overlays, titles=titles)
    save_rgb(os.path.join(save_dir, "overview.jpg"), overview)
    return overlays


def save_vector_feature_group(image_width, feat_list, save_dir, prefix, strip_height=96):
    ensure_dir(save_dir)

    panels, titles = [], []
    for i, feat in enumerate(feat_list, start=1):
        strip_gray = vector_to_heat_strip(feat, out_size=(image_width, strip_height))
        strip_rgb = apply_colormap_to_gray(strip_gray)
        save_rgb(os.path.join(save_dir, f"layer{i}_vector.jpg"), strip_rgb)
        panels.append(strip_rgb)
        titles.append(f"{prefix}_L{i}")

    overview = make_vertical_overview(panels, titles)
    if overview is not None:
        save_rgb(os.path.join(save_dir, "overview.jpg"), overview)

    return panels


def compare_feature_groups(left_panels, right_panels, left_prefix, right_prefix, save_dir, filename_prefix):
    ensure_dir(save_dir)

    count = min(len(left_panels), len(right_panels))
    if count == 0:
        return None

    pair_panels = []
    for i in range(count):
        comp = make_pair_strip(
            left_panels[i],
            right_panels[i],
            f"{left_prefix}_L{i+1}",
            f"{right_prefix}_L{i+1}",
        )
        save_rgb(os.path.join(save_dir, f"{filename_prefix}_L{i+1}.jpg"), comp)
        pair_panels.append(comp)

    summary = np.concatenate(pair_panels, axis=0)
    save_rgb(os.path.join(save_dir, f"summary_{filename_prefix}.jpg"), summary)
    return summary


# =========================================================
# Diagnostic heatmap helpers
# =========================================================
def feature_to_score_map(feat, mode="abs_mean"):
    """
    Convert feature [B,C,H,W] to raw 2D score map before normalization.

    Modes:
      abs_mean  : original energy-style visualization source.
      relu_mean : positive activation only.
      var       : channel-wise variance, useful for checking whether
                  spatial texture/detail is preserved.
      l2        : channel-wise RMS energy.
    """
    feat = feat.detach().float().cpu()
    if feat.ndim != 4:
        raise ValueError(f"Expected feature shape [B,C,H,W], got {tuple(feat.shape)}")

    x = feat[0]
    if mode == "abs_mean":
        score = x.abs().mean(dim=0)
    elif mode == "relu_mean":
        score = torch.relu(x).mean(dim=0)
    elif mode == "var":
        score = x.var(dim=0, unbiased=False)
    elif mode == "l2":
        score = torch.sqrt((x ** 2).mean(dim=0) + 1e-8)
    else:
        raise ValueError(f"Unsupported diagnostic heatmap mode: {mode}")

    return score.numpy().astype(np.float32)


def compute_norm_range(score_maps, low_percentile=5.0, high_percentile=99.5):
    vals = []
    for score in score_maps:
        if score is None:
            continue
        vals.append(score.reshape(-1))
    if not vals:
        return 0.0, 1.0

    vals = np.concatenate(vals, axis=0)
    q_low = float(np.percentile(vals, low_percentile))
    q_high = float(np.percentile(vals, high_percentile))
    if q_high <= q_low:
        q_low = float(vals.min())
        q_high = float(vals.max() + 1e-8)
    return q_low, q_high


def normalize_score_map(score, norm_range=None, low_percentile=5.0, high_percentile=99.5, gamma=1.35):
    if norm_range is None:
        norm_range = compute_norm_range([score], low_percentile, high_percentile)
    q_low, q_high = norm_range
    x = np.clip((score.astype(np.float32) - q_low) / (q_high - q_low + 1e-8), 0.0, 1.0)
    if gamma is not None:
        x = np.power(x, float(gamma))
    return np.clip(x, 0.0, 1.0)


def score_map_stats(score, norm_map=None):
    x = score.astype(np.float32)
    stats = {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "cv": float(np.std(x) / (np.mean(x) + 1e-8)),
        "p1": float(np.percentile(x, 1)),
        "p5": float(np.percentile(x, 5)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
    }

    if norm_map is not None:
        n = norm_map.astype(np.float32)
        stats.update({
            "norm_mean": float(np.mean(n)),
            "norm_std": float(np.std(n)),
            "high_response_ratio_090": float(np.mean(n > 0.90)),
            "high_response_ratio_075": float(np.mean(n > 0.75)),
            "low_response_ratio_025": float(np.mean(n < 0.25)),
        })

    return stats


def diagnostic_heat_uint8(score, out_size, norm_range=None):
    norm = normalize_score_map(score, norm_range=norm_range)
    heat = cv2.resize(norm, out_size, interpolation=cv2.INTER_LINEAR)
    return (heat * 255.0).astype(np.uint8), norm


def save_shared_before_after_diagnostics(
    image_rgb,
    before_feats,
    after_feats,
    save_dir,
    prefix,
    crop_box=None,
    modes=("abs_mean", "relu_mean", "var", "l2"),
):
    """
    Extra diagnostics only. It does not replace the original outputs.

    For each mode and each layer, before/after use a shared normalization range,
    so the comparison is not affected by separate per-image percentile scaling.
    """
    ensure_dir(save_dir)
    out_size = (image_rgb.shape[1], image_rgb.shape[0])

    for mode in modes:
        stats = {}
        pair_panels = []

        for i, (feat_b, feat_a) in enumerate(zip(before_feats, after_feats), start=1):
            score_b = feature_to_score_map(feat_b, mode=mode)
            score_a = feature_to_score_map(feat_a, mode=mode)
            shared_range = compute_norm_range([score_b, score_a])

            heat_b, norm_b = diagnostic_heat_uint8(score_b, out_size=out_size, norm_range=shared_range)
            heat_a, norm_a = diagnostic_heat_uint8(score_a, out_size=out_size, norm_range=shared_range)

            heat_b_rgb, overlay_b = apply_colormap_overlay(image_rgb, heat_b, alpha=0.45)
            heat_a_rgb, overlay_a = apply_colormap_overlay(image_rgb, heat_a, alpha=0.45)

            heat_b_rgb = crop_rgb_by_box(heat_b_rgb, crop_box)
            heat_a_rgb = crop_rgb_by_box(heat_a_rgb, crop_box)
            overlay_b = crop_rgb_by_box(overlay_b, crop_box)
            overlay_a = crop_rgb_by_box(overlay_a, crop_box)

            save_rgb(os.path.join(save_dir, f"{prefix}_{mode}_L{i}_before_shared_heat.jpg"), heat_b_rgb)
            save_rgb(os.path.join(save_dir, f"{prefix}_{mode}_L{i}_after_shared_heat.jpg"), heat_a_rgb)
            save_rgb(os.path.join(save_dir, f"{prefix}_{mode}_L{i}_before_shared_overlay.jpg"), overlay_b)
            save_rgb(os.path.join(save_dir, f"{prefix}_{mode}_L{i}_after_shared_overlay.jpg"), overlay_a)

            pair = make_pair_strip(
                overlay_b,
                overlay_a,
                f"before_{prefix}_{mode}_L{i}",
                f"after_{prefix}_{mode}_L{i}",
            )
            save_rgb(os.path.join(save_dir, f"{prefix}_{mode}_L{i}_before_vs_after_shared.jpg"), pair)
            pair_panels.append(pair)

            stats[f"L{i}"] = {
                "mode": mode,
                "shared_norm_range": {
                    "low": float(shared_range[0]),
                    "high": float(shared_range[1]),
                },
                "before": score_map_stats(score_b, norm_b),
                "after": score_map_stats(score_a, norm_a),
            }

        if pair_panels:
            summary = np.concatenate(pair_panels, axis=0)
            save_rgb(os.path.join(save_dir, f"summary_{prefix}_{mode}_before_vs_after_shared.jpg"), summary)

        save_json(os.path.join(save_dir, f"{prefix}_{mode}_shared_stats.json"), stats)


# =========================================================
# SSL Grad-CAM helpers
# =========================================================
def normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = cam.astype(np.float32)
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    return cam


def compute_ssl_gradcam_for_raw_layer(
    model,
    x: torch.Tensor,
    fg_mask: Optional[torch.Tensor],
    layer_idx: int,
    target_source: str = "global_norm",
) -> np.ndarray:
    """
    Compute representation-level Grad-CAM for Stage1 SSL model.

    This is not class-specific Grad-CAM because Stage1 SSL has no class logits.
    Instead, it uses a representation scalar as the target.

    Args:
        model:
            Stage1SslModel.
        x:
            Input tensor [1,3,H,W].
        fg_mask:
            Optional foreground mask [1,1,H,W].
        layer_idx:
            Index of raw_feats layer to visualize.
        target_source:
            global_norm:
                use norm of the last global embedding as target.
            local_mean:
                use mean positive local embedding response as target.

    Returns:
        cam: np.ndarray [h,w], normalized to [0,1].
    """
    model.zero_grad(set_to_none=True)

    # Do not detach. We need gradients.
    outputs = model.online(x, fg_mask=fg_mask)

    raw_feats = outputs["raw_feats"]
    if layer_idx >= len(raw_feats):
        raise IndexError(f"layer_idx={layer_idx} out of range, raw_feats={len(raw_feats)}")

    act = raw_feats[layer_idx]
    act.retain_grad()

    if target_source == "global_norm":
        global_embs = outputs.get("global_embs", None)
        if global_embs is None:
            raise ValueError("global_embs not found in model outputs.")

        if isinstance(global_embs, (list, tuple)):
            target_vec = global_embs[-1]
        else:
            target_vec = global_embs

        target = target_vec.norm(p=2)

    elif target_source == "local_mean":
        local_embs = outputs.get("local_embs", None)
        if local_embs is None:
            raise ValueError("local_embs not found in model outputs.")

        if isinstance(local_embs, (list, tuple)):
            target_feat = local_embs[min(layer_idx, len(local_embs) - 1)]
        else:
            target_feat = local_embs

        target = torch.relu(target_feat).mean()

    else:
        raise ValueError(f"Unsupported target_source: {target_source}")

    target.backward(retain_graph=False)

    grad = act.grad
    if grad is None:
        raise RuntimeError("Grad-CAM failed: activation gradient is None.")

    # Grad-CAM weights: global average pooling over spatial dimensions.
    weights = grad.mean(dim=(2, 3), keepdim=True)

    cam = (weights * act).sum(dim=1, keepdim=False)
    cam = torch.relu(cam)[0]

    cam_np = cam.detach().float().cpu().numpy()
    cam_np = normalize_cam(cam_np)

    return cam_np


def save_ssl_gradcam_raw_group(
    model,
    x: torch.Tensor,
    fg_mask: Optional[torch.Tensor],
    image_rgb_square: np.ndarray,
    save_dir: str,
    prefix: str,
    crop_box=None,
    target_source: str = "global_norm",
    num_layers: int = 3,
):
    """
    Save SSL representation Grad-CAM maps for raw backbone layers.
    """
    ensure_dir(save_dir)

    overlays = []
    titles = ["input"]

    image_rgb_save = crop_rgb_by_box(image_rgb_square, crop_box)

    for layer_idx in range(num_layers):
        cam = compute_ssl_gradcam_for_raw_layer(
            model=model,
            x=x,
            fg_mask=fg_mask,
            layer_idx=layer_idx,
            target_source=target_source,
        )

        heat = cv2.resize(
            cam,
            (image_rgb_square.shape[1], image_rgb_square.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        heat_uint8 = (heat * 255.0).clip(0, 255).astype(np.uint8)

        heat_rgb, overlay = apply_colormap_overlay(
            image_rgb_square,
            heat_uint8,
            alpha=0.45,
        )

        heat_rgb_save = crop_rgb_by_box(heat_rgb, crop_box)
        overlay_save = crop_rgb_by_box(overlay, crop_box)

        save_rgb(os.path.join(save_dir, f"layer{layer_idx + 1}_gradcam_heat.jpg"), heat_rgb_save)
        save_rgb(os.path.join(save_dir, f"layer{layer_idx + 1}_gradcam_overlay.jpg"), overlay_save)

        overlays.append(overlay_save)
        titles.append(f"{prefix}_L{layer_idx + 1}")

    overview = make_overview_strip(
        image_rgb_save,
        overlays,
        titles=titles,
    )
    save_rgb(os.path.join(save_dir, "overview_gradcam.jpg"), overview)

    return overlays


# =========================================================
# Metadata helpers
# =========================================================
def _safe_relpath(path, root):
    try:
        return os.path.relpath(path, root)
    except Exception:
        return os.path.basename(path)


def build_sample_meta(image_path, cfg, sample_idx, sample_dir, vis_rgb=None, mask_saved_path=None):
    """
    Save provenance so downstream analysis scripts can recover the original
    image filename and its foreground mask.
    """
    image_root = cfg["data"].get("train_dir", "")
    image_abs = os.path.abspath(image_path)
    image_root_abs = os.path.abspath(image_root) if image_root else ""
    image_rel = _safe_relpath(image_abs, image_root_abs) if image_root_abs else os.path.basename(image_abs)

    runtime_cfg = cfg.get("runtime", {})
    data_cfg = cfg.get("data", {})
    use_mask = bool(runtime_cfg.get("use_mask", False))
    mask_root = data_cfg.get("mask_root_dir", None)
    mask_suffix = data_cfg.get("mask_suffix", ".png")
    threshold = data_cfg.get("external_mask_threshold", 127)

    expected_mask_abs = ""
    mask_abs = ""
    mask_exists = False
    mask_status = "mask_disabled"

    if use_mask:
        if mask_root:
            expected_mask_path = _resolve_mask_path(image_path, image_root, mask_root, mask_suffix)
            expected_mask_abs = os.path.abspath(expected_mask_path)
            mask_exists = os.path.exists(expected_mask_path)
            mask_abs = expected_mask_abs if mask_exists else ""
            mask_status = "external_mask_found" if mask_exists else "external_mask_missing_all_one_used"
        else:
            mask_status = "mask_root_missing_all_one_used"

    try:
        with Image.open(image_path) as im:
            original_size = {"width": int(im.size[0]), "height": int(im.size[1])}
    except Exception:
        original_size = {"width": None, "height": None}

    resized_size = None
    if vis_rgb is not None:
        resized_size = {"width": int(vis_rgb.shape[1]), "height": int(vis_rgb.shape[0])}

    return {
        "sample_id": f"sample_{sample_idx:03d}",
        "sample_index": int(sample_idx),
        "image_path": image_abs,
        "image_root": image_root_abs,
        "image_rel_path": image_rel,
        "image_name": os.path.basename(image_abs),
        "image_stem": Path(image_abs).stem,
        "original_size": original_size,
        "resized_size": resized_size,
        "use_mask": use_mask,
        "mask_root": os.path.abspath(mask_root) if mask_root else "",
        "mask_suffix": mask_suffix,
        "mask_threshold": int(threshold),
        "expected_mask_path": expected_mask_abs,
        "mask_path": mask_abs,
        "mask_exists": bool(mask_exists),
        "mask_status": mask_status,
        "mask_saved_path": os.path.abspath(mask_saved_path) if mask_saved_path else "",
        "sample_dir": os.path.abspath(sample_dir),
    }


def save_resized_mask(mask_tensor, path, crop_box=None):
    """
    Save the model-input-sized mask as an analysis fallback.

    If crop_box is provided, save the unpadded mask region only.
    """
    if mask_tensor is None:
        return

    m = mask_tensor.detach().float().cpu()
    if m.ndim == 4:
        m = m[0, 0]
    elif m.ndim == 3:
        m = m[0]

    arr = (m.numpy() > 0.5).astype(np.uint8) * 255
    arr = crop_gray_by_box(arr, crop_box)

    Image.fromarray(arr, mode="L").save(path)


def as_feature_list(feat):
    if feat is None:
        return []
    if torch.is_tensor(feat):
        return [feat]
    if isinstance(feat, (list, tuple)):
        return list(feat)
    return []


# =========================================================
# Main visualization
# =========================================================
@torch.no_grad()
def visualize_one_experiment(
    exp_dir,
    base_cfg,
    device,
    num_images=12,
    ckpt_name="best.pth",
    data_yaml_path: str = None,
    dataset_split=None,
    crop_padding_visuals: bool = False,
):
    ckpt_path = os.path.join(exp_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        fallback = os.path.join(exp_dir, "last.pth")
        if os.path.exists(fallback):
            ckpt_path = fallback
        else:
            print(f"[WARN] checkpoint not found in {exp_dir}, skip.")
            return
    ckpt_path = Path(ckpt_path).resolve()

    cfg = build_cfg_for_experiment(base_cfg, exp_dir)
    model_before = build_model_from_cfg(cfg, device)
    model_after = load_checkpoint_model(ckpt_path, cfg, device)

    dataset_splits = parse_dataset_splits(dataset_split)
    split_items = []
    split_roots = {}

    if data_yaml_path:
        for split_name in dataset_splits:
            image_root = resolve_dataset_split_dir(Path(data_yaml_path), split=split_name)
            images = evenly_sample(iter_images(image_root), num_images)
            split_roots[split_name] = image_root
            split_items.extend(
                (split_name, image_root, local_idx, image_path)
                for local_idx, image_path in enumerate(images)
            )
    else:
        image_root = cfg["data"]["train_dir"]
        images = evenly_sample(iter_images(image_root), num_images)
        split_name = dataset_splits[0] if dataset_splits else "train_dir"
        split_roots[split_name] = Path(image_root)
        split_items.extend(
            (split_name, Path(image_root), local_idx, image_path)
            for local_idx, image_path in enumerate(images)
        )

    save_root = os.path.join(exp_dir, "feature_maps_backbone")
    save_root = Path(save_root).resolve()
    ensure_dir(save_root)

    print("=" * 80)
    print(f"[INFO] Visualizing experiment: {cfg['experiment']['name']}")
    print(f"       checkpoint: {ckpt_path}")
    print(f"       use_pos   : {cfg['runtime']['use_pos']}")
    print(f"       use_mask  : {cfg['runtime']['use_mask']}")
    print(f"       splits    : {dataset_splits}")
    print(f"       images    : {len(split_items)}")
    print(f"       num_images: {num_images} ({'all per split' if int(num_images) < 0 else 'evenly sampled per split'})")
    print(f"       crop_pad  : {crop_padding_visuals}")
    print(f"       save_root : {save_root}")
    print("=" * 80)
    
    all_sample_meta = []

    multi_split = len(dataset_splits) > 1

    for idx, (split_name, image_root, local_idx, image_path) in enumerate(split_items):
        # Keep train_dir aligned with the active split so mask relative paths are resolved correctly.
        cfg["data"]["train_dir"] = str(image_root)

        vis_rgb_square, vis_rgb, crop_box, x, m, vis_info = load_image_and_mask_for_model(
            image_path,
            cfg,
            crop_padding_visuals=crop_padding_visuals,
        )

        x = x.to(device)
        m = m.to(device)
        fg_mask = m if cfg["runtime"]["use_mask"] else None

        outputs_before = model_before.online(x, fg_mask=fg_mask)
        outputs_after = model_after.online(x, fg_mask=fg_mask)

        sample_id = f"{split_name}_sample_{local_idx:05d}" if multi_split else f"sample_{local_idx:03d}"
        sample_dir = os.path.join(save_root, sample_id)
        compare_dir = os.path.join(sample_dir, "compare")
        ensure_dir(sample_dir)
        ensure_dir(compare_dir)

        # Saved input image. If crop_padding_visuals=True, this has black padding removed.
        save_rgb(os.path.join(sample_dir, "input.jpg"), vis_rgb)

        mask_saved_path = os.path.join(sample_dir, "mask_resized.png")
        save_resized_mask(m, mask_saved_path, crop_box=crop_box)

        sample_meta = build_sample_meta(
            image_path=image_path,
            cfg=cfg,
            sample_idx=idx,
            sample_dir=sample_dir,
            vis_rgb=vis_rgb,
            mask_saved_path=mask_saved_path,
        )
        sample_meta["dataset_split"] = split_name
        sample_meta["image_root"] = os.path.abspath(str(image_root))
        sample_meta["sample_id"] = sample_id
        sample_meta["visual_resize_info"] = vis_info
        save_json(os.path.join(sample_dir, "sample_meta.json"), sample_meta)
        all_sample_meta.append(sample_meta)

        # -----------------------------------------------------
        # Backbone raw features
        # -----------------------------------------------------
        before_raw = save_spatial_feature_group(
            vis_rgb_square,
            outputs_before["raw_feats"],
            os.path.join(sample_dir, "before_raw"),
            "before_raw",
            crop_box=crop_box,
        )
        after_raw = save_spatial_feature_group(
            vis_rgb_square,
            outputs_after["raw_feats"],
            os.path.join(sample_dir, "after_raw"),
            "after_raw",
            crop_box=crop_box,
        )
        compare_feature_groups(
            before_raw, after_raw,
            "before_raw", "after_raw",
            compare_dir, "before_vs_after_raw",
        )

        save_shared_before_after_diagnostics(
            vis_rgb_square,
            outputs_before["raw_feats"],
            outputs_after["raw_feats"],
            compare_dir,
            "raw",
            crop_box=crop_box,
        )

        # -----------------------------------------------------
        # SSL representation Grad-CAM for raw backbone features
        # -----------------------------------------------------
        with torch.enable_grad():
            x_cam = x.detach().clone().requires_grad_(True)
            m_cam = m.detach().clone() if m is not None else None
            fg_mask_cam = m_cam if cfg["runtime"]["use_mask"] else None

            model_before.zero_grad(set_to_none=True)
            before_raw_gradcam = save_ssl_gradcam_raw_group(
                model=model_before,
                x=x_cam,
                fg_mask=fg_mask_cam,
                image_rgb_square=vis_rgb_square,
                save_dir=os.path.join(sample_dir, "before_raw_gradcam"),
                prefix="before_raw_gradcam",
                crop_box=crop_box,
                target_source="global_norm",
                num_layers=len(outputs_before["raw_feats"]),
            )

        with torch.enable_grad():
            x_cam = x.detach().clone().requires_grad_(True)
            m_cam = m.detach().clone() if m is not None else None
            fg_mask_cam = m_cam if cfg["runtime"]["use_mask"] else None

            model_after.zero_grad(set_to_none=True)
            after_raw_gradcam = save_ssl_gradcam_raw_group(
                model=model_after,
                x=x_cam,
                fg_mask=fg_mask_cam,
                image_rgb_square=vis_rgb_square,
                save_dir=os.path.join(sample_dir, "after_raw_gradcam"),
                prefix="after_raw_gradcam",
                crop_box=crop_box,
                target_source="global_norm",
                num_layers=len(outputs_after["raw_feats"]),
            )

        compare_feature_groups(
            before_raw_gradcam,
            after_raw_gradcam,
            "before_raw_gradcam",
            "after_raw_gradcam",
            compare_dir,
            "before_vs_after_raw_gradcam",
        )

        # -----------------------------------------------------
        # Backbone SPPF features
        # -----------------------------------------------------
        before_sppf = save_spatial_feature_group(
            vis_rgb_square,
            as_feature_list(outputs_before["sppf_feat"]),
            os.path.join(sample_dir, "before_sppf"),
            "before_sppf",
            crop_box=crop_box,
        )
        after_sppf = save_spatial_feature_group(
            vis_rgb_square,
            as_feature_list(outputs_after["sppf_feat"]),
            os.path.join(sample_dir, "after_sppf"),
            "after_sppf",
            crop_box=crop_box,
        )
        compare_feature_groups(
            before_sppf, after_sppf,
            "before_sppf", "after_sppf",
            compare_dir, "before_vs_after_sppf",
        )

        # -----------------------------------------------------
        # Position-aware features (only when enabled)
        # -----------------------------------------------------
        if cfg["runtime"]["use_pos"]:
            before_pos = save_spatial_feature_group(
                vis_rgb_square,
                outputs_before["pos_feats"],
                os.path.join(sample_dir, "before_pos"),
                "before_pos",
                crop_box=crop_box,
            )
            after_pos = save_spatial_feature_group(
                vis_rgb_square,
                outputs_after["pos_feats"],
                os.path.join(sample_dir, "after_pos"),
                "after_pos",
                crop_box=crop_box,
            )
            compare_feature_groups(
                before_pos, after_pos,
                "before_pos", "after_pos",
                compare_dir, "before_vs_after_pos",
            )
            save_shared_before_after_diagnostics(
                vis_rgb_square,
                outputs_before["pos_feats"],
                outputs_after["pos_feats"],
                compare_dir,
                "pos",
                crop_box=crop_box,
            )
            compare_feature_groups(
                after_raw, after_pos,
                "after_raw", "after_pos",
                compare_dir, "after_raw_vs_pos",
            )

        # -----------------------------------------------------
        # Main branch embeddings
        # -----------------------------------------------------
        if outputs_before.get("global_embs") is not None and outputs_after.get("global_embs") is not None:
            before_global = save_vector_feature_group(
                vis_rgb.shape[1],
                outputs_before["global_embs"],
                os.path.join(sample_dir, "before_global_embs"),
                "before_global",
            )
            after_global = save_vector_feature_group(
                vis_rgb.shape[1],
                outputs_after["global_embs"],
                os.path.join(sample_dir, "after_global_embs"),
                "after_global",
            )
            compare_feature_groups(
                before_global, after_global,
                "before_global", "after_global",
                compare_dir, "before_vs_after_global_embs",
            )

        if outputs_before.get("local_embs") is not None and outputs_after.get("local_embs") is not None:
            before_local = save_spatial_feature_group(
                vis_rgb_square,
                outputs_before["local_embs"],
                os.path.join(sample_dir, "before_local_embs"),
                "before_local",
                crop_box=crop_box,
            )
            after_local = save_spatial_feature_group(
                vis_rgb_square,
                outputs_after["local_embs"],
                os.path.join(sample_dir, "after_local_embs"),
                "after_local",
                crop_box=crop_box,
            )
            compare_feature_groups(
                before_local, after_local,
                "before_local", "after_local",
                compare_dir, "before_vs_after_local_embs",
            )

        # -----------------------------------------------------
        # Raw projection branch (only exists when aux_embedding is enabled)
        # -----------------------------------------------------
        if outputs_before.get("raw_global_embs") is not None and outputs_after.get("raw_global_embs") is not None:
            before_raw_global = save_vector_feature_group(
                vis_rgb.shape[1],
                outputs_before["raw_global_embs"],
                os.path.join(sample_dir, "before_raw_global_embs"),
                "before_raw_global",
            )
            after_raw_global = save_vector_feature_group(
                vis_rgb.shape[1],
                outputs_after["raw_global_embs"],
                os.path.join(sample_dir, "after_raw_global_embs"),
                "after_raw_global",
            )
            compare_feature_groups(
                before_raw_global, after_raw_global,
                "before_raw_global", "after_raw_global",
                compare_dir, "before_vs_after_raw_global_embs",
            )

        if outputs_before.get("raw_local_embs") is not None and outputs_after.get("raw_local_embs") is not None:
            before_raw_local = save_spatial_feature_group(
                vis_rgb_square,
                outputs_before["raw_local_embs"],
                os.path.join(sample_dir, "before_raw_local_embs"),
                "before_raw_local",
                crop_box=crop_box,
            )
            after_raw_local = save_spatial_feature_group(
                vis_rgb_square,
                outputs_after["raw_local_embs"],
                os.path.join(sample_dir, "after_raw_local_embs"),
                "after_raw_local",
                crop_box=crop_box,
            )
            compare_feature_groups(
                before_raw_local, after_raw_local,
                "before_raw_local", "after_raw_local",
                compare_dir, "before_vs_after_raw_local_embs",
            )

        # -----------------------------------------------------
        # After-training cross-branch compare
        # -----------------------------------------------------
        if outputs_after.get("local_embs") is not None and outputs_after.get("raw_local_embs") is not None:
            after_local = save_spatial_feature_group(
                vis_rgb_square,
                outputs_after["local_embs"],
                os.path.join(sample_dir, "after_local_embs"),
                "after_local",
                crop_box=crop_box,
            )
            after_raw_local = save_spatial_feature_group(
                vis_rgb_square,
                outputs_after["raw_local_embs"],
                os.path.join(sample_dir, "after_raw_local_embs"),
                "after_raw_local",
                crop_box=crop_box,
            )
            compare_feature_groups(
                after_raw_local, after_local,
                "after_raw_local", "after_local",
                compare_dir, "after_raw_local_vs_local_embs",
            )

        if outputs_after.get("global_embs") is not None and outputs_after.get("raw_global_embs") is not None:
            after_global = save_vector_feature_group(
                vis_rgb.shape[1],
                outputs_after["global_embs"],
                os.path.join(sample_dir, "after_global_embs"),
                "after_global",
            )
            after_raw_global = save_vector_feature_group(
                vis_rgb.shape[1],
                outputs_after["raw_global_embs"],
                os.path.join(sample_dir, "after_raw_global_embs"),
                "after_raw_global",
            )
            compare_feature_groups(
                after_raw_global, after_global,
                "after_raw_global", "after_global",
                compare_dir, "after_raw_global_vs_global_embs",
            )

        print(f"[INFO] Visualizing {Path(image_path).stem} is done.")

    exp_meta = {
        "experiment_name": cfg.get("experiment", {}).get("name", Path(exp_dir).name),
        "exp_dir": os.path.abspath(exp_dir),
        "checkpoint": os.path.abspath(ckpt_path),
        "save_root": os.path.abspath(save_root),
        "dataset_splits": dataset_splits,
        "split_roots": {k: os.path.abspath(str(v)) for k, v in split_roots.items()},
        "num_images_arg": int(num_images),
        "num_samples": int(len(all_sample_meta)),
        "crop_padding_visuals": bool(crop_padding_visuals),
        "samples": all_sample_meta,
    }
    save_json(os.path.join(save_root, "meta.json"), exp_meta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/ssl_config.yaml")
    parser.add_argument("--runs_dir", type=str, default="./runs/glcp_stage1_yolo_det")
    parser.add_argument("--exp_names", type=str, nargs="+", default=["use_pos_mask", "wo_mask", "wo_pos", "wo_pos_wo_mask"])
    parser.add_argument("--exp_target", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=12, help="Images per split. Use -1 to visualize all images in each split.")
    parser.add_argument("--ckpt_name", type=str, default="best.pth")
    parser.add_argument("--data_yaml", type=str, default="./data/det_dataset.yaml")
    parser.add_argument("--dataset_split", type=str, nargs="+", default=["val"], help="One or more dataset splits, e.g. --dataset_split val test or --dataset_split val,test")
    parser.add_argument(
        "--crop_padding_visuals",
        action="store_true",
        help=(
            "Crop black padding from saved visualization images. "
            "Model input remains ResizeAndPadToSquare(image_size), but saved overlays are cropped "
            "to the resized original aspect ratio with long side=image_size."
        ),
    )

    args = parser.parse_args()
    args.crop_padding_visuals = True
    
    base_cfg = validate_stage1_config(load_config(args.config))
    device = torch.device(base_cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    # 从 config 中获取
    if args.exp_target is None:
        args.exp_target = base_cfg["experiment"]["name"]

    # 适配默认的 all
    exp_list = args.exp_names if args.exp_target == "all" else [args.exp_target]

    for exp_name in exp_list:
        exp_dir = os.path.join(args.runs_dir, exp_name)
        if not os.path.isdir(exp_dir):
            print(f"[WARN] exp dir not found: {exp_dir}")
            continue

        visualize_one_experiment(
            exp_dir,
            base_cfg,
            device,
            args.num_images,
            args.ckpt_name,
            args.data_yaml,
            args.dataset_split,
            crop_padding_visuals=args.crop_padding_visuals,
        )

        print(f"[INFO] Visualizing experiment({exp_name}) Done.")


if __name__ == "__main__":
    try:
        main()
    finally:
        from utils.utils import cleanup_memory
        cleanup_memory()