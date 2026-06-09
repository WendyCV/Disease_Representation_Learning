from __future__ import annotations

import os
import sys
import json
import copy
import yaml
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import torch
import numpy as np
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "tools" else os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.yolo_model import YOLO  # type: ignore
from models.stage1_ssl_model import Stage1SslModel  # type: ignore
from utils.config_utils import validate_stage1_config  # type: ignore
from datasets.augmentations import build_base_transform  # type: ignore

import warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

DEFAULT_LAYERS = {
    "yolov8n.pt": [4, 6, 8, 9, 15, 18, 21],
    "yolov9t.pt": [4, 6, 8, 9, 15, 18, 21],
    "yolov10n.pt": [4, 6, 8, 9, 16, 19, 22],
    "yolo11n.pt": [4, 6, 8, 9, 16, 19, 22],
}

DEFAULT_ROLE_MAP = {
    4: "backbone_l1",
    6: "backbone_l2",
    8: "backbone_l3",
    9: "backbone_sppf",
    # yolov8n/yolov9t
    15: "prehead_p3",
    18: "prehead_p4",
    21: "prehead_p5",
    # yolov10n/yolo11n
    16: "prehead_p3",
    19: "prehead_p4",
    22: "prehead_p5",
}


# =========================================================
# Basic IO
# =========================================================
def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str | Path, base_dir: Optional[Path] = None) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    if base_dir is not None:
        return (base_dir / p).resolve()
    return (Path(PROJECT_ROOT) / p).resolve()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_images(root_dir: str | Path) -> List[str]:
    root = Path(root_dir)
    return [
        str(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    ]


def evenly_sample(items: Sequence[str], num_samples: int) -> List[str]:
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


def save_rgb(path: str | Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb).save(path)


def save_gray(path: str | Path, gray: np.ndarray) -> None:
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    Image.fromarray(gray, mode="L").save(path)


def write_json(path: str | Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_cfg_for_experiment(base_cfg: dict, exp_dir: str | Path) -> dict:
    exp_dir = Path(exp_dir)
    for name in ("config_used.yaml", "config.yaml"):
        p = exp_dir / name
        if p.exists():
            return load_config(p)
    return copy.deepcopy(base_cfg)


# =========================================================
# Dataset helpers
# =========================================================
def load_data_yaml(data_yaml_path: Path) -> dict:
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_dataset_split_dir(data_yaml_path: Path, dataset_split: str = "val") -> Path:
    data_yaml = load_data_yaml(data_yaml_path)
    split_key = dataset_split if dataset_split in data_yaml else None
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


def infer_dataset_root_from_data_yaml(data_yaml_path: Path) -> Path:
    data_yaml = load_data_yaml(data_yaml_path)
    if data_yaml.get("path", None):
        return resolve_path(str(data_yaml["path"]), data_yaml_path.parent)
    return data_yaml_path.parent


def image_to_label_path(image_path: str | Path) -> Path:
    p = Path(image_path)
    parts = list(p.parts)
    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return p.parent.parent / "labels" / f"{p.stem}.txt"


def candidate_mask_paths(
    image_path: str | Path,
    *,
    mask_root: Optional[Path],
    image_root: Optional[Path] = None,
    dataset_root: Optional[Path] = None,
) -> List[Path]:
    if mask_root is None:
        return []

    image_path = Path(image_path).resolve()
    out: List[Path] = []
    suffixes = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]

    rel_candidates: List[Path] = []

    if image_root is not None:
        try:
            rel_candidates.append(image_path.relative_to(image_root.resolve()).with_suffix(""))
        except Exception:
            pass

    if dataset_root is not None:
        try:
            rel_candidates.append(image_path.relative_to(dataset_root.resolve()).with_suffix(""))
        except Exception:
            pass

    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            rel_candidates.append(Path(*parts[idx + 1:]).with_suffix(""))
            break

    rel_candidates.append(Path(image_path.stem))

    seen = set()
    for rel in rel_candidates:
        for suffix in suffixes:
            p = (mask_root / f"{rel}{suffix}").resolve()
            if p not in seen:
                out.append(p)
                seen.add(p)

    return out


def find_mask_path(
    image_path: str | Path,
    *,
    mask_root: Optional[Path],
    image_root: Optional[Path],
    dataset_root: Optional[Path],
) -> Tuple[Optional[Path], List[str]]:
    cands = candidate_mask_paths(
        image_path,
        mask_root=mask_root,
        image_root=image_root,
        dataset_root=dataset_root,
    )
    for p in cands:
        if p.exists():
            return p, [str(x) for x in cands]
    return None, [str(x) for x in cands]


# =========================================================
# YOLO-style letterbox helpers
# =========================================================
def letterbox_rgb(
    image_rgb: np.ndarray,
    imgsz: int,
    pad_value: int = 114,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int], Dict[str, Any]]:
    """
    YOLO-style resize + center pad.

    Returns:
        model_rgb:
            Padded square image for model input, shape = imgsz x imgsz.
        visual_rgb_no_pad:
            Resized image without padding, long side = imgsz.
        crop_box:
            Valid non-padding region in model_rgb, format = (x1, y1, x2, y2).
        resize_info:
            Metadata for sample_meta.json.
    """
    h, w = image_rgb.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image shape: {image_rgb.shape}")

    scale = float(imgsz) / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    new_w = max(1, min(int(imgsz), new_w))
    new_h = max(1, min(int(imgsz), new_h))

    visual_rgb_no_pad = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    model_rgb = np.ones((imgsz, imgsz, 3), dtype=np.uint8) * int(pad_value)

    pad_w = int(imgsz) - new_w
    pad_h = int(imgsz) - new_h

    left = pad_w // 2
    top = pad_h // 2
    right = left + new_w
    bottom = top + new_h

    model_rgb[top:bottom, left:right, :] = visual_rgb_no_pad

    crop_box = (int(left), int(top), int(right), int(bottom))

    resize_info = {
        "original_size": {
            "width": int(w),
            "height": int(h),
        },
        "model_input_size": {
            "width": int(imgsz),
            "height": int(imgsz),
        },
        "resized_no_pad_size": {
            "width": int(new_w),
            "height": int(new_h),
        },
        "scale": float(scale),
        "pad": {
            "left": int(left),
            "top": int(top),
            "right": int(imgsz - right),
            "bottom": int(imgsz - bottom),
        },
        "crop_box_in_model_input": {
            "x1": int(left),
            "y1": int(top),
            "x2": int(right),
            "y2": int(bottom),
        },
    }

    return model_rgb, visual_rgb_no_pad, crop_box, resize_info


def crop_rgb_by_box(
    rgb: np.ndarray,
    crop_box: Optional[Tuple[int, int, int, int]],
) -> np.ndarray:
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
    if crop_box is None:
        return gray

    x1, y1, x2, y2 = crop_box
    h, w = gray.shape[:2]

    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(x1 + 1, min(w, int(x2)))
    y2 = max(y1 + 1, min(h, int(y2)))

    return gray[y1:y2, x1:x2].copy()


def letterbox_gray_like_image(
    gray: np.ndarray,
    imgsz: int,
    crop_box: Tuple[int, int, int, int],
    pad_value: int = 0,
) -> np.ndarray:
    """
    Resize a gray mask to the same valid region as the RGB letterbox result.
    """
    x1, y1, x2, y2 = crop_box
    valid_w = max(1, int(x2 - x1))
    valid_h = max(1, int(y2 - y1))

    resized = cv2.resize(gray, (valid_w, valid_h), interpolation=cv2.INTER_NEAREST)
    model_gray = np.ones((imgsz, imgsz), dtype=np.uint8) * int(pad_value)
    model_gray[y1:y2, x1:x2] = resized

    return model_gray


# =========================================================
# YOLO hook helpers
# =========================================================
class LeafnessHeadForViz(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        pooled = x.abs().mean(dim=1, keepdim=True)
        logit = self.scale.to(x.device, x.dtype) * pooled + self.bias.to(x.device, x.dtype)
        prob = torch.sigmoid(logit)
        return logit, prob


def load_lpot_proxy_heads_from_sidecar(exp_dir: Path, device: torch.device) -> Dict[int, torch.nn.Module]:
    sidecar_path = exp_dir / "weights" / "lpot_runtime_state.pt"
    if not sidecar_path.exists():
        return {}

    payload = torch.load(str(sidecar_path), map_location=device)

    block = payload.get("ema") or payload.get("model") or {}
    state = block.get("state", {}) or {}
    proxy_layer_indices = block.get("proxy_layer_indices", []) or payload.get("lpot_cfg", {}).get("proxy_layer_indices", [15, 18, 21])

    heads: Dict[int, torch.nn.Module] = {}

    for idx in proxy_layer_indices:
        name = f"_lpot_proxy_head_{int(idx)}"
        sd = state.get(name, None)
        if sd is None:
            continue

        head = LeafnessHeadForViz().to(device)
        head.load_state_dict(sd, strict=True)
        head.eval()
        heads[int(idx)] = head

    return heads


def _extract_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)):
        for item in reversed(output):
            if torch.is_tensor(item):
                return item
    if isinstance(output, dict):
        for item in output.values():
            if torch.is_tensor(item):
                return item
    raise TypeError(f"Unsupported hooked output type: {type(output)}")


class FeatureHook:
    def __init__(self, layer_idx: int, cache: Dict[int, torch.Tensor]):
        self.layer_idx = int(layer_idx)
        self.cache = cache

    def __call__(self, _module, _inputs, output):
        self.cache[self.layer_idx] = _extract_tensor(output).detach()


def build_hooks(det_model: torch.nn.Module, layer_indices: Sequence[int], cache: Dict[int, torch.Tensor]):
    handles = []
    total_layers = len(det_model.model)
    for idx in layer_indices:
        if idx < 0 or idx >= total_layers:
            raise IndexError(f"Layer index {idx} out of range. Model has {total_layers} layers.")
        handles.append(det_model.model[idx].register_forward_hook(FeatureHook(idx, cache)))
    return handles


# =========================================================
# Visualization helpers
# =========================================================
def preprocess_image(
    image_path: str | Path,
    imgsz: int,
    device: torch.device,
    crop_padding_visuals: bool = False,
):
    """
    YOLO-consistent preprocessing for visualization.

    Model input:
        original image -> keep aspect ratio -> resize long side to imgsz
        -> center pad to imgsz x imgsz.

    Saved visualization:
        if crop_padding_visuals=True:
            save non-padded resized image, no padding border.
        else:
            save padded square image.
    """
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    orig_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    model_rgb, visual_rgb_no_pad, crop_box, resize_info = letterbox_rgb(
        orig_rgb,
        imgsz=imgsz,
        pad_value=114,
    )

    visual_rgb = visual_rgb_no_pad if crop_padding_visuals else model_rgb.copy()

    tensor = torch.from_numpy(model_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)

    resize_info["crop_padding_visuals"] = bool(crop_padding_visuals)
    resize_info["saved_visual_size"] = {
        "width": int(visual_rgb.shape[1]),
        "height": int(visual_rgb.shape[0]),
    }

    return orig_rgb, model_rgb, visual_rgb, crop_box, tensor, resize_info


def normalize_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    vmin, vmax = float(np.min(x)), float(np.max(x))
    if vmax > vmin:
        x = (x - vmin) / (vmax - vmin + 1e-8)
    else:
        x = np.zeros_like(x, dtype=np.float32)
    return np.clip(x, 0.0, 1.0)


def feature_to_heatmap(feat: torch.Tensor, out_size: Tuple[int, int]) -> np.ndarray:
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
    # heat = cv2.GaussianBlur(heat, ksize=(0, 0), sigmaX=2.0, sigmaY=2.0)
    return (heat * 255.0).astype(np.uint8)


# =========================================================
# Diagnostic heatmap helpers
# =========================================================
def feature_to_score_map(feat: torch.Tensor, mode: str = "abs_mean") -> np.ndarray:
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


def score_map_stats(score: np.ndarray, norm_map: Optional[np.ndarray] = None) -> Dict[str, float]:
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


def collect_feature_diagnostic_stats(feat_map_dict: Dict[int, torch.Tensor], layer_indices: Sequence[int]) -> Dict[str, Any]:
    """
    Extra diagnostics only. Does not affect heatmap saving.
    """
    out: Dict[str, Any] = {}

    for idx in layer_indices:
        if idx not in feat_map_dict:
            continue

        role = infer_layer_role(idx)
        feat = feat_map_dict[idx]
        role_stats: Dict[str, Any] = {
            "layer_idx": int(idx),
            "role": role,
            "feature_shape": list(feat.shape),
            "modes": {},
        }

        for mode in ("abs_mean", "relu_mean", "var", "l2"):
            score = feature_to_score_map(feat, mode=mode)
            norm_range = compute_norm_range([score])
            norm = normalize_score_map(score, norm_range=norm_range)
            role_stats["modes"][mode] = {
                "norm_range": {
                    "low": float(norm_range[0]),
                    "high": float(norm_range[1]),
                },
                "stats": score_map_stats(score, norm),
            }

        out[role] = role_stats

    return out


def map01_to_uint8(x: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    x = normalize_01(x)
    x = cv2.resize(x, out_size, interpolation=cv2.INTER_LINEAR)
    return (x * 255.0).astype(np.uint8)


def apply_colormap_overlay(image_rgb: np.ndarray, heat_uint8: np.ndarray, alpha: float = 0.45):
    heat_color_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_color_rgb = cv2.cvtColor(heat_color_bgr, cv2.COLOR_BGR2RGB)
    overlay = (
        (1.0 - alpha) * image_rgb.astype(np.float32)
        + alpha * heat_color_rgb.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)
    return heat_color_rgb, overlay


def make_overview_strip(
    panels: List[np.ndarray],
    titles=None,
    title_height: int = 40,
    row_gap: int = 10,
    col_gap: int = 10,
    bg_color=(255, 255, 255),
):
    if len(panels) == 0:
        raise ValueError("panels cannot be empty")
    if len(panels) < 6:
        raise ValueError("This layout requires at least 5 panels")
    if titles is not None and len(titles) != len(panels):
        raise ValueError("titles length must match panels length")

    h, w, _ = panels[0].shape
    norm_panels = []
    for p in panels:
        if p.shape[:2] != (h, w):
            p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
        if p.ndim == 2:
            p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
        elif p.shape[2] == 4:
            p = cv2.cvtColor(p, cv2.COLOR_BGRA2BGR)
        norm_panels.append(p)

    row1 = list(range(0, 3))
    row2 = list(range(3, len(panels) - 3))
    row3 = list(range(len(panels) - 3, len(panels)))

    rows = [row1]
    if len(row2) > 0:
        rows.append(row2)
    rows.append(row3)

    def row_width(row_indices):
        n = len(row_indices)
        return n * w + (n - 1) * col_gap

    max_width = max(row_width(r) for r in rows)
    row_full_height = h + (title_height if titles is not None else 0)
    total_height = len(rows) * row_full_height + (len(rows) - 1) * row_gap
    canvas = np.ones((total_height, max_width, 3), dtype=np.uint8)
    canvas[:, :, :] = bg_color
    font = cv2.FONT_HERSHEY_SIMPLEX

    y = 0
    for row_indices in rows:
        x_start = 0
        for j, idx in enumerate(row_indices):
            x = x_start + j * (w + col_gap)
            if titles is not None:
                cv2.putText(
                    canvas,
                    str(titles[idx]),
                    (x + 10, y + 28),
                    font,
                    0.72,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                img_y = y + title_height
            else:
                img_y = y
            canvas[img_y:img_y + h, x:x + w, :] = norm_panels[idx]
        y += row_full_height + row_gap

    return canvas


def infer_layer_role(layer_idx: int) -> str:
    return DEFAULT_ROLE_MAP.get(layer_idx, f"layer_{layer_idx}")


def fuse_heat_uint8(maps: List[np.ndarray], out_size: Tuple[int, int]) -> Optional[np.ndarray]:
    if not maps:
        return None
    arr = []
    for m in maps:
        mm = cv2.resize(m.astype(np.float32) / 255.0, out_size, interpolation=cv2.INTER_LINEAR)
        arr.append(mm)
    fused = normalize_01(np.mean(arr, axis=0))
    return (fused * 255.0).astype(np.uint8)


def save_spatial_feature_group(
    image_rgb: np.ndarray,
    feat_map_dict: Dict[int, torch.Tensor],
    layer_indices: Sequence[int],
    save_dir: str | Path,
    prediction_rgb: Optional[np.ndarray] = None,
    crop_box: Optional[Tuple[int, int, int, int]] = None,
    annotation_rgb=None,
) -> Dict[str, Dict[str, Any]]:
    """
    Save spatial feature heatmaps and overlays.

    image_rgb:
        Padded square model-input RGB image.

    crop_box:
        If provided, saved heatmaps/overlays/overview are cropped to the valid
        non-padding region.
    """
    save_dir = ensure_dir(save_dir)

    image_rgb_save = crop_rgb_by_box(image_rgb, crop_box)

    panel_images = [image_rgb_save]
    titles = ["input"]

    feature_files: Dict[str, Dict[str, Any]] = {}
    role_heatmaps: Dict[str, np.ndarray] = {}

    if prediction_rgb is not None:
        if prediction_rgb.shape[:2] != image_rgb.shape[:2]:
            prediction_rgb = cv2.resize(
                prediction_rgb,
                (image_rgb.shape[1], image_rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        prediction_rgb_save = crop_rgb_by_box(prediction_rgb, crop_box)
        panel_images.append(prediction_rgb_save)
        titles.append("prediction")
    #
    if annotation_rgb is not None:
        panel_images.append(annotation_rgb)
        titles.append("GT annotation")

    for idx in layer_indices:
        feat = feat_map_dict[idx]
        role = infer_layer_role(idx)

        heat = feature_to_heatmap(
            feat,
            out_size=(image_rgb.shape[1], image_rgb.shape[0]),
        )
        heat_rgb, overlay = apply_colormap_overlay(image_rgb, heat, alpha=0.45)

        heat_save = crop_gray_by_box(heat, crop_box)
        heat_rgb_save = crop_rgb_by_box(heat_rgb, crop_box)
        overlay_save = crop_rgb_by_box(overlay, crop_box)

        raw_path = save_dir / f"{idx:02d}_{role}_raw.png"
        heat_path = save_dir / f"{idx:02d}_{role}_heat.jpg"
        overlay_path = save_dir / f"{idx:02d}_{role}_overlay.jpg"

        save_gray(raw_path, heat_save)
        save_rgb(heat_path, heat_rgb_save)
        save_rgb(overlay_path, overlay_save)

        role_heatmaps[role] = heat

        feature_files[role] = {
            "layer_idx": int(idx),
            "raw_path": str(raw_path),
            "gray_path": str(raw_path),
            "heat_path": str(heat_path),
            "heat_color_path": str(heat_path),
            "overlay_path": str(overlay_path),
            "source": "yolo_feature_heatmap",
            "crop_padding_visuals": bool(crop_box is not None),
        }

        panel_images.append(overlay_save)
        titles.append(f"{idx}:{role}")

    prehead_maps = [
        role_heatmaps[r]
        for r in ("prehead_p3", "prehead_p4", "prehead_p5")
        if r in role_heatmaps
    ]

    fused_prehead = fuse_heat_uint8(
        prehead_maps,
        out_size=(image_rgb.shape[1], image_rgb.shape[0]),
    )

    if fused_prehead is not None:
        heat_rgb, overlay = apply_colormap_overlay(image_rgb, fused_prehead, alpha=0.45)

        fused_save = crop_gray_by_box(fused_prehead, crop_box)
        heat_rgb_save = crop_rgb_by_box(heat_rgb, crop_box)
        overlay_save = crop_rgb_by_box(overlay, crop_box)

        raw_path = save_dir / "prehead_support_fused_raw.png"
        heat_path = save_dir / "prehead_support_fused_heat.jpg"
        overlay_path = save_dir / "prehead_support_fused_overlay.jpg"

        save_gray(raw_path, fused_save)
        save_rgb(heat_path, heat_rgb_save)
        save_rgb(overlay_path, overlay_save)

        feature_files["prehead_support_fused"] = {
            "raw_path": str(raw_path),
            "gray_path": str(raw_path),
            "heat_path": str(heat_path),
            "heat_color_path": str(heat_path),
            "overlay_path": str(overlay_path),
            "source": "fused_prehead_feature_heatmap",
            "note": "This is a feature-based support-like map, not the internal LPOT sigmoid probability unless LPOT support heads are explicitly saved.",
            "crop_padding_visuals": bool(crop_box is not None),
        }

    overview = make_overview_strip(panel_images, titles=titles)
    save_rgb(save_dir / "overview.jpg", overview)

    return feature_files


def _proxy_layer_role_from_idx(layer_idx: int) -> str:
    """Map LPOT proxy layer index to p3/p4/p5 style role names."""
    base_role = infer_layer_role(int(layer_idx))
    if base_role.startswith("prehead_"):
        return base_role.replace("prehead_", "lpot_proxy_prob_")
    return f"lpot_proxy_prob_layer_{int(layer_idx)}"


def _find_lpot_proxy_head(det_model: torch.nn.Module, layer_idx: int) -> Optional[torch.nn.Module]:
    names = [
        f"_lpot_proxy_head_{int(layer_idx)}",
        f"lpot_proxy_head_{int(layer_idx)}",
    ]
    candidates = [det_model, getattr(det_model, "model", None)]
    for obj in candidates:
        if obj is None:
            continue
        for name in names:
            head = getattr(obj, name, None)
            if isinstance(head, torch.nn.Module):
                return head
    return None


def _prob_tensor_to_uint8(prob: torch.Tensor, out_size: Tuple[int, int]) -> np.ndarray:
    prob_np = prob.detach().float().cpu()[0, 0].numpy()
    prob_np = np.clip(prob_np, 0.0, 1.0)
    prob_np = cv2.resize(prob_np, out_size, interpolation=cv2.INTER_LINEAR)
    return (np.clip(prob_np, 0.0, 1.0) * 255.0).astype(np.uint8)


@torch.no_grad()
def save_lpot_proxy_probability_maps(
    image_rgb: np.ndarray,
    feat_map_dict: Dict[int, torch.Tensor],
    det_model: torch.nn.Module,
    proxy_layer_indices: Sequence[int],
    save_dir: str | Path,
    sidecar_proxy_heads,
    crop_box: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    save_dir = ensure_dir(save_dir)
    feature_files: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    prob_maps: List[np.ndarray] = []
    prob_roles: List[str] = []
    out_size = (image_rgb.shape[1], image_rgb.shape[0])

    for layer_idx in proxy_layer_indices:
        layer_idx = int(layer_idx)
        role = _proxy_layer_role_from_idx(layer_idx)
        feat = feat_map_dict.get(layer_idx, None)
        head = sidecar_proxy_heads.get(layer_idx, None)
        if head is None:
            head = _find_lpot_proxy_head(det_model, layer_idx)

        if feat is None:
            missing.append(f"layer_{layer_idx}:missing_feature_cache")
            continue
        if head is None:
            missing.append(f"layer_{layer_idx}:missing_lpot_proxy_head")
            continue

        head = head.to(device=feat.device)
        head.eval()
        out = head(feat)

        if isinstance(out, (tuple, list)) and len(out) >= 2:
            prob = out[1]
        elif torch.is_tensor(out):
            prob = torch.sigmoid(out)
        else:
            missing.append(f"layer_{layer_idx}:unsupported_head_output_{type(out)}")
            continue

        prob_uint8 = _prob_tensor_to_uint8(prob, out_size=out_size)
        heat_rgb, overlay = apply_colormap_overlay(image_rgb, prob_uint8, alpha=0.45)

        prob_save = crop_gray_by_box(prob_uint8, crop_box)
        heat_rgb_save = crop_rgb_by_box(heat_rgb, crop_box)
        overlay_save = crop_rgb_by_box(overlay, crop_box)

        raw_path = save_dir / f"{role}_raw.png"
        heat_path = save_dir / f"{role}_heat.jpg"
        overlay_path = save_dir / f"{role}_overlay.jpg"

        save_gray(raw_path, prob_save)
        save_rgb(heat_path, heat_rgb_save)
        save_rgb(overlay_path, overlay_save)

        prob_maps.append(prob_uint8)
        prob_roles.append(role)

        feature_files[role] = {
            "layer_idx": int(layer_idx),
            "raw_path": str(raw_path),
            "gray_path": str(raw_path),
            "heat_path": str(heat_path),
            "heat_color_path": str(heat_path),
            "overlay_path": str(overlay_path),
            "source": "lpot_internal_proxy_probability",
            "note": "Sigmoid probability from the saved LPOT proxy head; this is the intended internal objectness-like proxy support map.",
            "crop_padding_visuals": bool(crop_box is not None),
        }

    if prob_maps:
        fused = fuse_heat_uint8(prob_maps, out_size=out_size)
        if fused is not None:
            heat_rgb, overlay = apply_colormap_overlay(image_rgb, fused, alpha=0.45)

            fused_save = crop_gray_by_box(fused, crop_box)
            heat_rgb_save = crop_rgb_by_box(heat_rgb, crop_box)
            overlay_save = crop_rgb_by_box(overlay, crop_box)

            raw_path = save_dir / "lpot_proxy_prob_fused_raw.png"
            heat_path = save_dir / "lpot_proxy_prob_fused_heat.jpg"
            overlay_path = save_dir / "lpot_proxy_prob_fused_overlay.jpg"

            save_gray(raw_path, fused_save)
            save_rgb(heat_path, heat_rgb_save)
            save_rgb(overlay_path, overlay_save)

            feature_files["lpot_proxy_prob_fused"] = {
                "raw_path": str(raw_path),
                "gray_path": str(raw_path),
                "heat_path": str(heat_path),
                "heat_color_path": str(heat_path),
                "overlay_path": str(overlay_path),
                "source": "fused_lpot_internal_proxy_probability",
                "fused_from": prob_roles,
                "note": "Mean fusion of internal LPOT proxy probabilities across available proxy layers.",
                "crop_padding_visuals": bool(crop_box is not None),
            }

    return feature_files, missing


# =========================================================
# Prediction helpers
# =========================================================
def _device_to_predict_arg(device: torch.device) -> str:
    if device.type == "cuda":
        return str(device.index) if device.index is not None else "0"
    return "cpu"


def _get_model_names(yolo) -> Dict[int, str]:
    names = getattr(yolo.model, "names", None)
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def draw_predictions(
    image_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    boxes_conf: np.ndarray,
    boxes_cls: np.ndarray,
    class_names: Dict[int, str],
    line_width: int = 2,
) -> np.ndarray:
    canvas = image_rgb.copy()
    h, w = canvas.shape[:2]

    for box, conf, cls_id in zip(boxes_xyxy, boxes_conf, boxes_cls):
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        cls_id_int = int(cls_id)
        label_name = class_names.get(cls_id_int, str(cls_id_int))
        label = f"{label_name} {float(conf):.2f}"

        color = (0, 255, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_width)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        text_y1 = max(0, y1 - th - baseline - 6)
        text_y2 = text_y1 + th + baseline + 6
        text_x2 = min(w - 1, x1 + tw + 10)

        cv2.rectangle(canvas, (x1, text_y1), (text_x2, text_y2), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 5, text_y2 - baseline - 3),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    return canvas


def _draw_label_box(
    canvas: np.ndarray,
    box_xyxy: Sequence[float],
    label: str,
    color: Tuple[int, int, int],
    line_width: int = 2,
    font_scale: float = 0.55,
) -> None:
    h, w = canvas.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_width)

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    text_y1 = max(0, y1 - th - baseline - 6)
    text_y2 = text_y1 + th + baseline + 6
    text_x2 = min(w - 1, x1 + tw + 10)

    cv2.rectangle(canvas, (x1, text_y1), (text_x2, text_y2), color, -1)
    cv2.putText(
        canvas,
        label,
        (x1 + 5, text_y2 - baseline - 3),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def load_yolo_label_boxes(
    label_path: str | Path,
    image_hw: Tuple[int, int],
    class_names: Dict[int, str],
) -> Tuple[List[Dict[str, Any]], bool]:
    label_path = Path(label_path)
    if not label_path.exists():
        return [], False

    h, w = image_hw
    boxes: List[Dict[str, Any]] = []

    for line_idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        vals = line.strip().split()
        if len(vals) < 5:
            continue

        try:
            cls_id = int(float(vals[0]))
            xc, yc, bw, bh = [float(v) for v in vals[1:5]]
        except Exception:
            continue

        x1 = (xc - bw / 2.0) * w
        y1 = (yc - bh / 2.0) * h
        x2 = (xc + bw / 2.0) * w
        y2 = (yc + bh / 2.0) * h

        x1 = max(0.0, min(x1, float(w - 1)))
        x2 = max(0.0, min(x2, float(w - 1)))
        y1 = max(0.0, min(y1, float(h - 1)))
        y2 = max(0.0, min(y2, float(h - 1)))

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        boxes.append({
            "gt_id": int(line_idx),
            "xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "cls_id": int(cls_id),
            "cls_name": class_names.get(int(cls_id), str(cls_id)),
        })

    return boxes, True


def draw_ground_truth_boxes(
    image_rgb: np.ndarray,
    gt_boxes: Sequence[Dict[str, Any]],
    line_width: int = 2,
) -> np.ndarray:
    canvas = image_rgb.copy()
    color = (255, 0, 0)
    for gt in gt_boxes:
        label = f"GT {gt.get('cls_name', gt.get('cls_id', ''))}"
        _draw_label_box(canvas, gt["xyxy"], label, color=color, line_width=line_width)
    return canvas


def add_panel_title(image_rgb: np.ndarray, title: str, title_height: int = 40) -> np.ndarray:
    canvas = np.ones((image_rgb.shape[0] + title_height, image_rgb.shape[1], 3), dtype=np.uint8) * 255
    canvas[title_height:, :, :] = image_rgb
    cv2.putText(canvas, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas


def make_prediction_annotation_comparison(pred_rgb: np.ndarray, ann_rgb: np.ndarray) -> np.ndarray:
    if ann_rgb.shape[:2] != pred_rgb.shape[:2]:
        ann_rgb = cv2.resize(ann_rgb, (pred_rgb.shape[1], pred_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    left = add_panel_title(pred_rgb, "Prediction")
    right = add_panel_title(ann_rgb, "Ground truth annotation")
    return np.concatenate([left, right], axis=1)


def run_and_save_prediction(
    yolo,
    image_path: str | Path,
    original_rgb: np.ndarray,
    save_path: str | Path,
    meta_path: str | Path,
    imgsz: int,
    device: torch.device,
    pred_conf: float,
    pred_iou: float,
):
    results = yolo.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=pred_conf,
        iou=pred_iou,
        device=_device_to_predict_arg(device),
        verbose=False,
        stream=False,
    )

    result = results[0]
    names = _get_model_names(yolo)

    if result.boxes is None or len(result.boxes) == 0:
        pred_rgb = original_rgb.copy()
        pred_meta = {
            "num_boxes": 0,
            "boxes": [],
            "pred_conf": float(pred_conf),
            "pred_iou": float(pred_iou),
        }
    else:
        boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
        boxes_conf = result.boxes.conf.detach().cpu().numpy()
        boxes_cls = result.boxes.cls.detach().cpu().numpy()

        pred_rgb = draw_predictions(original_rgb, boxes_xyxy, boxes_conf, boxes_cls, names)

        pred_meta = {
            "num_boxes": int(len(boxes_xyxy)),
            "boxes": [
                {
                    "xyxy": [float(v) for v in box.tolist()],
                    "conf": float(conf),
                    "cls_id": int(cls_id),
                    "cls_name": names.get(int(cls_id), str(int(cls_id))),
                }
                for box, conf, cls_id in zip(boxes_xyxy, boxes_conf, boxes_cls)
            ],
            "pred_conf": float(pred_conf),
            "pred_iou": float(pred_iou),
        }

    save_rgb(save_path, pred_rgb)
    write_json(meta_path, pred_meta)

    return pred_rgb, pred_meta


# =========================================================
# Stage1 teacher prior helpers
# =========================================================
def build_stage1_model_from_cfg(cfg: dict, device: torch.device) -> Stage1SslModel:
    cfg = validate_stage1_config(cfg)

    aux_cfg = cfg.get("loss", {}).get("aux_embedding", {})
    runtime_cfg = cfg.get("runtime", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    model_args = model_cfg.get("model_args", {}) or {}

    yolo_model = model_args.get("yolo_model", model_cfg.get("yolo_model"))
    nc = model_args.get("nc", model_cfg.get("nc", None))
    layer_indices = tuple(model_args.get("layer_indices", model_cfg.get("layer_indices", [4, 6, 8])))
    sppf_indice = model_args.get("sppf_indice", model_cfg.get("sppf_indice", 9))

    model = Stage1SslModel(
        yolo_model=yolo_model,
        nc=nc,
        layer_indices=layer_indices,
        image_size=data_cfg.get("image_size", data_cfg.get("imgsz", 640)),
        proj_dim=model_cfg.get("proj_dim", 256),
        local_dim=model_cfg.get("local_dim", 128),
        queue_size=model_cfg.get("queue_size", 4096),
        momentum=model_cfg.get("momentum", 0.999),
        sppf_indice=sppf_indice,
        use_pos=runtime_cfg.get("use_pos", True),
        pos_pe_channels=model_cfg.get("pos_pe_channels", 64),
        pos_pe_spans=model_cfg.get("pos_pe_spans", [1, 1, 1]),
        scale_weight_before=model_cfg.get("scale_weight_before", True),
        pos_init_scales=model_cfg.get("pos_init_scales", [0.1, 0.5, 1.0]),
        pos_enable_fg_guidance=model_cfg.get("pos_enable_fg_guidance", True),
        pos_fg_gate_init=model_cfg.get("pos_fg_gate_init", 1.0),
        enable_raw_projection=aux_cfg.get("enabled", False),
        separate_projector=model_cfg.get("separate_projector", False),
        use_snapshot_teacher=runtime_cfg.get(
            "needs_snapshot_teacher",
            model_cfg.get("snapshot_teacher", {}).get("enabled", False),
        ),
    ).to(device)

    model.eval()
    return model


def load_stage1_teacher(
    teacher_cfg_path: Path,
    teacher_ckpt_path: Path,
    device: torch.device,
) -> Tuple[Optional[Stage1SslModel], Optional[dict]]:
    if not teacher_cfg_path.exists() or not teacher_ckpt_path.exists():
        return None, None

    cfg = validate_stage1_config(load_config(teacher_cfg_path))
    model = build_stage1_model_from_cfg(cfg, device)

    ckpt = torch.load(str(teacher_ckpt_path), map_location=device)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, cfg


def build_teacher_preprocess(image_size: int):
    base_transform = build_base_transform(image_size=image_size)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return base_transform, mean, std


def load_teacher_input(
    image_path: str | Path,
    mask_path: Optional[Path],
    teacher_cfg: dict,
    device: torch.device,
):
    image_size = int(teacher_cfg.get("data", {}).get("image_size", 640))
    base_transform, mean, std = build_teacher_preprocess(image_size)

    img = Image.open(str(image_path)).convert("RGB")

    if mask_path is not None and mask_path.exists():
        mask = Image.open(str(mask_path)).convert("L")
        if mask.size != img.size:
            mask = mask.resize(img.size, Image.NEAREST)

        mask_np = (
            np.array(mask, dtype=np.uint8)
            > int(teacher_cfg.get("data", {}).get("external_mask_threshold", 127))
        ).astype(np.uint8) * 255

        mask = Image.fromarray(mask_np, mode="L")
    else:
        mask = Image.fromarray(
            np.ones((img.size[1], img.size[0]), dtype=np.uint8) * 255,
            mode="L",
        )

    rgba = img.copy()
    rgba.putalpha(mask)

    rgba_tensor = base_transform(rgba)
    rgb_tensor = rgba_tensor[:3]
    mask_tensor = (rgba_tensor[3:].float() > 0.5).float()

    x = ((rgb_tensor - mean) / std).unsqueeze(0).to(device)
    m = mask_tensor.unsqueeze(0).to(device)

    return x, m


@torch.no_grad()
def save_teacher_prior_maps(
    sample_dir: Path,
    image_rgb: np.ndarray,
    image_path: str | Path,
    mask_path: Optional[Path],
    teacher_model: Optional[Stage1SslModel],
    teacher_cfg: Optional[dict],
    teacher_branch: str,
    teacher_feature_indices: Sequence[int],
    device: torch.device,
    crop_box: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, Dict[str, Any]]:
    feature_files: Dict[str, Dict[str, Any]] = {}

    if teacher_model is None or teacher_cfg is None:
        return feature_files

    x, m = load_teacher_input(image_path, mask_path, teacher_cfg, device)
    fg_mask = m if teacher_cfg.get("runtime", {}).get("use_mask", True) else None

    outputs = teacher_model.online(x, fg_mask=fg_mask)
    feats = outputs.get(teacher_branch, None)

    if feats is None:
        return feature_files

    h, w = image_rgb.shape[:2]
    heatmaps: List[np.ndarray] = []

    for _, feat_i in enumerate(teacher_feature_indices):
        if feat_i < 0 or feat_i >= len(feats):
            continue

        heat = feature_to_heatmap(feats[feat_i], out_size=(w, h))
        heat_rgb, overlay = apply_colormap_overlay(image_rgb, heat, alpha=0.45)

        heat_save = crop_gray_by_box(heat, crop_box)
        heat_rgb_save = crop_rgb_by_box(heat_rgb, crop_box)
        overlay_save = crop_rgb_by_box(overlay, crop_box)

        role = f"teacher_prior_l{feat_i + 1}"

        raw_path = sample_dir / f"{role}_raw.png"
        heat_path = sample_dir / f"{role}_heat.jpg"
        overlay_path = sample_dir / f"{role}_overlay.jpg"

        save_gray(raw_path, heat_save)
        save_rgb(heat_path, heat_rgb_save)
        save_rgb(overlay_path, overlay_save)

        heatmaps.append(heat)

        feature_files[role] = {
            "raw_path": str(raw_path),
            "gray_path": str(raw_path),
            "heat_path": str(heat_path),
            "heat_color_path": str(heat_path),
            "overlay_path": str(overlay_path),
            "source": f"stage1_teacher_{teacher_branch}",
            "teacher_feature_index": int(feat_i),
            "crop_padding_visuals": bool(crop_box is not None),
        }

    fused = fuse_heat_uint8(heatmaps, out_size=(w, h))

    if fused is not None:
        heat_rgb, overlay = apply_colormap_overlay(image_rgb, fused, alpha=0.45)

        fused_save = crop_gray_by_box(fused, crop_box)
        heat_rgb_save = crop_rgb_by_box(heat_rgb, crop_box)
        overlay_save = crop_rgb_by_box(overlay, crop_box)

        raw_path = sample_dir / "teacher_prior_fused_raw.png"
        heat_path = sample_dir / "teacher_prior_fused_heat.jpg"
        overlay_path = sample_dir / "teacher_prior_fused_overlay.jpg"

        save_gray(raw_path, fused_save)
        save_rgb(heat_path, heat_rgb_save)
        save_rgb(overlay_path, overlay_save)

        feature_files["teacher_prior_fused"] = {
            "raw_path": str(raw_path),
            "gray_path": str(raw_path),
            "heat_path": str(heat_path),
            "heat_color_path": str(heat_path),
            "overlay_path": str(overlay_path),
            "source": f"stage1_teacher_{teacher_branch}_fused",
            "teacher_feature_indices": [int(x) for x in teacher_feature_indices],
            "crop_padding_visuals": bool(crop_box is not None),
        }

    return feature_files


# =========================================================
# Main visualization
# =========================================================
def default_output_dir(exp_dir: Path) -> Path:
    return exp_dir / "layer_feature_maps"


def get_ckpt_path(exp_dir: Path, ckpt_name: str) -> Path:
    p = exp_dir / "weights" / ckpt_name
    if p.exists():
        return p

    if ckpt_name != "last.pt":
        fallback = exp_dir / "weights" / "last.pt"
        if fallback.exists():
            return fallback

    raise FileNotFoundError(f"Checkpoint not found under {exp_dir / 'weights'}: {ckpt_name}")


def infer_device(cfg: dict) -> torch.device:
    wanted = str(cfg.get("train", {}).get("device", "cuda"))
    if wanted != "cpu" and torch.cuda.is_available():
        return torch.device(wanted)
    return torch.device("cpu")


def infer_imgsz(cfg: dict) -> int:
    return int(cfg.get("data", {}).get("imgsz", cfg.get("data", {}).get("image_size", 640)))


def infer_data_yaml(cfg: dict, data_yaml, config_base_dir: Path) -> Path:
    data_yaml = data_yaml if data_yaml else cfg.get("data", {}).get("data_yaml", None)

    if not data_yaml:
        raise KeyError("Stage2 config must contain data.data_yaml")

    return resolve_path(str(data_yaml), config_base_dir)


def resolve_teacher_paths(
    cfg: dict,
    cli_teacher_ckpt: str,
    cli_teacher_ssl_config: str,
) -> Tuple[Optional[Path], Optional[Path], str, List[int]]:
    stage1_cfg = cfg.get("stage1_init", {}) or {}
    leafaux_cfg = cfg.get("leaf_prior_auxiliary", {}) or {}
    spatial_cfg = cfg.get("spatial_alignment", {}) or {}
    fpd_cfg = cfg.get("foreground_prior_distillation", {}) or {}
    lpot_cfg = cfg.get("leaf_prior_objectness_transfer", {}) or {}

    teacher_cfg = (
        lpot_cfg if lpot_cfg.get("enabled", False) else
        leafaux_cfg if leafaux_cfg.get("enabled", False) else
        fpd_cfg if fpd_cfg.get("enabled", False) else
        spatial_cfg if spatial_cfg.get("enabled", False) else
        {}
    )
    print("="*80)
    print(f"teacher_cfg         : {teacher_cfg}")
    print(f"stage1_cfg          : {stage1_cfg}")
    print("="*80)

    teacher_ckpt = cli_teacher_ckpt or teacher_cfg.get("teacher_ckpt_path", None) or stage1_cfg.get("ckpt_path", None)
    teacher_ssl = cli_teacher_ssl_config or teacher_cfg.get("teacher_ssl_config", None) or stage1_cfg.get("ssl_config", None)
    teacher_branch = str(teacher_cfg.get("teacher_branch", "pos_feats"))

    teacher_indices = teacher_cfg.get("teacher_feature_indices", [0, 1, 2])
    teacher_indices = [int(x) for x in teacher_indices]

    ckpt_path = resolve_path(teacher_ckpt, Path(PROJECT_ROOT)) if teacher_ckpt else None
    cfg_path = resolve_path(teacher_ssl, Path(PROJECT_ROOT)) if teacher_ssl else None

    return ckpt_path, cfg_path, teacher_branch, teacher_indices


@torch.no_grad()
def visualize_one_experiment(
    exp_dir: str | Path,
    base_cfg: dict,
    device: torch.device,
    num_images: int = 12,
    ckpt_name: str = "best.pt",
    layer_indices: Sequence[int] = None,
    data_yaml=None,
    dataset_split=None,
    pred_conf: float = 0.25,
    pred_iou: float = 0.70,
    mask_root: Optional[Path] = None,
    teacher_ckpt_path: str = "",
    teacher_ssl_config: str = "",
    crop_padding_visuals: bool = False,
):
    exp_dir = Path(exp_dir)
    cfg = build_cfg_for_experiment(base_cfg, exp_dir)

    data_yaml_path = infer_data_yaml(cfg, data_yaml, Path(PROJECT_ROOT))
    dataset_root = infer_dataset_root_from_data_yaml(data_yaml_path)

    dataset_splits = parse_dataset_splits(dataset_split)

    split_items = []
    split_roots: Dict[str, Path] = {}

    for split_name in dataset_splits:
        image_root = resolve_dataset_split_dir(data_yaml_path, dataset_split=split_name)
        images = evenly_sample(iter_images(image_root), num_images)
        split_roots[split_name] = image_root
        split_items.extend(
            (split_name, image_root, local_idx, image_path)
            for local_idx, image_path in enumerate(images)
        )

    imgsz = infer_imgsz(cfg)
    ckpt_path = get_ckpt_path(exp_dir, ckpt_name)
    ckpt_path = Path(ckpt_path).resolve()

    save_root = default_output_dir(exp_dir)
    save_root = Path(save_root).resolve()
    ensure_dir(save_root)

    teacher_ckpt, teacher_cfg_path, teacher_branch, teacher_indices = resolve_teacher_paths(
        cfg,
        teacher_ckpt_path,
        teacher_ssl_config,
    )
    print("="*80)
    print(f"teacher_ckpt_path        : {teacher_ckpt_path}")
    print(f"teacher_ssl_config       : {teacher_ssl_config}")
    print(f"teacher_ckpt             : {teacher_ckpt}")
    print(f"teacher_cfg_path         : {teacher_cfg_path}")
    print(f"teacher_branch           : {teacher_branch}")
    print(f"teacher_indices          : {teacher_indices}")
    print("="*80)

    teacher_model, teacher_cfg = (None, None)

    if teacher_ckpt is not None and teacher_cfg_path is not None and teacher_ckpt.exists() and teacher_cfg_path.exists():
        try:
            teacher_model, teacher_cfg = load_stage1_teacher(teacher_cfg_path, teacher_ckpt, device)
            print(f"       teacher   : {teacher_ckpt}")
        except Exception as e:
            print(f"[WARN] failed to load Stage1 teacher for {exp_dir.name}: {e}")
            teacher_model, teacher_cfg = None, None
    else:
        print(f"[WARN] Stage1 teacher not found for {exp_dir.name}; teacher prior maps will be skipped.")

    print("=" * 80)
    print(f"[INFO] Visualizing experiment: {exp_dir.name}")
    print(f"       checkpoint: {ckpt_path}")
    print(f"       splits    : {dataset_splits}")
    print(f"       image_roots: { {k: str(v) for k, v in split_roots.items()} }")
    print(f"       layers    : {list(layer_indices)}")
    print(f"       images    : {len(split_items)}")
    print(f"       num_images: {num_images} ({'all per split' if int(num_images) < 0 else 'evenly sampled per split'})")
    print(f"       imgsz     : {imgsz}")
    print(f"       crop_pad  : {crop_padding_visuals}")
    print(f"       save_root : {save_root}")
    print(f"       mask_root : {mask_root}")
    print(f"    teacher_ckpt : {teacher_ckpt}")
    print(f"teacher_cfg_path : {teacher_cfg_path}")
    print("=" * 80)

    yolo = YOLO(str(ckpt_path))
    det_model = yolo.model.to(device)
    det_model.eval()

    cache: Dict[int, torch.Tensor] = {}
    handles = build_hooks(det_model, layer_indices, cache)

    sidecar_proxy_heads = load_lpot_proxy_heads_from_sidecar(exp_dir, device)

    metadata: Dict[str, Any] = {
        "experiment": exp_dir.name,
        "checkpoint": str(ckpt_path),
        "dataset_splits": dataset_splits,
        "split_roots": {k: str(v) for k, v in split_roots.items()},
        "dataset_root": str(dataset_root),
        "data_yaml_path": str(data_yaml_path),
        "imgsz": imgsz,
        "device": str(device),
        "layers": list(layer_indices),
        "layer_roles": {str(i): infer_layer_role(i) for i in layer_indices},
        "num_images_arg": int(num_images),
        "num_images": len(split_items),
        "pred_conf": float(pred_conf),
        "pred_iou": float(pred_iou),
        "mask_root": str(mask_root) if mask_root else "",
        "teacher_ckpt_path": str(teacher_ckpt) if teacher_ckpt else "",
        "teacher_ssl_config": str(teacher_cfg_path) if teacher_cfg_path else "",
        "teacher_branch": teacher_branch,
        "teacher_feature_indices": teacher_indices,
        "crop_padding_visuals": bool(crop_padding_visuals),
        "samples": [],
        "notes": [
            "Feature visualization input uses YOLO-style letterbox: keep aspect ratio, resize long side to imgsz, then center-pad to imgsz x imgsz.",
            "If crop_padding_visuals=True, saved feature maps and overlays are cropped to the non-padded resized region.",
            "prehead_support_fused is a feature-based support-like map from pre-head feature heatmaps.",
            "lpot_proxy_prob_* maps are internal LPOT proxy-head sigmoid probabilities when _lpot_proxy_head_* modules exist in the checkpoint.",
            "If LPOT proxy heads are absent from the checkpoint, lpot_proxy_missing in sample_meta.json records the reason.",
        ],
    }

    try:
        summary = []
        multi_split = len(dataset_splits) > 1

        for idx, (split_name, image_root, local_idx, image_path) in enumerate(split_items):
            sample_id = f"{split_name}_sample_{local_idx:05d}" if multi_split else f"sample_{local_idx:03d}"
            sample_dir = ensure_dir(save_root / sample_id)

            image_path_p = Path(image_path).resolve()
            label_path = image_to_label_path(image_path_p)

            mask_path, mask_candidates = find_mask_path(
                image_path_p,
                mask_root=mask_root,
                image_root=image_root,
                dataset_root=dataset_root,
            )

            orig_rgb, model_rgb, visual_rgb, crop_box, x, resize_info = preprocess_image(
                image_path_p,
                imgsz,
                device,
                crop_padding_visuals=crop_padding_visuals,
            )

            crop_box_for_save = crop_box if crop_padding_visuals else None

            cache.clear()
            _ = det_model(x)

            input_original_path = sample_dir / "input_original.jpg"
            input_resized_path = sample_dir / "input_resized.jpg"
            input_model_square_path = sample_dir / "input_model_square.jpg"

            save_rgb(input_original_path, orig_rgb)
            save_rgb(input_resized_path, visual_rgb)
            save_rgb(input_model_square_path, model_rgb)

            if mask_path is not None and mask_path.exists():
                m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    m_model = letterbox_gray_like_image(
                        m,
                        imgsz=imgsz,
                        crop_box=crop_box,
                        pad_value=0,
                    )
                    m_save = crop_gray_by_box(m_model, crop_box_for_save)
                    save_gray(sample_dir / "mask_resized.png", m_save)

            pred_rgb, pred_meta = run_and_save_prediction(
                yolo=yolo,
                image_path=image_path_p,
                original_rgb=orig_rgb,
                save_path=sample_dir / "prediction.jpg",
                meta_path=sample_dir / "prediction_meta.json",
                imgsz=imgsz,
                device=device,
                pred_conf=pred_conf,
                pred_iou=pred_iou,
            )

            class_names = _get_model_names(yolo)

            gt_boxes, label_exists_for_viz = load_yolo_label_boxes(
                label_path=label_path,
                image_hw=orig_rgb.shape[:2],
                class_names=class_names,
            )

            annotation_rgb = draw_ground_truth_boxes(orig_rgb, gt_boxes)

            annotation_image_path = sample_dir / "annotation_bbox.jpg"
            annotation_meta_path = sample_dir / "annotation_bbox_meta.json"
            pred_vs_annotation_path = sample_dir / "prediction_vs_annotation.jpg"

            save_rgb(annotation_image_path, annotation_rgb)
            write_json(annotation_meta_path, {
                "label_path": str(label_path),
                "label_exists": bool(label_exists_for_viz),
                "num_gt_boxes": int(len(gt_boxes)),
                "boxes": gt_boxes,
            })

            pred_vs_annotation_rgb = make_prediction_annotation_comparison(pred_rgb, annotation_rgb)
            save_rgb(pred_vs_annotation_path, pred_vs_annotation_rgb)

            missing = [i for i in layer_indices if i not in cache]
            if missing:
                raise KeyError(f"Missing captured layers {missing}; captured={sorted(cache.keys())}")

            layer_shapes = {str(i): list(cache[i].shape) for i in layer_indices}
            feature_diagnostic_stats = collect_feature_diagnostic_stats(cache, layer_indices)
            feature_diagnostic_stats_path = sample_dir / "feature_stats_diagnostic.json"
            write_json(feature_diagnostic_stats_path, feature_diagnostic_stats)

            pred_model_rgb, _, _, _ = letterbox_rgb(
                pred_rgb,
                imgsz=imgsz,
                pad_value=114,
            )

            feature_files = save_spatial_feature_group(
                image_rgb=model_rgb,
                feat_map_dict=cache,
                layer_indices=layer_indices,
                save_dir=sample_dir,
                prediction_rgb=pred_model_rgb,
                crop_box=crop_box_for_save,
                annotation_rgb=annotation_rgb,
            )

            lpot_cfg = cfg.get("leaf_prior_objectness_transfer", {}) or {}
            proxy_layer_indices = [int(x) for x in lpot_cfg.get("proxy_layer_indices", [15, 18, 21])]

            lpot_proxy_feature_files, lpot_proxy_missing = save_lpot_proxy_probability_maps(
                image_rgb=model_rgb,
                feat_map_dict=cache,
                det_model=det_model,
                proxy_layer_indices=proxy_layer_indices,
                save_dir=sample_dir,
                sidecar_proxy_heads=sidecar_proxy_heads,
                crop_box=crop_box_for_save,
            )
            feature_files.update(lpot_proxy_feature_files)

            teacher_feature_files = save_teacher_prior_maps(
                sample_dir=sample_dir,
                image_rgb=model_rgb,
                image_path=image_path_p,
                mask_path=mask_path,
                teacher_model=teacher_model,
                teacher_cfg=teacher_cfg,
                teacher_branch=teacher_branch,
                teacher_feature_indices=teacher_indices,
                device=device,
                crop_box=crop_box_for_save,
            )
            feature_files.update(teacher_feature_files)

            try:
                image_rel_path = str(image_path_p.relative_to(image_root))
            except Exception:
                image_rel_path = image_path_p.name

            try:
                dataset_rel_path = str(image_path_p.relative_to(dataset_root))
            except Exception:
                dataset_rel_path = image_rel_path

            sample_meta = {
                "sample_id": sample_id,
                "sample_index": idx,
                "dataset_split": split_name,
                "experiment": exp_dir.name,
                "image_path": str(image_path_p),
                "image_root": str(image_root),
                "image_rel_path": image_rel_path,
                "dataset_root": str(dataset_root),
                "dataset_rel_path": dataset_rel_path,
                "image_name": image_path_p.name,
                "image_stem": image_path_p.stem,
                "label_path": str(label_path),
                "label_exists": bool(label_path.exists()),
                "mask_root": str(mask_root) if mask_root else "",
                "expected_mask_paths": mask_candidates,
                "mask_path": str(mask_path) if mask_path else "",
                "mask_exists": bool(mask_path is not None and mask_path.exists()),
                "input_original_path": str(input_original_path),
                "input_resized_path": str(input_resized_path),
                "input_model_square_path": str(input_model_square_path),
                "prediction_image_path": str(sample_dir / "prediction.jpg"),
                "prediction_meta_path": str(sample_dir / "prediction_meta.json"),
                "annotation_image_path": str(annotation_image_path),
                "annotation_meta_path": str(annotation_meta_path),
                "prediction_vs_annotation_path": str(pred_vs_annotation_path),
                "overview_path": str(sample_dir / "overview.jpg"),
                "original_hw": list(orig_rgb.shape[:2]),
                "resized_hw": list(visual_rgb.shape[:2]),
                "model_input_hw": list(model_rgb.shape[:2]),
                "resize_info": resize_info,
                "crop_padding_visuals": bool(crop_padding_visuals),
                "layer_shapes": layer_shapes,
                "feature_diagnostic_stats_path": str(feature_diagnostic_stats_path),
                "prediction": pred_meta,
                "feature_files": feature_files,
                "lpot_proxy_missing": lpot_proxy_missing,
            }

            write_json(sample_dir / "sample_meta.json", sample_meta)

            summary.append({
                "sample_id": sample_id,
                "sample_index": idx,
                "dataset_split": split_name,
                "image_path": str(image_path_p),
                "image_rel_path": image_rel_path,
                "mask_path": sample_meta["mask_path"],
                "mask_exists": sample_meta["mask_exists"],
                "label_path": str(label_path),
                "annotation_image_path": str(annotation_image_path),
                "prediction_vs_annotation_path": str(pred_vs_annotation_path),
                "num_gt_boxes": int(len(gt_boxes)),
                "prediction_num_boxes": pred_meta["num_boxes"],
                "crop_padding_visuals": bool(crop_padding_visuals),
                "resize_info": resize_info,
                "feature_files": feature_files,
            })

            print(
                f"[OK] {sample_id}: {image_path_p.name} | "
                f"boxes={pred_meta['num_boxes']} | "
                f"mask={sample_meta['mask_exists']} | "
                f"crop_pad={crop_padding_visuals}"
            )

        metadata["samples"] = summary
        write_json(save_root / "meta.json", metadata)
        write_json(save_root / "summary.json", summary)

    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="./configs/det_config.yaml")
    parser.add_argument("--runs_dir", type=str, default="./runs/glcp_stage2_yolo_det")
    parser.add_argument(
        "--exp_names",
        type=str,
        nargs="+",
        default=[
            "baseline",
            "full_no_freeze_use_pos_mask_sw_lesion_sensitive",
            "leafaux_best_use_pos_mask_rpd_hybrid_w020",
        ],
    )
    parser.add_argument("--exp_target", type=str, default=None)
    parser.add_argument(
        "--num_images",
        type=int,
        default=12,
        help="Images per split. Use -1 to visualize all images in each split.",
    )
    parser.add_argument("--ckpt_name", type=str, default="best.pt")
    parser.add_argument("--data_yaml", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, nargs="+", default=["val"], help="One or more dataset splits")
    parser.add_argument("--layers", type=str, default=None)
    parser.add_argument("--pred_conf", type=float, default=0.30)
    parser.add_argument("--pred_iou", type=float, default=0.40)
    parser.add_argument("--mask_root", type=str, default="./data/unlabeled_train/foreground_masks")
    parser.add_argument("--teacher_ckpt_path", type=str, default="")
    parser.add_argument("--teacher_ssl_config", type=str, default="")
    parser.add_argument(
        "--crop_padding_visuals",
        action="store_true",
        help=(
            "Crop padding from saved visualization images. "
            "The model input still uses YOLO-style letterbox imgsz x imgsz, "
            "but saved feature heatmaps, overlays, masks, and overview images are cropped "
            "to the non-padded resized region whose long side equals imgsz."
        ),
    )

    args = parser.parse_args()
    args.crop_padding_visuals = True

    base_cfg = load_config(args.config)
    device = infer_device(base_cfg)

    if not args.teacher_ckpt_path:
        args.teacher_ckpt_path = resolve_path(base_cfg.get("stage1_init", {}).get("ckpt_path", None), Path(PROJECT_ROOT))
    if not args.teacher_ssl_config:
        args.teacher_ssl_config = resolve_path(base_cfg.get("stage1_init", {}).get("ssl_config", None), Path(PROJECT_ROOT))

    if args.layers is None or args.layers == "":
        yolo_model = base_cfg["model"]["yolo_model"]
        args.layers = ",".join(map(str, DEFAULT_LAYERS[yolo_model]))

    layer_indices = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    if args.exp_target is None:
        args.exp_target = base_cfg["train"]["name"]

    exp_list = args.exp_names if args.exp_target == "all" else [args.exp_target]
    mask_root = resolve_path(args.mask_root, Path(PROJECT_ROOT)) if args.mask_root else None

    for exp_name in exp_list:
        exp_dir = os.path.join(args.runs_dir, exp_name)

        if not os.path.isdir(exp_dir):
            print(f"[WARN] exp dir not found: {exp_dir}")
            continue

        visualize_one_experiment(
            exp_dir=exp_dir,
            base_cfg=base_cfg,
            device=device,
            num_images=args.num_images,
            ckpt_name=args.ckpt_name,
            layer_indices=layer_indices,
            data_yaml=args.data_yaml,
            dataset_split=args.dataset_split,
            pred_conf=args.pred_conf,
            pred_iou=args.pred_iou,
            mask_root=mask_root,
            teacher_ckpt_path=args.teacher_ckpt_path,
            teacher_ssl_config=args.teacher_ssl_config,
            crop_padding_visuals=args.crop_padding_visuals,
        )

        print(f"[INFO] Visualizing experiment({exp_name}) Done.")


if __name__ == "__main__":
    try:
        main()
    finally:
        from utils.utils import cleanup_memory
        cleanup_memory()