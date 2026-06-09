from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_str: str | Path, base: Optional[Path] = None) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return ((base or PROJECT_ROOT) / p).resolve()


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def existing_path(value: Any) -> Optional[Path]:
    if not value:
        return None
    p = Path(str(value))
    return p.resolve() if p.exists() else None


def load_gray01(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f'Failed to read image: {path}')
    return np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0)


def load_mask01(path: str | Path, out_hw: Tuple[int, int]) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f'Failed to read mask: {path}')
    if m.shape[:2] != out_hw:
        m = cv2.resize(m, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (m.astype(np.float32) > 127.0).astype(np.uint8)


def build_mask_index(mask_root: str | Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    root = Path(mask_root)
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            index.setdefault(p.stem, []).append(p.resolve())
    return index


def read_sample_meta(sample_dir: Path) -> dict:
    p = sample_dir / 'sample_meta.json'
    if p.exists():
        try:
            return read_json(p)
        except Exception:
            return {}
    return {}


def read_root_meta(fmap_root: Path) -> dict:
    p = fmap_root / 'meta.json'
    if p.exists():
        try:
            return read_json(p)
        except Exception:
            return {}
    return {}


def find_mask(mask_index: Dict[str, List[Path]], sample_dir: Path, meta: dict, mask_root: Path, allow_sample_mask_fallback: bool = False) -> Optional[Path]:
    for key in ('mask_path', 'expected_mask_path'):
        p = existing_path(meta.get(key))
        if p is not None:
            return p
    for key in ('expected_mask_paths', 'candidate_mask_paths'):
        vals = meta.get(key, []) or []
        if isinstance(vals, str):
            vals = [vals]
        for v in vals:
            p = existing_path(v)
            if p is not None:
                return p
    for rel_key in ('image_rel_path', 'dataset_rel_path'):
        rel = meta.get(rel_key, '')
        if rel:
            rel_no_suffix = str(Path(str(rel)).with_suffix(''))
            for suffix in ('.png', '.jpg', '.jpeg', '.bmp', '.webp'):
                p = (mask_root / f'{rel_no_suffix}{suffix}').resolve()
                if p.exists():
                    return p
    candidates: List[str] = []
    for key in ('image_stem', 'image_name', 'image_path'):
        v = meta.get(key, '')
        if v:
            candidates.append(Path(str(v)).stem)
    candidates.append(sample_dir.name)
    for stem in candidates:
        if stem in mask_index and mask_index[stem]:
            return mask_index[stem][0]
    if allow_sample_mask_fallback:
        for name in ('mask_resized.png', 'mask.png', 'foreground_mask.png'):
            p = sample_dir / name
            if p.exists():
                return p.resolve()
    return None


def read_prediction_meta(sample_dir: Path, meta: dict) -> List[dict]:
    p = existing_path(meta.get('prediction_meta_path')) or (sample_dir / 'prediction_meta.json')
    if not p.exists():
        return []
    data = read_json(p)
    if isinstance(data, list):
        preds = data
    elif isinstance(data, dict):
        preds = data.get('predictions') or data.get('detections') or data.get('boxes') or []
    else:
        preds = []

    out = []
    for i, d in enumerate(preds):
        if not isinstance(d, dict):
            continue
        xyxy = d.get('xyxy') or d.get('box') or d.get('bbox')
        if xyxy is None and all(k in d for k in ['x1', 'y1', 'x2', 'y2']):
            xyxy = [d['x1'], d['y1'], d['x2'], d['y2']]
        if xyxy is None or len(xyxy) != 4:
            continue
        out.append({
            'pred_id': i,
            'xyxy': [float(x) for x in xyxy],
            'conf': float(d.get('conf', d.get('confidence', d.get('score', 0.0)))),
            'cls_id': int(d.get('cls_id', d.get('class_id', d.get('cls', -1)))),
            'cls_name': str(d.get('cls_name', d.get('class_name', d.get('name', '')))),
        })
    return out


def image_hw(sample_dir: Path, meta: dict) -> Tuple[int, int]:
    if isinstance(meta.get('original_hw'), list) and len(meta['original_hw']) == 2:
        return int(meta['original_hw'][0]), int(meta['original_hw'][1])
    for key in ('input_original_path', 'image_path', 'input_resized_path'):
        p = existing_path(meta.get(key))
        if p is not None:
            img = cv2.imread(str(p))
            if img is not None:
                return img.shape[0], img.shape[1]
    for p in [sample_dir / 'input_original.jpg', sample_dir / 'input_resized.jpg']:
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return img.shape[0], img.shape[1]
    return 640, 640


def label_path_from_image(image_path: str) -> Optional[Path]:
    if not image_path:
        return None
    p = Path(image_path)
    parts = list(p.parts)
    for idx, part in enumerate(parts):
        if part == 'images':
            parts[idx] = 'labels'
            return Path(*parts).with_suffix('.txt')
    return p.parent.parent / 'labels' / f'{p.stem}.txt'


def find_label_path(meta: dict) -> Optional[Path]:
    p = existing_path(meta.get('label_path'))
    if p is not None:
        return p
    image_path = str(meta.get('image_path', ''))
    lp = label_path_from_image(image_path)
    return lp if lp is not None and lp.exists() else None


def load_yolo_labels(meta: dict, hw: Tuple[int, int]) -> List[dict]:
    h, w = hw
    lp = find_label_path(meta)
    if lp is None:
        return []
    gts = []
    for line in lp.read_text(encoding='utf-8').splitlines():
        vals = line.strip().split()
        if len(vals) < 5:
            continue
        cls = int(float(vals[0]))
        xc, yc, bw, bh = map(float, vals[1:5])
        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h
        gts.append({'cls_id': cls, 'xyxy': [x1, y1, x2, y2], 'label_path': str(lp)})
    return gts


def box_clip_xyxy(box: Sequence[float], hw: Tuple[int, int]) -> List[float]:
    h, w = hw
    x1, y1, x2, y2 = [float(x) for x in box]
    x1 = max(0.0, min(x1, w - 1.0)); x2 = max(0.0, min(x2, w - 1.0))
    y1 = max(0.0, min(y1, h - 1.0)); y2 = max(0.0, min(y2, h - 1.0))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def box_intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def box_containment_ratio(candidate_box: Sequence[float], kept_box: Sequence[float]) -> float:
    """Fraction of candidate_box area covered by kept_box.

    Used only inside the same predicted class. This does not suppress
    overlapping boxes from different disease classes, so co-occurring
    diseases on the same leaf are preserved.
    """
    area = box_area(candidate_box)
    if area <= 0:
        return 0.0
    return box_intersection_area(candidate_box, kept_box) / area


def box_min_area_overlap(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection divided by the smaller box area.

    This catches same-class duplicate boxes in the common "large box contains
    small box" case where ordinary IoU can be low because the union is large.
    It is used only after class-aware grouping, so cross-disease overlaps are
    still preserved.
    """
    min_area = min(box_area(a), box_area(b))
    if min_area <= 0:
        return 0.0
    return box_intersection_area(a, b) / min_area


def same_class_duplicate_suppression(
    preds: List[dict],
    *,
    iou_thr: float = 0.50,
    containment_thr: float = 0.80,
) -> Tuple[List[dict], List[dict]]:
    """Suppress duplicate predictions only within the same class.

    This is intentionally class-aware: predictions from different disease
    classes are never compared or removed, because one leaf can contain
    multiple diseases. Within each class, lower-confidence boxes are removed
    when they are highly overlapping with, or mostly contained by, a higher-
    confidence box of the same class.
    """
    if not preds:
        return [], []

    kept_all: List[dict] = []
    removed_all: List[dict] = []

    classes = sorted({int(p.get('cls_id', -1)) for p in preds})
    for cls_id in classes:
        cls_preds = [dict(p) for p in preds if int(p.get('cls_id', -1)) == cls_id]
        order = sorted(cls_preds, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
        kept_cls: List[dict] = []

        for pred in order:
            suppress_reason = ''
            for kept in kept_cls:
                iou = box_iou(pred['xyxy'], kept['xyxy'])
                # Candidate-containment: lower-confidence candidate is mostly covered by kept box.
                containment = box_containment_ratio(pred['xyxy'], kept['xyxy'])
                # Reverse-containment: kept box is mostly covered by lower-confidence candidate.
                # This catches a low-confidence large box surrounding a high-confidence small box.
                reverse_containment = box_containment_ratio(kept['xyxy'], pred['xyxy'])
                # Min-area overlap catches both containment directions and near-contained boxes
                # even when ordinary IoU is low due to a large union area.
                min_area_overlap = box_min_area_overlap(pred['xyxy'], kept['xyxy'])

                if iou >= iou_thr:
                    suppress_reason = f'same_class_iou>={iou_thr:.2f}'
                    break
                if containment_thr > 0 and containment >= containment_thr:
                    suppress_reason = f'same_class_candidate_containment>={containment_thr:.2f}'
                    break
                if containment_thr > 0 and reverse_containment >= containment_thr:
                    suppress_reason = f'same_class_reverse_containment>={containment_thr:.2f}'
                    break
                if containment_thr > 0 and min_area_overlap >= containment_thr:
                    suppress_reason = f'same_class_min_area_overlap>={containment_thr:.2f}'
                    break
            if suppress_reason:
                pred = dict(pred)
                pred['suppressed_by_same_class_postprocess'] = True
                pred['suppression_reason'] = suppress_reason
                removed_all.append(pred)
            else:
                kept_cls.append(pred)

        kept_all.extend(kept_cls)

    kept_all = sorted(kept_all, key=lambda x: int(x.get('pred_id', 0)))
    removed_all = sorted(removed_all, key=lambda x: int(x.get('pred_id', 0)))
    return kept_all, removed_all


def infer_dataset_split(sample_id: str, meta: dict | None = None) -> str:
    """Support old sample_000 and new val_sample_00000/test_sample_00000 names."""
    if meta:
        for key in ("dataset_split", "split"):
            v = str(meta.get(key, "")).strip()
            if v:
                return v
    m = re.match(r"^([A-Za-z0-9]+)_sample_\d+", sample_id)
    return m.group(1) if m else "unknown"


def is_sample_dir(path: Path) -> bool:
    """Accept sample_000, val_sample_00000, test_sample_00000, etc."""
    return path.is_dir() and re.match(r"^(?:[A-Za-z0-9]+_)?sample_\d+$", path.name) is not None


def iter_sample_dirs(fmap_root: Path) -> List[Path]:
    if not fmap_root.exists():
        return []
    return sorted([p for p in fmap_root.iterdir() if is_sample_dir(p)])



def box_pair_metrics(pred_box: Sequence[float], gt_box: Sequence[float]) -> dict:
    """Return IoU and asymmetric coverage metrics for one prediction/GT pair."""
    inter = box_intersection_area(pred_box, gt_box)
    pred_area = box_area(pred_box)
    gt_area = box_area(gt_box)
    union = pred_area + gt_area - inter
    return {
        'iou': inter / union if union > 0 else 0.0,
        'gt_coverage': inter / gt_area if gt_area > 0 else 0.0,          # GT area covered by prediction
        'pred_coverage': inter / pred_area if pred_area > 0 else 0.0,    # prediction area covered by GT
        'area_ratio': pred_area / gt_area if gt_area > 0 else float('inf'),
        'inter_area': inter,
        'pred_area': pred_area,
        'gt_area': gt_area,
    }


def _empty_match() -> dict:
    return {
        'is_tp': False,
        'is_fp': True,
        'is_annotation_fp': True,
        'is_gt_covered_candidate': False,
        'matched_gt_iou': 0.0,
        'matched_gt_class': -1,
        'matched_gt_coverage': 0.0,
        'matched_pred_coverage': 0.0,
        'matched_area_ratio': float('nan'),
        'match_reason': 'none',
        'matched_gt_index': -1,
        'best_any_gt_iou': 0.0,
        'best_any_gt_class': -1,
        'best_any_gt_coverage': 0.0,
        'best_any_pred_coverage': 0.0,
        'best_any_area_ratio': float('nan'),
    }


def _is_lesion_aware_positive(metrics: dict, *, iou_thr: float, relaxed_iou_thr: float, gt_coverage_thr: float) -> Tuple[bool, str]:
    """A disease-lesion aware box match.

    This keeps the standard IoU rule, but also accepts coarse boxes that cover
    most of a tight GT lesion box. It intentionally does NOT reject large boxes
    by area ratio, because in this dataset a prediction may cover a larger
    diseased leaf region while the annotation is a tight local lesion box.
    """
    if metrics['iou'] >= iou_thr:
        return True, 'strict_iou'
    if gt_coverage_thr > 0 and metrics['gt_coverage'] >= gt_coverage_thr:
        return True, 'gt_coverage'
    if relaxed_iou_thr > 0 and metrics['iou'] >= relaxed_iou_thr and metrics['gt_coverage'] >= max(0.50, gt_coverage_thr * 0.50):
        return True, 'relaxed_iou_with_gt_coverage'
    return False, 'none'


def match_predictions(
    preds: List[dict],
    gts: List[dict],
    iou_thr: float,
    *,
    match_mode: str = 'lesion_aware',
    relaxed_iou_thr: float = 0.35,
    gt_coverage_thr: float = 0.70,
) -> Dict[int, dict]:
    """Match predictions to GT boxes.

    Modes:
      - strict_iou: standard one-to-one detection matching with IoU >= iou_thr.
      - lesion_aware: standard matching plus a GT-coverage rule for tight lesion annotations.

    Important: a prediction that covers a GT already matched by a higher-confidence
    prediction is marked as gt_covered_coarse_or_duplicate, not as annotation FP.
    This prevents large-box-over-small-box cases from being counted as false positives
    while avoiding inflated TP counts.
    """
    matches: Dict[int, dict] = {}
    used_gt = set()
    order = sorted(range(len(preds)), key=lambda i: preds[i]['conf'], reverse=True)
    lesion_mode = str(match_mode).lower() in {'lesion_aware', 'coverage', 'gt_coverage'}

    # Cache same-class metrics for both available-GT matching and duplicate/coarse diagnostics.
    metrics_by_pred: Dict[int, List[Tuple[int, dict]]] = {}
    for pi, p in enumerate(preds):
        pairs = []
        for gi, g in enumerate(gts):
            if p['cls_id'] >= 0 and g['cls_id'] >= 0 and p['cls_id'] != g['cls_id']:
                continue
            pairs.append((gi, box_pair_metrics(p['xyxy'], g['xyxy'])))
        metrics_by_pred[pi] = pairs

    for pi in order:
        best_available = None
        best_available_score = (-1, -1.0, -1.0)  # strict flag / gt coverage / iou

        for gi, metrics in metrics_by_pred.get(pi, []):
            if gi in used_gt:
                continue
            if lesion_mode:
                ok, reason = _is_lesion_aware_positive(
                    metrics,
                    iou_thr=iou_thr,
                    relaxed_iou_thr=relaxed_iou_thr,
                    gt_coverage_thr=gt_coverage_thr,
                )
            else:
                ok = metrics['iou'] >= iou_thr
                reason = 'strict_iou' if ok else 'none'
            if not ok:
                continue
            score = (1 if metrics['iou'] >= iou_thr else 0, metrics['gt_coverage'], metrics['iou'])
            if score > best_available_score:
                best_available_score = score
                best_available = (gi, metrics, reason)

        if best_available is not None:
            gi, metrics, reason = best_available
            used_gt.add(gi)
            matches[pi] = {
                'is_tp': True,
                'is_fp': False,
                'is_annotation_fp': False,
                'is_gt_covered_candidate': False,
                'matched_gt_iou': metrics['iou'],
                'matched_gt_class': gts[gi]['cls_id'],
                'matched_gt_coverage': metrics['gt_coverage'],
                'matched_pred_coverage': metrics['pred_coverage'],
                'matched_area_ratio': metrics['area_ratio'],
                'match_reason': reason,
                'matched_gt_index': gi,
                'best_any_gt_iou': metrics['iou'],
                'best_any_gt_class': gts[gi]['cls_id'],
                'best_any_gt_coverage': metrics['gt_coverage'],
                'best_any_pred_coverage': metrics['pred_coverage'],
                'best_any_area_ratio': metrics['area_ratio'],
            }
            continue

        # No unused GT matched. Check whether this prediction still covers any same-class GT.
        # If so, it is a coarse/duplicate GT-covered candidate rather than annotation FP.
        best_any = None
        best_any_score = (-1.0, -1.0)
        for gi, metrics in metrics_by_pred.get(pi, []):
            score = (metrics['gt_coverage'], metrics['iou'])
            if score > best_any_score:
                best_any_score = score
                best_any = (gi, metrics)

        m = _empty_match()
        if best_any is not None:
            gi, metrics = best_any
            m.update({
                'matched_gt_iou': metrics['iou'],
                'matched_gt_class': gts[gi]['cls_id'],
                'matched_gt_coverage': metrics['gt_coverage'],
                'matched_pred_coverage': metrics['pred_coverage'],
                'matched_area_ratio': metrics['area_ratio'],
                'matched_gt_index': gi,
                'best_any_gt_iou': metrics['iou'],
                'best_any_gt_class': gts[gi]['cls_id'],
                'best_any_gt_coverage': metrics['gt_coverage'],
                'best_any_pred_coverage': metrics['pred_coverage'],
                'best_any_area_ratio': metrics['area_ratio'],
            })
            if lesion_mode:
                ok, reason = _is_lesion_aware_positive(
                    metrics,
                    iou_thr=iou_thr,
                    relaxed_iou_thr=relaxed_iou_thr,
                    gt_coverage_thr=gt_coverage_thr,
                )
                if ok:
                    m.update({
                        'is_fp': False,
                        'is_annotation_fp': False,
                        'is_gt_covered_candidate': True,
                        'match_reason': f'gt_covered_coarse_or_duplicate:{reason}',
                    })
                else:
                    m['match_reason'] = 'none'
        matches[pi] = m

    return matches


def box_mask_overlap(mask: np.ndarray, box: Sequence[float]) -> Tuple[float, bool]:
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = box_clip_xyxy(box, (h, w))
    xi1, yi1, xi2, yi2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    xi2 = max(xi1 + 1, xi2); yi2 = max(yi1 + 1, yi2)
    crop = mask[yi1:yi2, xi1:xi2]
    overlap = float(crop.mean()) if crop.size else 0.0
    cx, cy = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
    cx = max(0, min(cx, w - 1)); cy = max(0, min(cy, h - 1))
    return overlap, bool(mask[cy, cx] > 0)


def find_heat_by_role(sample_dir: Path, role: str, meta: dict | None = None) -> Optional[Path]:
    """
    Resolve the quantitative map for a semantic role.

    Priority:
      1) explicit raw/gray paths recorded in sample_meta.json feature_files
      2) raw/gray/map/prob files in the sample directory
      3) legacy color heatmap files as fallback

    This keeps old visualization outputs usable, while preferring the newly
    saved single-channel raw maps for quantitative analysis.
    """
    role_l = role.lower()

    # 1) Prefer explicit raw/gray paths written by the visualization script.
    if meta and isinstance(meta.get('feature_files'), dict):
        rec = meta['feature_files'].get(role) or meta['feature_files'].get(role_l)
        if isinstance(rec, dict):
            for key in (
                'raw_path',
                'gray_path',
                'heat_raw_path',
                'map_path',
                'prob_path',
                'support_path',
                'prior_path',
                # fallback to display heatmap only if no raw map is present
                'heat_path',
                'heat_color_path',
            ):
                p = existing_path(rec.get(key))
                if p is not None:
                    return p

    # 2) Prefer raw single-channel maps by filename.
    raw_patterns = (
        '*_raw.png', '*_raw.jpg', '*_raw.jpeg',
        '*_gray.png', '*_gray.jpg', '*_gray.jpeg',
        '*_map.png', '*_map.jpg', '*_map.jpeg',
        '*_prob.png', '*_prob.jpg', '*_prob.jpeg',
    )
    for pat in raw_patterns:
        for p in sorted(sample_dir.glob(pat)):
            name = p.name.lower()
            if role_l in name:
                return p

    # 3) Recursive raw-map fallback, useful if later outputs are nested.
    for p in sorted(sample_dir.rglob('*')):
        if not (p.is_file() and p.suffix.lower() in IMG_EXTS):
            continue
        name = p.name.lower()
        if role_l in name and any(k in name for k in ('raw', 'gray', 'map', 'prob')):
            return p

    # 4) Legacy fallback: color heatmaps.
    for pat in ('*_heat.jpg', '*_heat.png', '*_heat.jpeg'):
        for p in sorted(sample_dir.glob(pat)):
            if role_l in p.name.lower():
                return p
    for p in sorted(sample_dir.rglob('*')):
        if p.is_file() and p.suffix.lower() in IMG_EXTS and 'heat' in p.name.lower() and role_l in p.name.lower():
            return p
    return None


def resize_like(x: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    h, w = hw
    if x.shape[:2] == (h, w):
        return x
    return cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR)


def load_map_by_roles(sample_dir: Path, roles: Sequence[str], meta: dict, hw: Tuple[int, int]) -> Optional[np.ndarray]:
    maps = []
    for role in roles:
        hp = find_heat_by_role(sample_dir, role, meta=meta)
        # print(f"load_heat_map: {hp}")
        if hp is not None:
            maps.append(resize_like(load_gray01(hp), hw))
    if not maps:
        return None
    x = np.mean(maps, axis=0).astype(np.float32)
    vmin, vmax = float(np.min(x)), float(np.max(x))
    if vmax > vmin:
        x = (x - vmin) / (vmax - vmin + 1e-8)
    return np.clip(x, 0.0, 1.0)


def box_mean_map(x: Optional[np.ndarray], box: Sequence[float], hw: Tuple[int, int]) -> float:
    if x is None:
        return float('nan')
    x = resize_like(x, hw)
    h, w = hw
    x1, y1, x2, y2 = box_clip_xyxy(box, hw)
    xi1, yi1, xi2, yi2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    xi2 = max(xi1 + 1, xi2); yi2 = max(yi1 + 1, yi2)
    crop = x[yi1:yi2, xi1:xi2]
    return float(np.mean(crop)) if crop.size else float('nan')


def analyze_experiment(
    exp_dir: Path,
    out_dir: Path,
    mask_index: Dict[str, List[Path]],
    mask_root: Path,
    iou_thr: float,
    bg_overlap_thr: float,
    high_conf_thr: float,
    feature_root: str = None,
    allow_sample_mask_fallback: bool = False,
    fair_postprocess: bool = False,
    fair_nms_iou: float = 0.50,
    fair_containment_thr: float = 0.80,
    match_mode: str = 'lesion_aware',
    relaxed_iou_thr: float = 0.35,
    gt_coverage_thr: float = 0.70,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    fmap_root = exp_dir / (feature_root or 'layer_feature_maps')
    rows: List[dict] = []
    missing: List[dict] = []

    if not fmap_root.exists():
        missing.append({'experiment': exp_dir.name, 'reason': 'missing_layer_feature_maps'})
    else:
        root_meta = read_root_meta(fmap_root)
        for sample_dir in iter_sample_dirs(fmap_root):
            meta = read_sample_meta(sample_dir)
            if not meta and isinstance(root_meta.get('samples'), list):
                for s in root_meta['samples']:
                    if s.get('sample_id') == sample_dir.name:
                        meta = dict(s)
                        break
            hw = image_hw(sample_dir, meta)
            mask_path = find_mask(mask_index, sample_dir, meta, mask_root, allow_sample_mask_fallback=allow_sample_mask_fallback)
            if mask_path is None:
                missing.append({'experiment': exp_dir.name, 'sample_id': sample_dir.name, 'image_path': meta.get('image_path', ''), 'reason': 'missing_mask'})
                continue
            mask = load_mask01(mask_path, hw)
            raw_preds = read_prediction_meta(sample_dir, meta)
            for p in raw_preds:
                p['xyxy'] = box_clip_xyxy(p['xyxy'], hw)

            if fair_postprocess:
                preds, removed_preds = same_class_duplicate_suppression(
                    raw_preds,
                    iou_thr=fair_nms_iou,
                    containment_thr=fair_containment_thr,
                )
                postprocess_mode = 'same_class_duplicate_suppression'
            else:
                preds = raw_preds
                removed_preds = []
                postprocess_mode = 'raw_saved_predictions'

            duplicate_removed_count = len(removed_preds)

            gts = load_yolo_labels(meta, hw)
            for g in gts:
                g['xyxy'] = box_clip_xyxy(g['xyxy'], hw)
            matches = match_predictions(
                preds,
                gts,
                iou_thr=iou_thr,
                match_mode=match_mode,
                relaxed_iou_thr=relaxed_iou_thr,
                gt_coverage_thr=gt_coverage_thr,
            ) if gts else {}

            # Prefer true Stage1 teacher prior if saved; fallback to backbone_l2/l3.
            teacher_prior = load_map_by_roles(sample_dir, ['teacher_prior_fused'], meta, hw)
            prior_map = teacher_prior if teacher_prior is not None else load_map_by_roles(sample_dir, ['backbone_l2', 'backbone_l3'], meta, hw)
            prior_source = 'teacher_prior_fused' if teacher_prior is not None else 'fallback_backbone_l2_l3'

            # Prefer fused prehead support-like map if saved; fallback to P3/P4/P5 feature heatmaps.
            support_map = load_map_by_roles(sample_dir, ['prehead_support_fused'], meta, hw)
            support_source = 'prehead_support_fused'
            if support_map is None:
                support_map = load_map_by_roles(sample_dir, ['prehead_p4', 'prehead_p5', 'prehead_p3'], meta, hw)
                support_source = 'fallback_prehead_feature_fusion'

            for i, p in enumerate(preds):
                fg_overlap, center_in_fg = box_mask_overlap(mask, p['xyxy'])
                m = matches.get(i, _empty_match())
                rows.append({
                    'experiment': exp_dir.name,
                    'sample_id': sample_dir.name,
                    'dataset_split': infer_dataset_split(sample_dir.name, meta),
                    'postprocess_mode': postprocess_mode,
                    'num_raw_predictions': len(raw_preds),
                    'num_predictions_after_postprocess': len(preds),
                    'duplicate_removed_count': duplicate_removed_count,
                    'image_path': meta.get('image_path', ''),
                    'image_rel_path': meta.get('image_rel_path', ''),
                    'mask_path': str(mask_path),
                    'label_path': str(find_label_path(meta) or ''),
                    'pred_id': p['pred_id'],
                    'cls_id': p['cls_id'],
                    'cls_name': p['cls_name'],
                    'conf': p['conf'],
                    'x1': p['xyxy'][0], 'y1': p['xyxy'][1], 'x2': p['xyxy'][2], 'y2': p['xyxy'][3],
                    'box_area': box_area(p['xyxy']),
                    'center_in_fg': center_in_fg,
                    'box_fg_overlap': fg_overlap,
                    'is_bg_candidate': bool(fg_overlap < bg_overlap_thr),
                    'is_high_conf': bool(p['conf'] >= high_conf_thr),
                    'is_high_conf_bg_candidate': bool(p['conf'] >= high_conf_thr and fg_overlap < bg_overlap_thr),
                    'box_teacher_prior_score': box_mean_map(teacher_prior, p['xyxy'], hw),
                    'box_prior_score': box_mean_map(prior_map, p['xyxy'], hw),
                    'box_prehead_support_score': box_mean_map(support_map, p['xyxy'], hw),
                    'prior_source': prior_source,
                    'support_source': support_source,
                    'matched_gt_iou': m['matched_gt_iou'],
                    'matched_gt_class': m['matched_gt_class'],
                    'matched_gt_coverage': m.get('matched_gt_coverage', float('nan')),
                    'matched_pred_coverage': m.get('matched_pred_coverage', float('nan')),
                    'matched_area_ratio': m.get('matched_area_ratio', float('nan')),
                    'matched_gt_index': m.get('matched_gt_index', -1),
                    'best_any_gt_iou': m.get('best_any_gt_iou', float('nan')),
                    'best_any_gt_class': m.get('best_any_gt_class', -1),
                    'best_any_gt_coverage': m.get('best_any_gt_coverage', float('nan')),
                    'best_any_pred_coverage': m.get('best_any_pred_coverage', float('nan')),
                    'best_any_area_ratio': m.get('best_any_area_ratio', float('nan')),
                    'match_mode': match_mode,
                    'match_reason': m.get('match_reason', 'none'),
                    'is_tp': bool(m['is_tp']),
                    'is_gt_covered_candidate': bool(m.get('is_gt_covered_candidate', False)),
                    'is_fp': bool(m.get('is_annotation_fp', not m['is_tp'])) if gts else bool(fg_overlap < bg_overlap_thr),
                    'is_annotation_fp': bool(m.get('is_annotation_fp', not m['is_tp'])) if gts else bool(fg_overlap < bg_overlap_thr),
                    'is_background_fp': bool(m.get('is_annotation_fp', not m['is_tp']) and fg_overlap < bg_overlap_thr) if gts else bool(fg_overlap < bg_overlap_thr),
                    'is_foreground_unmatched_fp': bool(m.get('is_annotation_fp', not m['is_tp']) and fg_overlap >= bg_overlap_thr) if gts else False,
                    'is_high_conf_background_fp': bool(m.get('is_annotation_fp', not m['is_tp']) and p['conf'] >= high_conf_thr and fg_overlap < bg_overlap_thr) if gts else bool(p['conf'] >= high_conf_thr and fg_overlap < bg_overlap_thr),
                    'fp_type': (
                        'tp' if bool(m['is_tp']) else
                        ('gt_covered_coarse_or_duplicate' if bool(m.get('is_gt_covered_candidate', False)) else
                         ('background_fp' if fg_overlap < bg_overlap_thr else 'foreground_unmatched_fp'))
                    ),
                    'num_gt': len(gts),
                })

    df = pd.DataFrame(rows)
    missing_df = pd.DataFrame(missing)
    exp_out = ensure_dir(out_dir / exp_dir.name)
    df.to_csv(exp_out / 'candidate_quality.csv', index=False, encoding='utf-8-sig')
    summary = summarize(df)
    summary.to_csv(exp_out / 'candidate_quality_summary.csv', index=False, encoding='utf-8-sig')
    missing_df.to_csv(exp_out / 'candidate_quality_missing_samples.csv', index=False, encoding='utf-8-sig')
    save_plots(df, exp_out)
    report = {
        'experiment': exp_dir.name,
        'num_rows': int(len(df)),
        'num_missing_samples': int(len(missing)),
        'missing_samples_preview': missing[:30],
        'fair_postprocess': bool(fair_postprocess),
        'fair_postprocess_rule': 'same-class duplicate suppression only; cross-class overlaps are preserved; duplicate check uses IoU, candidate containment, reverse containment, and min-area overlap',
        'fair_nms_iou': float(fair_nms_iou),
        'fair_containment_thr': float(fair_containment_thr),
        'match_mode': str(match_mode),
        'iou_thr': float(iou_thr),
        'relaxed_iou_thr': float(relaxed_iou_thr),
        'gt_coverage_thr': float(gt_coverage_thr),
    }
    (exp_out / 'candidate_quality_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return df, missing_df, report


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    group_cols = ['experiment']
    if 'dataset_split' in df.columns:
        group_cols.append('dataset_split')
    if 'postprocess_mode' in df.columns:
        group_cols.append('postprocess_mode')

    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        pred_rows = sub[sub.get('pred_id', -1) != -1] if 'pred_id' in sub.columns else sub
        n_img = max(1, sub['sample_id'].nunique())
        n_pred = len(pred_rows)
        if 'duplicate_removed_count' in sub.columns and not sub.empty:
            duplicate_removed_count = int(sub.groupby('sample_id')['duplicate_removed_count'].max().sum())
            num_raw_predictions = int(sub.groupby('sample_id')['num_raw_predictions'].max().sum()) if 'num_raw_predictions' in sub.columns else n_pred + duplicate_removed_count
        else:
            duplicate_removed_count = 0
            num_raw_predictions = n_pred

        tp = int(pred_rows['is_tp'].sum()) if 'is_tp' in pred_rows else 0
        ann_fp = int(pred_rows['is_fp'].sum()) if 'is_fp' in pred_rows else 0
        bg_fp = int(pred_rows['is_background_fp'].sum()) if 'is_background_fp' in pred_rows else np.nan
        fg_unmatched_fp = int(pred_rows['is_foreground_unmatched_fp'].sum()) if 'is_foreground_unmatched_fp' in pred_rows else np.nan
        high_conf_bg_fp = int(pred_rows['is_high_conf_background_fp'].sum()) if 'is_high_conf_background_fp' in pred_rows else np.nan
        gt_covered = int(pred_rows['is_gt_covered_candidate'].sum()) if 'is_gt_covered_candidate' in pred_rows else 0

        row.update({
            'num_images': n_img,
            'num_raw_predictions': num_raw_predictions,
            'num_predictions': n_pred,
            'num_predictions_after_postprocess': n_pred,
            'duplicate_removed_count': duplicate_removed_count,
            'duplicate_removed_per_image': duplicate_removed_count / n_img,
            'pred_per_image': n_pred / n_img,
            'mean_conf': pred_rows['conf'].mean() if 'conf' in pred_rows and not pred_rows.empty else np.nan,
            'foreground_pred_ratio_center': pred_rows['center_in_fg'].mean() if 'center_in_fg' in pred_rows and not pred_rows.empty else np.nan,
            'mean_box_fg_overlap': pred_rows['box_fg_overlap'].mean() if 'box_fg_overlap' in pred_rows and not pred_rows.empty else np.nan,
            'bg_candidate_per_image': pred_rows['is_bg_candidate'].sum() / n_img if 'is_bg_candidate' in pred_rows else np.nan,
            'high_conf_bg_candidate_per_image': pred_rows['is_high_conf_bg_candidate'].sum() / n_img if 'is_high_conf_bg_candidate' in pred_rows else np.nan,
            'mean_box_teacher_prior_score': pred_rows['box_teacher_prior_score'].mean() if 'box_teacher_prior_score' in pred_rows and not pred_rows.empty else np.nan,
            'mean_box_prior_score': pred_rows['box_prior_score'].mean() if 'box_prior_score' in pred_rows and not pred_rows.empty else np.nan,
            'mean_box_prehead_support_score': pred_rows['box_prehead_support_score'].mean() if 'box_prehead_support_score' in pred_rows and not pred_rows.empty else np.nan,
            'tp_count': tp,
            'annotation_fp_count': ann_fp,
            'background_fp_count': bg_fp,
            'foreground_unmatched_fp_count': fg_unmatched_fp,
            'high_conf_background_fp_count': high_conf_bg_fp,
            'gt_covered_candidate_count': gt_covered,
            'gt_covered_candidate_per_image': gt_covered / n_img,
            'background_fp_per_image': bg_fp / n_img if np.isfinite(bg_fp) else np.nan,
            'foreground_unmatched_fp_per_image': fg_unmatched_fp / n_img if np.isfinite(fg_unmatched_fp) else np.nan,
            'high_conf_background_fp_per_image': high_conf_bg_fp / n_img if np.isfinite(high_conf_bg_fp) else np.nan,
            'precision_annotation_level': tp / max((tp + ann_fp), 1),
            'precision_background_only_fp': tp / max((tp + bg_fp), 1) if np.isfinite(bg_fp) else np.nan,
        })
        rows.append(row)
    return pd.DataFrame(rows)

def save_plots(df: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if df.empty:
        return
    for metric in ['box_fg_overlap', 'box_teacher_prior_score', 'box_prior_score', 'box_prehead_support_score', 'conf']:
        if metric not in df.columns:
            continue
        plt.figure(figsize=(8, 5))
        labels, data = [], []
        for exp, sub in df.groupby('experiment'):
            vals = sub[metric].dropna().values
            if vals.size:
                labels.append(exp)
                data.append(vals)
        if data:
            plt.boxplot(data, tick_labels=labels, showfliers=False)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel(metric)
            plt.tight_layout()
            plt.savefig(out_dir / f'{metric}_boxplot.png', dpi=200)
        plt.close()


def collect_experiment_names(runs_dir: Path, exp_names_args: list[str]):
    if len(exp_names_args) == 1 and exp_names_args[0].strip().lower() == "all":
        return [p.name for p in runs_dir.iterdir() if p.is_dir()]
    return exp_names_args

def main():
    parser = argparse.ArgumentParser(description='Analyze detection candidate quality with foreground masks and saved Stage2 visualization meta.')
    parser.add_argument('--runs_dir', type=str, default="./runs/glcp_stage2_yolo_det")
    parser.add_argument('--exp_names', type=str, nargs='+', required=True)
    parser.add_argument('--feature_root', type=str, default='layer_feature_maps')  # kept for CLI compatibility
    parser.add_argument('--mask_root', type=str, default="./data/unlabeled_train/foreground_masks")
    parser.add_argument('--out_dir', type=str, default='./runs/analysis_stage2_candidate_quality')
    parser.add_argument('--iou_thr', type=float, default=0.50)
    parser.add_argument('--match_mode', type=str, default='lesion_aware', choices=['strict_iou', 'lesion_aware'], help='strict_iou uses only IoU>=threshold; lesion_aware also treats high GT coverage as a valid/coarse match.')
    parser.add_argument('--relaxed_iou_thr', type=float, default=0.35, help='Optional relaxed IoU used with GT coverage in lesion-aware matching.')
    parser.add_argument('--gt_coverage_thr', type=float, default=0.70, help='Minimum fraction of a GT box covered by a prediction to avoid annotation-FP in lesion-aware matching.')
    parser.add_argument('--bg_overlap_thr', type=float, default=0.20)
    parser.add_argument('--high_conf_thr', type=float, default=0.50)
    parser.add_argument('--allow_sample_mask_fallback', action='store_true')
    parser.add_argument('--fair_postprocess', action='store_true', help='Apply same-class duplicate suppression before analysis. Cross-class overlaps are always preserved.')
    parser.add_argument('--fair_nms_iou', type=float, default=0.50, help='IoU threshold for suppressing lower-confidence duplicate boxes within the same class.')
    parser.add_argument('--fair_containment_thr', type=float, default=0.80, help='Containment threshold for suppressing lower-confidence contained boxes within the same class. Use <=0 to disable.')
    args = parser.parse_args()

    runs_dir = resolve_path(args.runs_dir)
    out_dir = ensure_dir(resolve_path(args.out_dir))
    mask_root = resolve_path(args.mask_root)
    mask_index = build_mask_index(mask_root)
    all_rows: List[pd.DataFrame] = []
    all_missing: List[pd.DataFrame] = []
    reports = []
    exp_names = collect_experiment_names(runs_dir, args.exp_names)

    for exp in exp_names:
        exp_dir = runs_dir.joinpath(exp)
        if not exp_dir.exists():
            print(f'[WARN] missing experiment: {exp_dir}')
            all_missing.append(pd.DataFrame([{'experiment': exp, 'reason': 'missing_experiment'}]))
            continue
        df, missing_df, report = analyze_experiment(
            exp_dir=exp_dir,
            feature_root=args.feature_root,
            out_dir=out_dir,
            mask_index=mask_index,
            mask_root=mask_root,
            iou_thr=args.iou_thr,
            bg_overlap_thr=args.bg_overlap_thr,
            high_conf_thr=args.high_conf_thr,
            allow_sample_mask_fallback=args.allow_sample_mask_fallback,
            fair_postprocess=args.fair_postprocess,
            fair_nms_iou=args.fair_nms_iou,
            fair_containment_thr=args.fair_containment_thr,
            match_mode=args.match_mode,
            relaxed_iou_thr=args.relaxed_iou_thr,
            gt_coverage_thr=args.gt_coverage_thr,
        )
        if not df.empty:
            all_rows.append(df)
        if not missing_df.empty:
            all_missing.append(missing_df)
        reports.append(report)
        print(f"[OK] Run {exp} finish.")

    df_all = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    missing_all = pd.concat(all_missing, ignore_index=True) if all_missing else pd.DataFrame()
    df_all.to_csv(out_dir / 'candidate_quality_all.csv', index=False, encoding='utf-8-sig')
    df_all.to_csv(out_dir / 'candidate_quality.csv', index=False, encoding='utf-8-sig')
    summary_all = summarize(df_all)
    summary_all.to_csv(out_dir / 'candidate_quality_summary_all.csv', index=False, encoding='utf-8-sig')
    summary_all.to_csv(out_dir / 'candidate_quality_summary.csv', index=False, encoding='utf-8-sig')
    missing_all.to_csv(out_dir / 'candidate_quality_missing_samples.csv', index=False, encoding='utf-8-sig')
    save_plots(df_all, out_dir)

    report_all = {
        'num_rows': int(len(df_all)),
        'num_missing_samples': int(len(missing_all)),
        'experiments': args.exp_names,
        'reports': reports,
        'fair_postprocess': bool(args.fair_postprocess),
        'fair_postprocess_rule': 'same-class duplicate suppression only; cross-class overlaps are preserved; duplicate check uses IoU, candidate containment, reverse containment, and min-area overlap',
        'fair_nms_iou': float(args.fair_nms_iou),
        'fair_containment_thr': float(args.fair_containment_thr),
        'match_mode': str(args.match_mode),
        'iou_thr': float(args.iou_thr),
        'relaxed_iou_thr': float(args.relaxed_iou_thr),
        'gt_coverage_thr': float(args.gt_coverage_thr),
    }
    (out_dir / 'candidate_quality_report_all.json').write_text(json.dumps(report_all, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[OK] rows={len(df_all)} missing={len(missing_all)} saved to {out_dir}')


if __name__ == '__main__':
    main()
