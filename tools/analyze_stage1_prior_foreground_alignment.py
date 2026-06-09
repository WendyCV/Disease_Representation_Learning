from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MASK_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

BRANCH_NAMES = [
    'after_raw_local_embs', 'after_local_embs', 'after_raw', 'after_pos',
    'before_raw_local_embs', 'before_local_embs', 'before_raw', 'before_pos',
    'raw_local_embs', 'local_embs', 'raw', 'pos',
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_str: str | Path, base: Optional[Path] = None) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return ((base or PROJECT_ROOT) / p).resolve()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


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


def is_sample_dir(path: Path) -> bool:
    """Support both legacy sample_000 and split-aware val_sample_00000/test_sample_00000 names."""
    name = path.name.lower()
    return name.startswith('sample_') or '_sample_' in name


def infer_dataset_split(sample_dir: Path, meta: Dict) -> str:
    for key in ('dataset_split', 'split'):
        v = meta.get(key, '')
        if v:
            return str(v)
    name = sample_dir.name
    if '_sample_' in name:
        return name.split('_sample_', 1)[0]
    return 'unknown'


def iter_sample_dirs(feature_root: Path) -> List[Path]:
    dirs = sorted([p for p in feature_root.rglob('*') if p.is_dir() and is_sample_dir(p)])
    return dirs if dirs else [feature_root]


def read_meta(sample_dir: Path) -> Dict:
    for name in ('sample_meta.json', 'meta.json'):
        p = sample_dir / name
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding='utf-8'))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    return {}


def build_mask_index(mask_root: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    if not mask_root.exists():
        return idx
    for p in mask_root.rglob('*'):
        if p.is_file() and p.suffix.lower() in MASK_EXTS:
            idx[p.stem.lower()] = p.resolve()
            idx[p.name.lower()] = p.resolve()
    return idx


def find_external_mask(meta: Dict, mask_root: Path, mask_index: Dict[str, Path]) -> Optional[Path]:
    """Find an external foreground mask. Does NOT use sample_dir/mask_resized.png by default.

    This is important for wo_pos_wo_mask: visualization may save an all-one fallback
    mask because training did not use masks. For foreground-prior analysis, baseline
    heatmaps must still be evaluated against the same external foreground masks.
    """
    candidates: List[Path] = []

    # 1) If meta already records an external expected mask path, use it first.
    for key in ('expected_mask_path', 'mask_path'):
        v = meta.get(key, '')
        if v:
            p = Path(v)
            # avoid sample-local fallback masks such as sample_000/mask_resized.png
            if p.name.lower() != 'mask_resized.png' and p.exists():
                candidates.append(p)

    # 2) Reconstruct by relative path when possible.
    rel = meta.get('image_rel_path', '') or meta.get('rel_path', '')
    suffix = meta.get('mask_suffix', '.png') or '.png'
    if rel:
        rel_p = Path(rel)
        candidates.append(mask_root / rel_p.with_suffix(suffix))

    # 3) Lookup by original image stem/name.
    for key in ('image_path', 'image_name', 'image_stem'):
        v = meta.get(key, '')
        if not v:
            continue
        q = Path(str(v))
        for token in (q.stem.lower(), q.name.lower()):
            if token in mask_index:
                candidates.append(mask_index[token])

    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


def fallback_sample_mask(sample_dir: Path) -> Optional[Path]:
    for name in ('mask_resized.png', 'mask.png'):
        p = sample_dir / name
        if p.exists():
            return p.resolve()
    return None


def parse_branch_layer(heat_path: Path, sample_dir: Path) -> Tuple[str, str]:
    # branch is usually the directory directly under sample_dir: sample_000/after_pos/layer2_heatmap.jpg
    try:
        rel_parts = heat_path.relative_to(sample_dir).parts
        if len(rel_parts) >= 2:
            branch = rel_parts[0]
        else:
            branch = 'unknown'
    except Exception:
        branch = 'unknown'

    if branch == 'unknown':
        joined = '/'.join([p.lower() for p in heat_path.parts])
        for b in BRANCH_NAMES:
            if f'/{b}/' in joined or b in heat_path.stem.lower():
                branch = b
                break

    stem = heat_path.stem.lower()
    m = re.search(r'layer[_-]?([123])', stem)
    if m:
        return branch, f'L{m.group(1)}'
    m = re.search(r'(^|[_\-])l([123])($|[_\-])', stem)
    if m:
        return branch, f'L{m.group(2)}'
    return branch, 'unknown'


def find_heatmaps(sample_dir: Path, include_keywords: Iterable[str]) -> List[Path]:
    kws = [k.lower() for k in include_keywords]
    out: List[Path] = []
    for p in sample_dir.rglob('*'):
        if not (p.is_file() and p.suffix.lower() in IMG_EXTS):
            continue
        s = str(p).lower()
        name = p.name.lower()
        if any(x in name for x in ('overlay', 'overview', 'prediction', 'input')):
            continue
        if 'heat' not in name:
            continue
        if kws and not any(k in s for k in kws):
            continue
        out.append(p)
    return sorted(out)


def safe_mean(x: np.ndarray, mask: np.ndarray) -> float:
    vals = x[mask.astype(bool)]
    return float(vals.mean()) if vals.size else float('nan')


def topk_fg_coverage(heat: np.ndarray, mask: np.ndarray, top_ratio: float) -> Tuple[float, float]:
    thr = float(np.quantile(heat.reshape(-1), 1.0 - top_ratio))
    top = heat >= thr
    denom = int(top.sum())
    if denom == 0:
        return float('nan'), float('nan')
    cov = float(np.logical_and(top, mask.astype(bool)).sum() / denom)
    actual_area = float(denom / heat.size)
    return cov, actual_area


def binary_metrics_from_quantile(heat: np.ndarray, mask: np.ndarray, q: float) -> Tuple[float, float, float]:
    thr = float(np.quantile(heat.reshape(-1), q))
    pred = heat >= thr
    fg = mask.astype(bool)
    inter = np.logical_and(pred, fg).sum()
    union = np.logical_or(pred, fg).sum()
    pred_sum = pred.sum()
    fg_sum = fg.sum()
    iou = float(inter / union) if union > 0 else float('nan')
    dice = float((2 * inter) / (pred_sum + fg_sum)) if (pred_sum + fg_sum) > 0 else float('nan')
    pred_area = float(pred_sum / heat.size)
    return iou, dice, pred_area


def random_iou_expectation(mask_area_ratio: float, pred_area_ratio: float) -> float:
    inter = mask_area_ratio * pred_area_ratio
    union = mask_area_ratio + pred_area_ratio - inter
    return float(inter / union) if union > 0 else float('nan')


def analyze_one(sample_id: str, heat_path: Path, sample_dir: Path, mask_path: Path, meta: Dict, exp_name: str) -> Dict:
    heat = load_gray01(heat_path)
    mask = load_mask01(mask_path, heat.shape[:2])
    fg = mask.astype(bool)
    bg = ~fg
    mean_fg = safe_mean(heat, fg)
    mean_bg = safe_mean(heat, bg)
    fbr = mean_fg / max(mean_bg, 1e-8) if np.isfinite(mean_fg) and np.isfinite(mean_bg) else float('nan')
    gap = mean_fg - mean_bg if np.isfinite(mean_fg) and np.isfinite(mean_bg) else float('nan')
    bsr = mean_bg / max(mean_fg, 1e-8) if np.isfinite(mean_fg) and np.isfinite(mean_bg) else float('nan')
    mask_area = float(fg.mean())
    bg_area = float(bg.mean())

    top10_cov, top10_area = topk_fg_coverage(heat, mask, 0.10)
    top20_cov, top20_area = topk_fg_coverage(heat, mask, 0.20)
    top10_lift = top10_cov / max(mask_area, 1e-8) if np.isfinite(top10_cov) else float('nan')
    top20_lift = top20_cov / max(mask_area, 1e-8) if np.isfinite(top20_cov) else float('nan')

    iou70, dice70, pred_area70 = binary_metrics_from_quantile(heat, mask, 0.70)
    iou80, dice80, pred_area80 = binary_metrics_from_quantile(heat, mask, 0.80)
    riou70 = random_iou_expectation(mask_area, pred_area70)
    riou80 = random_iou_expectation(mask_area, pred_area80)
    iou_lift70 = iou70 / max(riou70, 1e-8) if np.isfinite(iou70) and np.isfinite(riou70) else float('nan')
    iou_lift80 = iou80 / max(riou80, 1e-8) if np.isfinite(iou80) and np.isfinite(riou80) else float('nan')

    branch, layer = parse_branch_layer(heat_path, sample_dir)
    dataset_split = infer_dataset_split(sample_dir, meta)
    return {
        'experiment': exp_name,
        'dataset_split': dataset_split,
        'sample_id': sample_id,
        'image_path': meta.get('image_path', ''),
        'image_rel_path': meta.get('image_rel_path', ''),
        'mask_path': str(mask_path),
        'heat_path': str(heat_path),
        'branch': branch,
        'layer': layer,
        'mean_fg': mean_fg,
        'mean_bg': mean_bg,
        'foreground_background_response_ratio': fbr,
        'foreground_background_response_gap': gap,
        'background_suppression_ratio': bsr,
        'mask_area_ratio': mask_area,
        'background_area_ratio': bg_area,
        'top10_fg_coverage': top10_cov,
        'top20_fg_coverage': top20_cov,
        'top10_actual_area_ratio': top10_area,
        'top20_actual_area_ratio': top20_area,
        'top10_fg_lift': top10_lift,
        'top20_fg_lift': top20_lift,
        'iou_q70': iou70,
        'dice_q70': dice70,
        'pred_area_ratio_q70': pred_area70,
        'iou_random_q70': riou70,
        'iou_lift_q70': iou_lift70,
        'iou_q80': iou80,
        'dice_q80': dice80,
        'pred_area_ratio_q80': pred_area80,
        'iou_random_q80': riou80,
        'iou_lift_q80': iou_lift80,
    }


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        'mean_fg', 'mean_bg', 'foreground_background_response_ratio', 'foreground_background_response_gap',
        'background_suppression_ratio', 'mask_area_ratio', 'top10_fg_coverage', 'top20_fg_coverage',
        'top10_fg_lift', 'top20_fg_lift', 'iou_q70', 'dice_q70', 'iou_lift_q70', 'iou_q80', 'dice_q80', 'iou_lift_q80',
    ]
    group_cols = ['experiment', 'branch', 'layer']
    if 'dataset_split' in df.columns:
        group_cols = ['experiment', 'dataset_split', 'branch', 'layer']
    return df.groupby(group_cols, dropna=False)[metrics].agg(['mean', 'std', 'count']).reset_index()


def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = ['_'.join([str(c) for c in col if c]) if isinstance(col, tuple) else str(col) for col in df.columns]
    return df


def save_plots(df: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if df.empty:
        return
    for metric in ['foreground_background_response_ratio', 'top20_fg_lift', 'iou_lift_q70']:
        labels, data = [], []
        group_cols = ['experiment', 'branch', 'layer']
        if 'dataset_split' in df.columns:
            group_cols = ['experiment', 'dataset_split', 'branch', 'layer']
        for key, sub in df.groupby(group_cols):
            if len(group_cols) == 4:
                exp, split, branch, layer = key
                label = f'{exp}\n{split}\n{branch}\n{layer}'
            else:
                exp, branch, layer = key
                label = f'{exp}\n{branch}\n{layer}'
            vals = sub[metric].dropna().values
            if vals.size:
                labels.append(label)
                data.append(vals)
        if not data:
            continue
        plt.figure(figsize=(max(10, 0.45 * len(labels)), 5))
        plt.boxplot(data, tick_labels=labels, showfliers=False)
        plt.xticks(rotation=60, ha='right')
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(out_dir / f'{metric}_boxplot.png', dpi=200)
        plt.close()


def collect_experiment_names(runs_dir: Path, exp_names_args: list[str]):
    if len(exp_names_args) == 1 and exp_names_args[0].strip().lower() == "all":
        return [p.name for p in runs_dir.iterdir() if p.is_dir()]
    return exp_names_args


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze Stage1 foreground prior alignment using external foreground masks.')
    parser.add_argument('--runs_dir', type=str, default='./runs/glcp_stage1_yolo_det')
    parser.add_argument('--exp_names', type=str, nargs='+', required=True)
    parser.add_argument('--feature_root', type=str, default='feature_maps_backbone')
    parser.add_argument('--mask_root', type=str, default='./data/unlabeled_train/foreground_masks')
    parser.add_argument('--out_dir', type=str, default='./runs/analysis_stage1_prior_quality')
    parser.add_argument('--include_keywords', type=str, nargs='*', default=['after_raw', 'after_pos', 'after_local_embs', 'after_raw_local_embs'])
    parser.add_argument('--allow_sample_mask_fallback', action='store_true', help='Allow sample_dir/mask_resized.png fallback. Not recommended for wo_pos_wo_mask.')
    args = parser.parse_args()

    runs_dir = resolve_path(args.runs_dir)
    mask_root = resolve_path(args.mask_root)
    out_dir = ensure_dir(resolve_path(args.out_dir))
    mask_index = build_mask_index(mask_root)

    all_rows: List[Dict] = []
    report = {'mask_root': str(mask_root), 'experiments': {}}
    exp_names = collect_experiment_names(runs_dir, args.exp_names)

    for exp in exp_names:
        exp_dir = runs_dir / exp
        feature_root = exp_dir / args.feature_root
        exp_out = ensure_dir(out_dir / exp)
        rows: List[Dict] = []
        missing_masks: List[str] = []
        missing_heatmaps: List[str] = []

        for sample_dir in iter_sample_dirs(feature_root):
            meta = read_meta(sample_dir)
            mask_path = find_external_mask(meta, mask_root, mask_index)
            if mask_path is None and args.allow_sample_mask_fallback:
                mask_path = fallback_sample_mask(sample_dir)
            if mask_path is None:
                missing_masks.append(str(sample_dir))
                continue
            heatmaps = find_heatmaps(sample_dir, args.include_keywords)
            if not heatmaps:
                missing_heatmaps.append(str(sample_dir))
            for hp in heatmaps:
                try:
                    rows.append(analyze_one(sample_dir.name, hp, sample_dir, mask_path, meta, exp))
                except Exception as e:
                    print(f'[WARN] failed {hp}: {e}')

        df = pd.DataFrame(rows)
        df.to_csv(exp_out / 'prior_quality.csv', index=False, encoding='utf-8-sig')
        if not df.empty:
            summary = flatten_cols(make_summary(df))
            summary.to_csv(exp_out / 'prior_quality_summary.csv', index=False, encoding='utf-8-sig')
            for metric in ['foreground_background_response_ratio', 'top20_fg_lift', 'iou_lift_q70', 'mask_area_ratio']:
                pivot = df.pivot_table(index=['branch', 'layer'], values=metric, aggfunc='mean')
                pivot.to_csv(exp_out / f'pivot_{metric}_mean.csv', encoding='utf-8-sig')
            save_plots(df, exp_out)
            all_rows.extend(rows)
        report['experiments'][exp] = {
            'feature_root': str(feature_root),
            'num_rows': int(len(df)),
            'num_samples': int(len(iter_sample_dirs(feature_root))),
            'num_missing_masks': int(len(missing_masks)),
            'num_missing_heatmaps': int(len(missing_heatmaps)),
            'missing_masks': missing_masks[:50],
            'missing_heatmaps': missing_heatmaps[:50],
        }
        print(f"[OK] Run {exp} finish.")

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(out_dir / 'prior_quality_all.csv', index=False, encoding='utf-8-sig')
    if not all_df.empty:
        flatten_cols(make_summary(all_df)).to_csv(out_dir / 'prior_quality_summary_all.csv', index=False, encoding='utf-8-sig')
        save_plots(all_df, out_dir)
    (out_dir / 'prior_quality_report_all.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[OK] saved to {out_dir}; rows={len(all_df)}')


if __name__ == '__main__':
    main()
