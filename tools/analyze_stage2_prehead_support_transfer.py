from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
DEFAULT_PREHEAD_ROLES = ['prehead_p3', 'prehead_p4', 'prehead_p5', 'prehead_support_fused']
DEFAULT_PRIOR_ROLES = ['teacher_prior_fused']  # preferred true Stage1 teacher prior saved by visualization script
FALLBACK_PRIOR_ROLES = ['backbone_l2', 'backbone_l3']

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


def existing_path(value: Any) -> Optional[Path]:
    if not value:
        return None
    p = Path(str(value))
    return p.resolve() if p.exists() else None


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


def is_sample_dir(path: Path) -> bool:
    """Support both legacy sample_000 and split-aware val_sample_00000/test_sample_00000 names."""
    name = path.name.lower()
    return name.startswith('sample_') or '_sample_' in name


def iter_sample_dirs(fmap_root: Path) -> List[Path]:
    return sorted([p for p in fmap_root.iterdir() if p.is_dir() and is_sample_dir(p)])


def infer_dataset_split(sample_dir: Path, meta: dict) -> str:
    for key in ('dataset_split', 'split'):
        v = meta.get(key, '')
        if v:
            return str(v)
    name = sample_dir.name
    if '_sample_' in name:
        return name.split('_sample_', 1)[0]
    return 'unknown'


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


def resize_like(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if x.shape[:2] == ref.shape[:2]:
        return x
    return cv2.resize(x, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LINEAR)


def fuse_maps(maps: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    maps = [m for m in maps if m is not None]
    if not maps:
        return None
    ref = maps[0]
    arr = [resize_like(m, ref) for m in maps]
    fused = np.mean(arr, axis=0).astype(np.float32)
    vmin, vmax = float(np.min(fused)), float(np.max(fused))
    if vmax > vmin:
        fused = (fused - vmin) / (vmax - vmin + 1e-8)
    return np.clip(fused, 0.0, 1.0)


def safe_mean(x: np.ndarray, mask: np.ndarray) -> float:
    vals = x[mask.astype(bool)]
    return float(np.mean(vals)) if vals.size else float('nan')


def corr_flat(a: np.ndarray, b: np.ndarray) -> float:
    b = resize_like(b, a)
    av = a.reshape(-1).astype(np.float64)
    bv = b.reshape(-1).astype(np.float64)
    if np.std(av) < 1e-12 or np.std(bv) < 1e-12:
        return float('nan')
    return float(np.corrcoef(av, bv)[0, 1])


def topk_mask(x: np.ndarray, top_ratio: float = 0.20) -> np.ndarray:
    thr = float(np.quantile(x.reshape(-1), 1.0 - top_ratio))
    return x >= thr


def topk_agreement(reference: np.ndarray, target: np.ndarray, top_ratio: float = 0.20) -> float:
    target = resize_like(target, reference)
    ref_top = topk_mask(reference, top_ratio)
    tgt_top = topk_mask(target, top_ratio)
    denom = tgt_top.sum()
    if denom == 0:
        return float('nan')
    return float(np.logical_and(ref_top, tgt_top).sum() / denom)


def topk_fg_coverage(target: np.ndarray, mask: np.ndarray, top_ratio: float = 0.20) -> float:
    top = topk_mask(target, top_ratio)
    denom = top.sum()
    if denom == 0:
        return float('nan')
    return float(np.logical_and(top, mask.astype(bool)).sum() / denom)


def load_prior_map(sample_dir: Path, meta: dict, prior_roles: List[str], use_fallback_backbone_prior: bool) -> Tuple[Optional[np.ndarray], List[str], str]:
    maps, paths = [], []
    source = 'requested_prior_roles'
    for role in prior_roles:
        hp = find_heat_by_role(sample_dir, role, meta=meta)
        # print(f"load_prior_map: {hp}")
        if hp is not None:
            maps.append(load_gray01(hp))
            paths.append(str(hp))
    if maps:
        return fuse_maps(maps), paths, source

    if use_fallback_backbone_prior:
        source = 'fallback_backbone_l2_l3'
        for role in FALLBACK_PRIOR_ROLES:
            hp = find_heat_by_role(sample_dir, role, meta=meta)
            # print(f"load_prior_map: {hp}")
            if hp is not None:
                maps.append(load_gray01(hp))
                paths.append(str(hp))
        if maps:
            return fuse_maps(maps), paths, source
    return None, paths, 'missing_prior'


def analyze_experiment(
    exp_dir: Path,
    out_dir: Path,
    mask_index: Dict[str, List[Path]],
    mask_root: Path,
    prior_roles: List[str],
    prehead_roles: List[str],
    feature_root: str = None,
    allow_sample_mask_fallback: bool = False,
    use_fallback_backbone_prior: bool = False,
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

            dataset_split = infer_dataset_split(sample_dir, meta)
            mask_path = find_mask(mask_index, sample_dir, meta, mask_root, allow_sample_mask_fallback=allow_sample_mask_fallback)
            if mask_path is None:
                missing.append({'experiment': exp_dir.name, 'sample_id': sample_dir.name, 'image_path': meta.get('image_path', ''), 'reason': 'missing_mask'})
                continue
            
            prior, prior_paths, prior_source = load_prior_map(sample_dir, meta, prior_roles, use_fallback_backbone_prior)
            if prior is None:
                missing.append({'experiment': exp_dir.name, 'sample_id': sample_dir.name, 'image_path': meta.get('image_path', ''), 'reason': 'missing_prior_heatmap'})
                continue

            # Per-layer prehead analysis.
            for role in prehead_roles:
                hp = find_heat_by_role(sample_dir, role, meta=meta)
                # print(f"load_prehead_map: {hp}")
                if hp is None:
                    missing.append({'experiment': exp_dir.name, 'sample_id': sample_dir.name, 'layer_role': role, 'reason': 'missing_prehead_heatmap'})
                    continue
                heat = load_gray01(hp)
                prior_resized = resize_like(prior, heat)
                mask_h = load_mask01(mask_path, heat.shape[:2])
                fg = mask_h.astype(bool)
                bg = ~fg
                mean_fg = safe_mean(heat, fg)
                mean_bg = safe_mean(heat, bg)
                gap = mean_fg - mean_bg if np.isfinite(mean_fg) and np.isfinite(mean_bg) else np.nan
                ratio = mean_fg / max(mean_bg, 1e-8) if np.isfinite(mean_fg) and np.isfinite(mean_bg) else np.nan
                mask_area_ratio = float(fg.mean()) if fg.size else np.nan
                top20_fg = topk_fg_coverage(heat, mask_h, 0.20)
                rows.append({
                    'experiment': exp_dir.name,
                    'dataset_split': dataset_split,
                    'sample_id': sample_dir.name,
                    'image_path': meta.get('image_path', ''),
                    'image_rel_path': meta.get('image_rel_path', ''),
                    'mask_path': str(mask_path),
                    'mask_area_ratio': mask_area_ratio,
                    'layer_role': role,
                    'heat_path': str(hp),
                    'prior_source': prior_source,
                    'prior_roles': ','.join(prior_roles),
                    'prior_heat_paths': ';'.join(prior_paths),
                    'mean_fg': mean_fg,
                    'mean_bg': mean_bg,
                    'support_gap': gap,
                    'support_ratio': ratio,
                    'top20_fg_coverage': top20_fg,
                    'top20_fg_lift': top20_fg / max(mask_area_ratio, 1e-8) if np.isfinite(top20_fg) and np.isfinite(mask_area_ratio) else np.nan,
                    'corr_with_teacher_prior': corr_flat(heat, prior_resized),
                    'top20_agreement_with_teacher_prior': topk_agreement(prior_resized, heat, 0.20),
                    'top10_agreement_with_teacher_prior': topk_agreement(prior_resized, heat, 0.10),
                })

            # Fused prehead support map if available.
            fused_hp = find_heat_by_role(sample_dir, 'prehead_support_fused', meta=meta)
            if fused_hp is not None:
                heat = load_gray01(fused_hp)
                prior_resized = resize_like(prior, heat)
                mask_h = load_mask01(mask_path, heat.shape[:2])
                fg = mask_h.astype(bool)
                bg = ~fg
                mean_fg = safe_mean(heat, fg)
                mean_bg = safe_mean(heat, bg)
                mask_area_ratio = float(fg.mean()) if fg.size else np.nan
                top20_fg = topk_fg_coverage(heat, mask_h, 0.20)
                rows.append({
                    'experiment': exp_dir.name,
                    'dataset_split': dataset_split,
                    'sample_id': sample_dir.name,
                    'image_path': meta.get('image_path', ''),
                    'image_rel_path': meta.get('image_rel_path', ''),
                    'mask_path': str(mask_path),
                    'mask_area_ratio': mask_area_ratio,
                    'layer_role': 'prehead_support_fused',
                    'heat_path': str(fused_hp),
                    'prior_source': prior_source,
                    'prior_roles': ','.join(prior_roles),
                    'prior_heat_paths': ';'.join(prior_paths),
                    'mean_fg': mean_fg,
                    'mean_bg': mean_bg,
                    'support_gap': mean_fg - mean_bg if np.isfinite(mean_fg) and np.isfinite(mean_bg) else np.nan,
                    'support_ratio': mean_fg / max(mean_bg, 1e-8) if np.isfinite(mean_fg) and np.isfinite(mean_bg) else np.nan,
                    'top20_fg_coverage': top20_fg,
                    'top20_fg_lift': top20_fg / max(mask_area_ratio, 1e-8) if np.isfinite(top20_fg) and np.isfinite(mask_area_ratio) else np.nan,
                    'corr_with_teacher_prior': corr_flat(heat, prior_resized),
                    'top20_agreement_with_teacher_prior': topk_agreement(prior_resized, heat, 0.20),
                    'top10_agreement_with_teacher_prior': topk_agreement(prior_resized, heat, 0.10),
                })

    df = pd.DataFrame(rows)
    missing_df = pd.DataFrame(missing)
    exp_out = ensure_dir(out_dir / exp_dir.name)
    df.to_csv(exp_out / 'support_transfer.csv', index=False, encoding='utf-8-sig')
    summary = summarize(df)
    summary.to_csv(exp_out / 'support_transfer_summary.csv', index=False, encoding='utf-8-sig')
    missing_df.to_csv(exp_out / 'support_transfer_missing_samples.csv', index=False, encoding='utf-8-sig')
    save_plots(df, exp_out)
    report = {
        'experiment': exp_dir.name,
        'num_rows': int(len(df)),
        'num_missing_samples': int(len(missing)),
        'missing_samples_preview': missing[:30],
        'prior_roles': prior_roles,
        'prehead_roles': prehead_roles,
        'use_fallback_backbone_prior': bool(use_fallback_backbone_prior),
    }
    (exp_out / 'support_transfer_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return df, missing_df, report


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    metrics = [
        'mask_area_ratio', 'mean_fg', 'mean_bg', 'support_gap', 'support_ratio',
        'top20_fg_coverage', 'top20_fg_lift',
        'corr_with_teacher_prior',
        'top20_agreement_with_teacher_prior',
        'top10_agreement_with_teacher_prior',
    ]
    group_cols = ['experiment', 'layer_role']
    if 'dataset_split' in df.columns:
        group_cols = ['experiment', 'dataset_split', 'layer_role']
    summary = df.groupby(group_cols)[metrics].agg(['mean', 'std', 'count']).reset_index()
    summary.columns = ['_'.join([x for x in c if x]) if isinstance(c, tuple) else c for c in summary.columns]
    return summary


def save_plots(df: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if df.empty:
        return
    for metric in ['support_gap', 'support_ratio', 'top20_fg_lift', 'corr_with_teacher_prior', 'top20_agreement_with_teacher_prior']:
        plt.figure(figsize=(10, 5))
        labels, data = [], []
        group_cols = ['experiment', 'layer_role']
        if 'dataset_split' in df.columns:
            group_cols = ['experiment', 'dataset_split', 'layer_role']
        for key, sub in df.groupby(group_cols):
            if len(group_cols) == 3:
                exp, split, layer = key
                label = f'{exp}\n{split}\n{layer}'
            else:
                exp, layer = key
                label = f'{exp}\n{layer}'
            vals = sub[metric].dropna().values
            if vals.size:
                labels.append(label)
                data.append(vals)
        if data:
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


def main():
    parser = argparse.ArgumentParser(description='Analyze Stage2 pre-head support transfer using saved Stage1 teacher prior and Stage2 pre-head maps.')
    parser.add_argument('--runs_dir', type=str, default="./runs/glcp_stage2_yolo_det")
    parser.add_argument('--exp_names', type=str, nargs='+', required=True)
    parser.add_argument('--feature_root', type=str, default='layer_feature_maps')  # kept for CLI compatibility
    parser.add_argument('--mask_root', type=str, default="./data/unlabeled_train/foreground_masks")
    parser.add_argument('--out_dir', type=str, default='./runs/analysis_stage2_support_transfer')
    parser.add_argument('--prior_roles', type=str, default=','.join(DEFAULT_PRIOR_ROLES))
    parser.add_argument('--prehead_roles', type=str, default=','.join(DEFAULT_PREHEAD_ROLES))
    parser.add_argument('--allow_sample_mask_fallback', action='store_true')
    parser.add_argument('--use_fallback_backbone_prior', action='store_true', help='If teacher_prior_fused is missing, fall back to Stage2 backbone_l2/l3 heatmaps. Not recommended for final mechanism evidence.')
    args = parser.parse_args()

    runs_dir = resolve_path(args.runs_dir)
    out_dir = ensure_dir(resolve_path(args.out_dir))
    mask_root = resolve_path(args.mask_root)
    mask_index = build_mask_index(mask_root)
    prior_roles = [x.strip() for x in args.prior_roles.split(',') if x.strip()]
    prehead_roles = [x.strip() for x in args.prehead_roles.split(',') if x.strip()]

    all_rows: List[pd.DataFrame] = []
    all_missing: List[pd.DataFrame] = []
    reports = []
    exp_names = collect_experiment_names(runs_dir, args.exp_names)

    for exp in exp_names:
        exp_dir = runs_dir.joinpath(exp)
        if not exp_dir.exists():
            print(f'[WARN] missing experiment: {exp_dir}')
            missing = pd.DataFrame([{'experiment': exp, 'reason': 'missing_experiment'}])
            all_missing.append(missing)
            continue
        df, missing_df, report = analyze_experiment(
            exp_dir=exp_dir,
            feature_root=args.feature_root,
            out_dir=out_dir,
            mask_index=mask_index,
            mask_root=mask_root,
            prior_roles=prior_roles,
            prehead_roles=prehead_roles,
            allow_sample_mask_fallback=args.allow_sample_mask_fallback,
            use_fallback_backbone_prior=args.use_fallback_backbone_prior,
        )
        if not df.empty:
            all_rows.append(df)
        if not missing_df.empty:
            all_missing.append(missing_df)
        reports.append(report)
        print(f"[OK] Run {exp} finish.")

    df_all = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    missing_all = pd.concat(all_missing, ignore_index=True) if all_missing else pd.DataFrame()
    df_all.to_csv(out_dir / 'support_transfer_all.csv', index=False, encoding='utf-8-sig')
    df_all.to_csv(out_dir / 'support_transfer.csv', index=False, encoding='utf-8-sig')
    summary_all = summarize(df_all)
    summary_all.to_csv(out_dir / 'support_transfer_summary_all.csv', index=False, encoding='utf-8-sig')
    summary_all.to_csv(out_dir / 'support_transfer_summary.csv', index=False, encoding='utf-8-sig')
    missing_all.to_csv(out_dir / 'support_transfer_missing_samples.csv', index=False, encoding='utf-8-sig')
    save_plots(df_all, out_dir)
    report_all = {
        'num_rows': int(len(df_all)),
        'num_missing_samples': int(len(missing_all)),
        'experiments': args.exp_names,
        'reports': reports,
        'prior_roles': prior_roles,
        'prehead_roles': prehead_roles,
    }
    (out_dir / 'support_transfer_report_all.json').write_text(json.dumps(report_all, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[OK] rows={len(df_all)} missing={len(missing_all)} saved to {out_dir}')


if __name__ == '__main__':
    main()
