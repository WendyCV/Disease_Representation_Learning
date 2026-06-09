import os
import argparse
import zipfile
from pathlib import Path


DEFAULT_INCLUDE_FILES = {
    "train_log.csv",
    "config_used.yaml",
    "config.yaml",
    "samples.zip",
}

DEFAULT_INCLUDE_DIRS = {
    "feature_maps_backbone",
}


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_comma_list(value: str | None) -> list[str]:
    """
    Parse comma-separated string into a clean list.

    Example:
        "compare,__pycache__" -> ["compare", "__pycache__"]
        "" -> []
        None -> []
    """
    if value is None:
        return []
    value = str(value).strip()
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def add_file_to_zip(zf: zipfile.ZipFile, file_path: Path, arcname: Path) -> bool:
    if file_path.exists() and file_path.is_file():
        zf.write(file_path, arcname.as_posix())
        return True
    return False


def add_dir_to_zip(
    zf: zipfile.ZipFile,
    dir_path: Path,
    arcroot: Path,
    arc_prefix: Path | None = None,
    ignore_dirs: list[str] | None = None,
) -> int:
    """
    Add a directory recursively into zip.

    Args:
        zf:
            Open ZipFile object.
        dir_path:
            Directory to add.
        arcroot:
            Root used to compute relative path.
            Example:
                exp_dir = runs/xxx/exp_name
                sample_dir = exp_dir/feature_maps_backbone/sample_000
                arcroot = exp_dir
                then arc path starts with feature_maps_backbone/sample_000/...
        arc_prefix:
            Optional prefix inside zip, usually experiment name.
        ignore_dirs:
            Directory names to ignore. If empty, no directory is ignored.

    Returns:
        Number of files added.
    """
    if not dir_path.exists() or not dir_path.is_dir():
        return 0

    ignore_set = set(ignore_dirs or [])
    count = 0

    for p in sorted(dir_path.rglob("*")):
        if not p.is_file():
            continue

        rel_to_dir = p.relative_to(dir_path)

        # Skip if any parent folder under dir_path is in ignore_set.
        # Example: sample_001/compare/a.jpg -> parts[:-1] = ("sample_001", "compare")
        if any(part in ignore_set for part in rel_to_dir.parts[:-1]):
            continue

        rel = p.relative_to(arcroot)
        arcname = rel if arc_prefix is None else (arc_prefix / rel)
        zf.write(p, arcname.as_posix())
        count += 1

    return count


def collect_experiment_names(runs_dir: Path, exp_names_arg: str) -> list[str]:
    """
    Collect experiment names.

    Args:
        exp_names_arg:
            "all" or comma-separated experiment names.
    """
    if exp_names_arg.strip().lower() == "all":
        if not runs_dir.exists():
            return []
        return [p.name for p in sorted(runs_dir.iterdir()) if p.is_dir()]

    names = [x.strip() for x in exp_names_arg.split(",") if x.strip()]
    return names


def collect_sample_dirs(
    fmap_dir: Path,
    max_samples: int,
    sample_names: list[str],
) -> list[Path]:
    """
    Collect sample directories from feature_maps_backbone.

    Priority:
        1. If sample_names is provided, use exactly those names if they exist.
        2. Otherwise use sample_000 ~ sample_{max_samples-1}.
        3. If max_samples < 0, include all sample-like directories.
    """
    if not fmap_dir.exists() or not fmap_dir.is_dir():
        return []

    if sample_names:
        sample_dirs = []
        for name in sample_names:
            p = fmap_dir / name
            if p.exists() and p.is_dir():
                sample_dirs.append(p)
            else:
                print(f"[WARN] sample dir not found: {p}")
        return sample_dirs

    if max_samples is not None and max_samples >= 0:
        sample_dirs = []
        for i in range(max_samples):
            sample_dir = fmap_dir / f"sample_{i:03d}"
            if sample_dir.exists() and sample_dir.is_dir():
                sample_dirs.append(sample_dir)
        return sample_dirs

    # max_samples < 0 means include all sample directories.
    return [
        p for p in sorted(fmap_dir.iterdir())
        if p.is_dir() and "sample" in p.name
    ]


def pack_experiment(
    zf: zipfile.ZipFile,
    exp_dir: Path,
    exp_name: str,
    include_ckpt: bool,
    include_feature_maps: bool,
    max_samples: int,
    sample_names: list[str],
    ignore_dirs: list[str],
) -> dict:
    """
    Pack one experiment and return stats.
    """
    stats = {
        "experiment": exp_name,
        "files_added": 0,
        "key_files": [],
        "checkpoints": [],
        "feature_maps": [],
        "samples": [],
        "ignored_dirs": ignore_dirs,
    }

    print(f"[PACK] {exp_name}")

    # 1) Pack key files
    for fname in sorted(DEFAULT_INCLUDE_FILES):
        file_path = exp_dir / fname
        if file_path.exists():
            arcname = Path(exp_name) / fname
            if add_file_to_zip(zf, file_path, arcname):
                stats["files_added"] += 1
                stats["key_files"].append(fname)

    # 2) Optional checkpoints
    if include_ckpt:
        for ckpt_name in ("best.pth", "last.pth"):
            ckpt_path = exp_dir / ckpt_name
            if ckpt_path.exists():
                arcname = Path(exp_name) / ckpt_name
                if add_file_to_zip(zf, ckpt_path, arcname):
                    stats["files_added"] += 1
                    stats["checkpoints"].append(ckpt_name)

    # 3) Optional feature maps
    # Important:
    #   We do NOT skip feature_maps_backbone even if samples.zip exists,
    #   because compare/summary images are needed for RPD/LS analysis.
    if include_feature_maps:
        fmap_dir = exp_dir / "feature_maps_backbone"
        if fmap_dir.exists() and fmap_dir.is_dir():
            # meta.json at feature root
            meta_path = fmap_dir / "meta.json"
            if meta_path.exists():
                arcname = Path(exp_name) / "feature_maps_backbone" / "meta.json"
                if add_file_to_zip(zf, meta_path, arcname):
                    stats["files_added"] += 1
                    stats["feature_maps"].append("feature_maps_backbone/meta.json")

            summary_path = fmap_dir / "summary.json"
            if summary_path.exists():
                arcname = Path(exp_name) / "feature_maps_backbone" / "summary.json"
                if add_file_to_zip(zf, summary_path, arcname):
                    stats["files_added"] += 1
                    stats["feature_maps"].append("feature_maps_backbone/summary.json")

            sample_dirs = collect_sample_dirs(
                fmap_dir=fmap_dir,
                max_samples=max_samples,
                sample_names=sample_names,
            )

            if not sample_dirs:
                print(f"[WARN] no sample dirs found under: {fmap_dir}")

            for sample_dir in sample_dirs:
                added = add_dir_to_zip(
                    zf,
                    sample_dir,
                    arcroot=exp_dir,
                    arc_prefix=Path(exp_name),
                    ignore_dirs=ignore_dirs,
                )
                stats["files_added"] += added
                stats["samples"].append({
                    "sample": sample_dir.name,
                    "files_added": added,
                })
        else:
            print(f"[WARN] feature_maps_backbone not found: {fmap_dir}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Pack Stage1 training logs/configs/feature maps for RPD/LS analysis."
    )

    parser.add_argument(
        "--runs_dir",
        type=str,
        default="./runs/glcp_stage1_yolo_det",
        help="Stage1 runs root directory.",
    )

    parser.add_argument(
        "--exp_names",
        type=str,
        default="all",
        help='Comma-separated experiment names, or "all".',
    )

    parser.add_argument(
        "--output_zip",
        type=str,
        default="./runs/glcp_stage1_yolo_det/stage1_results_bundle.zip",
        help="Output zip path.",
    )

    parser.add_argument(
        "--include_ckpt",
        action="store_true",
        help="Include best.pth and last.pth.",
    )

    parser.add_argument(
        "--include_feature_maps",
        action="store_true",
        help="Include feature_maps_backbone directory.",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help=(
            "If include_feature_maps is enabled and sample_names is not provided, "
            "pack sample_000 ~ sample_{max_samples-1:03d}. "
            "Use -1 to include all sample-like directories."
        ),
    )

    parser.add_argument(
        "--sample_names",
        type=str,
        default="",
        help=(
            "Comma-separated sample directory names to pack. "
            "Example: sample_000,sample_001,sample_002,sample_007,sample_008,sample_009. "
            "If provided, max_samples is ignored."
        ),
    )

    parser.add_argument(
        "--ignore_dirs",
        type=str,
        default="",
        help=(
            "Comma-separated directory names to ignore when packing sample folders. "
            "Default is empty, meaning no directory is ignored. "
            "Example: --ignore_dirs __pycache__,tmp. "
            "Do NOT ignore compare if you want RPD/LS visual analysis."
        ),
    )

    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    output_zip = Path(args.output_zip).resolve()
    ensure_parent(output_zip)

    if not runs_dir.exists() or not runs_dir.is_dir():
        raise FileNotFoundError(f"runs_dir not found: {runs_dir}")

    exp_names = collect_experiment_names(runs_dir, args.exp_names)
    sample_names = parse_comma_list(args.sample_names)
    ignore_dirs = parse_comma_list(args.ignore_dirs)

    print("[INFO] Stage1 pack settings")
    print(f"       runs_dir             : {runs_dir}")
    print(f"       exp_names            : {exp_names}")
    print(f"       output_zip           : {output_zip}")
    print(f"       include_ckpt         : {args.include_ckpt}")
    print(f"       include_feature_maps : {args.include_feature_maps}")
    print(f"       max_samples          : {args.max_samples}")
    print(f"       sample_names         : {sample_names}")
    print(f"       ignore_dirs          : {ignore_dirs}")
    print("")

    stats_all = []

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for exp_name in exp_names:
            exp_dir = runs_dir / exp_name
            if not exp_dir.exists() or not exp_dir.is_dir():
                print(f"[SKIP] missing experiment dir: {exp_dir}")
                continue

            stats = pack_experiment(
                zf=zf,
                exp_dir=exp_dir,
                exp_name=exp_name,
                include_ckpt=args.include_ckpt,
                include_feature_maps=args.include_feature_maps,
                max_samples=args.max_samples,
                sample_names=sample_names,
                ignore_dirs=ignore_dirs,
            )
            stats_all.append(stats)

        # Write a simple manifest inside zip.
        manifest_text = {
            "runs_dir": str(runs_dir),
            "output_zip": str(output_zip),
            "experiments": stats_all,
            "include_ckpt": bool(args.include_ckpt),
            "include_feature_maps": bool(args.include_feature_maps),
            "max_samples": int(args.max_samples),
            "sample_names": sample_names,
            "ignore_dirs": ignore_dirs,
            "notes": [
                "For train_log.csv with duplicated rows, analyze the last 50 valid records.",
                "compare directory is included by default because summary_before_vs_after_raw/pos images are needed.",
                "Use --ignore_dirs only when you intentionally want to exclude some folders.",
            ],
        }

        import json
        zf.writestr(
            "pack_manifest.json",
            json.dumps(manifest_text, ensure_ascii=False, indent=2),
        )

    print(f"\nDone. Saved to: {output_zip}")
    print(output_zip)


if __name__ == "__main__":
    main()