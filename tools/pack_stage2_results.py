from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


DEFAULT_INCLUDE_FILES = {
    "results.csv",
    "config_used.yaml",
    "config.yaml",
}


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_comma_list(value: str | None) -> list[str]:
    """
    Parse comma-separated string into a clean list.

    Example:
        "sample_000,sample_001" -> ["sample_000", "sample_001"]
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


def add_file(
    zf: zipfile.ZipFile,
    file_path: Path,
    arcname: Path,
) -> bool:
    if file_path.exists() and file_path.is_file():
        zf.write(file_path, arcname.as_posix())
        return True
    return False


def add_dir(
    zf: zipfile.ZipFile,
    dir_path: Path,
    arcname_root: Path,
    ignore_dirs: list[str] | None = None,
) -> int:
    """
    Add a directory recursively into zip.

    Args:
        zf:
            Open ZipFile object.
        dir_path:
            Directory to add.
        arcname_root:
            Root path inside zip.
        ignore_dirs:
            Directory names to ignore.

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

        # Example:
        # sample_001/compare/a.jpg -> parts[:-1] = ("sample_001", "compare")
        # If "compare" is in ignore_dirs, this file will be skipped.
        if any(part in ignore_set for part in rel_to_dir.parts[:-1]):
            continue

        arcname = arcname_root / rel_to_dir
        zf.write(p, arcname.as_posix())
        count += 1

    return count


def collect_exp_names(runs_dir: Path, exp_names_arg: str) -> list[str]:
    """
    Collect experiment names from runs_dir.

    Args:
        exp_names_arg:
            "all" or comma-separated experiment names.
    """
    if exp_names_arg.strip().lower() == "all":
        if not runs_dir.exists():
            return []
        return [p.name for p in sorted(runs_dir.iterdir()) if p.is_dir()]

    return [x.strip() for x in exp_names_arg.split(",") if x.strip()]


def collect_sample_dirs(
    feature_dir: Path,
    max_samples: int,
    sample_names: list[str],
) -> list[Path]:
    """
    Collect sample directories from layer_feature_maps.

    Priority:
        1. If sample_names is provided, use exactly those sample folders.
        2. Otherwise use sample_000 ~ sample_{max_samples-1}.
        3. If max_samples < 0, include all sample-like directories.
    """
    if not feature_dir.exists() or not feature_dir.is_dir():
        return []

    if sample_names:
        sample_dirs: list[Path] = []
        for name in sample_names:
            sample_dir = feature_dir / name
            if sample_dir.exists() and sample_dir.is_dir():
                sample_dirs.append(sample_dir)
            else:
                print(f"[WARN] sample dir not found: {sample_dir}")
        return sample_dirs

    if max_samples is not None and max_samples >= 0:
        sample_dirs = []
        for i in range(max_samples):
            sample_dir = feature_dir / f"sample_{i:03d}"
            if sample_dir.exists() and sample_dir.is_dir():
                sample_dirs.append(sample_dir)
        return sample_dirs

    return [
        p for p in sorted(feature_dir.iterdir())
        if p.is_dir() and "sample" in p.name
    ]


def pack_experiment(
    zf: zipfile.ZipFile,
    exp_dir: Path,
    exp_name: str,
    feature_dir_name: str,
    include_weights: bool,
    include_feature_maps: bool,
    max_samples: int,
    sample_names: list[str],
    ignore_dirs: list[str],
) -> dict:
    """
    Pack one Stage2 experiment.

    Returns:
        stats dict for manifest.
    """
    stats = {
        "experiment": exp_name,
        "files_added": 0,
        "key_files": [],
        "weights": [],
        "feature_dir_name": feature_dir_name,
        "feature_maps_included": bool(include_feature_maps),
        "samples": [],
        "ignored_dirs": ignore_dirs,
    }

    print(f"[PACK] {exp_name}")

    # 1) Pack files outside weights and layer_feature_maps.
    # This preserves the original behavior: pack all useful metadata/log files
    # but avoid huge weights and huge feature maps unless explicitly requested.
    ignore_top_dirs = {"weights", feature_dir_name}

    for p in sorted(exp_dir.rglob("*")):
        if not p.is_file():
            continue

        rel = p.relative_to(exp_dir)

        if any(part in ignore_top_dirs for part in rel.parts[:-1]):
            continue

        arcname = Path(exp_name) / rel
        if add_file(zf, p, arcname):
            stats["files_added"] += 1
            if rel.name in DEFAULT_INCLUDE_FILES:
                stats["key_files"].append(rel.as_posix())

    # 2) Optional checkpoints.
    if include_weights:
        for ckpt_name in ("best.pt", "last.pt"):
            ckpt_path = exp_dir / "weights" / ckpt_name
            arcname = Path(exp_name) / "weights" / ckpt_name
            if add_file(zf, ckpt_path, arcname):
                stats["files_added"] += 1
                stats["weights"].append(f"weights/{ckpt_name}")

    # 3) Optional feature maps.
    if include_feature_maps:
        feature_dir = exp_dir / feature_dir_name

        if feature_dir.exists() and feature_dir.is_dir():
            # Root metadata of layer_feature_maps.
            for fname in ("meta.json", "summary.json"):
                src = feature_dir / fname
                arcname = Path(exp_name) / feature_dir_name / fname
                if add_file(zf, src, arcname):
                    stats["files_added"] += 1

            sample_dirs = collect_sample_dirs(
                feature_dir=feature_dir,
                max_samples=max_samples,
                sample_names=sample_names,
            )

            if not sample_dirs:
                print(f"[WARN] no sample dirs found under: {feature_dir}")

            for sample_dir in sample_dirs:
                added = add_dir(
                    zf=zf,
                    dir_path=sample_dir,
                    arcname_root=Path(exp_name) / feature_dir_name / sample_dir.name,
                    ignore_dirs=ignore_dirs,
                )
                stats["files_added"] += added
                stats["samples"].append({
                    "sample": sample_dir.name,
                    "files_added": added,
                })
        else:
            print(f"[WARN] feature dir not found: {feature_dir}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Pack Stage2 results, predictions, metadata, and selected layer feature maps for analysis."
    )

    parser.add_argument(
        "--runs_dir",
        type=str,
        default="./runs/glcp_stage2_yolo_det",
        help="Stage2 runs root directory.",
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
        default="./runs/glcp_stage2_yolo_det/stage2_results_bundle.zip",
        help="Output zip path.",
    )

    parser.add_argument(
        "--include_weights",
        action="store_true",
        help="Include weights/best.pt and weights/last.pt.",
    )

    parser.add_argument(
        "--include_feature_maps",
        action="store_true",
        help="Include selected samples from the layer feature map directory.",
    )

    parser.add_argument(
        "--feature_dir_name",
        type=str,
        default="layer_feature_maps",
        help="Feature map directory name inside each experiment.",
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
            "Do not ignore folders containing prediction/annotation/meta files if you want analysis."
        ),
    )

    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    output_zip = Path(args.output_zip).resolve()
    feature_dir_name = args.feature_dir_name.strip()

    ensure_parent(output_zip)

    if not runs_dir.exists() or not runs_dir.is_dir():
        raise FileNotFoundError(f"runs_dir not found: {runs_dir}")

    exp_names = collect_exp_names(runs_dir, args.exp_names)
    sample_names = parse_comma_list(args.sample_names)
    ignore_dirs = parse_comma_list(args.ignore_dirs)

    print("[INFO] Stage2 pack settings")
    print(f"       runs_dir             : {runs_dir}")
    print(f"       exp_names            : {exp_names}")
    print(f"       output_zip           : {output_zip}")
    print(f"       include_weights      : {args.include_weights}")
    print(f"       include_feature_maps : {args.include_feature_maps}")
    print(f"       feature_dir_name     : {feature_dir_name}")
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
                feature_dir_name=feature_dir_name,
                include_weights=args.include_weights,
                include_feature_maps=args.include_feature_maps,
                max_samples=args.max_samples,
                sample_names=sample_names,
                ignore_dirs=ignore_dirs,
            )
            stats_all.append(stats)

        manifest = {
            "runs_dir": str(runs_dir),
            "output_zip": str(output_zip),
            "experiments": stats_all,
            "include_weights": bool(args.include_weights),
            "include_feature_maps": bool(args.include_feature_maps),
            "feature_dir_name": feature_dir_name,
            "max_samples": int(args.max_samples),
            "sample_names": sample_names,
            "ignore_dirs": ignore_dirs,
            "notes": [
                "This pack is intended for Stage2 baseline/LS/RPD transfer comparison.",
                "results.csv and config files are packed automatically if present.",
                "weights are excluded unless --include_weights is used.",
                "layer feature maps are excluded unless --include_feature_maps is used.",
                "sample_names has priority over max_samples.",
                "For strict comparison, pack the same sample_names across baseline, LS transfer, and RPD transfer experiments.",
            ],
        }

        zf.writestr(
            "pack_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )

    print(f"\nDone. Saved to: {output_zip}")
    print(output_zip)


if __name__ == "__main__":
    main()