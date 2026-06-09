from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable


ANALYSIS_DIR_NAMES = [
    "analysis_stage1_prior_quality",
    "analysis_stage2_candidate_quality",
    "analysis_stage2_support_transfer",
]


def project_root() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name.lower() == "tools":
        return here.parent.parent
    return here.parent


def add_dir_to_zip(zf: zipfile.ZipFile, src_dir: Path, arc_prefix: str = "", exclude_suffixes: set[str] | None = None) -> int:
    exclude_suffixes = exclude_suffixes or set()
    count = 0
    if not src_dir.exists() or not src_dir.is_dir():
        return 0
    for p in sorted(src_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in exclude_suffixes:
            continue
        rel = p.relative_to(src_dir.parent).as_posix()
        if arc_prefix:
            rel = f"{arc_prefix.rstrip('/')}/{rel}"
        zf.write(p, rel)
        count += 1
    return count


def resolve_analysis_dirs(runs_dir: Path) -> list[Path]:
    """Find the three standard analysis directories under a given run directory.

    It supports both exact standard names and version-suffixed names, e.g.:
      analysis_stage1_prior_quality
      analysis_stage1_prior_quality_yolov8n
    """
    out: list[Path] = []
    for name in ANALYSIS_DIR_NAMES:
        exact = runs_dir / name
        if exact.exists() and exact.is_dir():
            out.append(exact)
            continue
        matches = sorted([p for p in runs_dir.glob(f"{name}*") if p.is_dir()])
        if matches:
            out.extend(matches)
    # de-duplicate while preserving order
    seen = set()
    deduped = []
    for d in out:
        key = d.resolve()
        if key not in seen:
            deduped.append(d)
            seen.add(key)
    return deduped


def pack_analysis_dirs(runs_dir: Path, out_zip: Path, *, exclude_suffixes: set[str]) -> dict:
    runs_dir = runs_dir.resolve()
    out_zip = out_zip.resolve()
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    analysis_dirs = resolve_analysis_dirs(runs_dir)
    reports = []
    total_files = 0

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for d in analysis_dirs:
            file_count = add_dir_to_zip(zf, d, exclude_suffixes=exclude_suffixes)
            total_files += file_count
            reports.append({
                "name": d.name,
                "source": str(d),
                "exists": True,
                "files": file_count,
            })

        for name in ANALYSIS_DIR_NAMES:
            has_any = any(r["name"].startswith(name) for r in reports)
            if not has_any:
                reports.append({
                    "name": name,
                    "source": str(runs_dir / name),
                    "exists": False,
                    "files": 0,
                })

    summary = {
        "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "runs_dir": str(runs_dir),
        "out_zip": str(out_zip),
        "total_files": total_files,
        "analysis_dirs": reports,
    }
    summary_path = out_zip.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack the three paper-scheme analysis directories under a specified runs_dir into one zip file."
    )
    parser.add_argument(
        "--runs_dir",
        required=True,
        help="Directory containing analysis_stage1_prior_quality, analysis_stage2_candidate_quality, and analysis_stage2_support_transfer. Usually ./runs or a version-specific runs directory.",
    )
    parser.add_argument(
        "--out_zip",
        default="",
        help="Output zip path. Default: <runs_dir>/analysis_dirs_bundle_<timestamp>.zip",
    )
    parser.add_argument(
        "--exclude_weights",
        action="store_true",
        help="Exclude .pt/.pth files if any exist under analysis directories.",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = project_root() / runs_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_zip = Path(args.out_zip) if args.out_zip else runs_dir / f"analysis_dirs_bundle_{timestamp}.zip"
    if not out_zip.is_absolute():
        out_zip = project_root() / out_zip

    exclude_suffixes = {".pt", ".pth"} if args.exclude_weights else set()
    summary = pack_analysis_dirs(runs_dir, out_zip, exclude_suffixes=exclude_suffixes)

    print("=" * 80)
    print("[PACK] Analysis directories packed into one zip")
    print(f"runs_dir = {summary['runs_dir']}")
    print(f"OUT_ZIP  = {summary['out_zip']}")
    print(f"FILES    = {summary['total_files']}")
    print("-" * 80)
    for r in summary["analysis_dirs"]:
        status = "OK" if r["exists"] else "MISSING"
        print(f"[{status}] {r['name']} | files={r['files']} | source={r['source']}")
    print(f"[OK] summary: {Path(summary['out_zip']).with_suffix('.json')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
