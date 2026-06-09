from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# (source zip relative to runs_dir, output directory inside merged zip)
# The output directory keeps the original run directory structure.
REQUIRED_ZIPS: List[Tuple[str, str]] = [
    ("analysis_results_bundle.zip", ""),
    ("glcp_stage1_yolo_det/stage1_results_bundle.zip", "glcp_stage1_yolo_det"),
    ("glcp_stage2_yolo_det/stage2_results_bundle.zip", "glcp_stage2_yolo_det"),
]


def _norm_zip_name(name: str) -> str:
    """Normalize zip entry names for consistent forward-slash paths."""
    return name.replace("\\", "/").lstrip("/")


def _under_prefix(name: str, prefix: str) -> bool:
    name = _norm_zip_name(name)
    prefix = prefix.strip("/")
    return name == prefix or name.startswith(prefix + "/")


def merge_zip_into(zout: zipfile.ZipFile, src_zip: Path, prefix: str) -> int:
    """Merge src_zip into zout while preserving the intended run-dir structure.

    Example:
      runs_yolov8n/glcp_stage1_yolo_det/stage1_bundle_results.zip
    is expanded under:
      glcp_stage1_yolo_det/<original contents>

    If the inner zip already contains the prefix directory, the prefix is not
    duplicated.
    """
    count = 0
    prefix = prefix.strip("/")
    with zipfile.ZipFile(src_zip, "r") as zin:
        for info in zin.infolist():
            if info.is_dir():
                continue
            inner_name = _norm_zip_name(info.filename)
            if not inner_name:
                continue

            # Avoid duplicate paths such as analysis_stage1_prior_quality/analysis_stage1_prior_quality/...
            if _under_prefix(inner_name, prefix):
                arcname = inner_name
            else:
                arcname = f"{prefix}/{inner_name}" if prefix else f"{inner_name}"

            zout.writestr(arcname, zin.read(info.filename))
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge five paper-scheme zip bundles under a runs_dir into one zip while keeping original directory structure."
    )
    parser.add_argument(
        "--runs_dir",
        required=True,
        help="Directory containing the five zip bundles, e.g. ./runs_yolov8n",
    )
    parser.add_argument(
        "--out_zip",
        default="",
        help="Output zip path. Default: <runs_dir>.zip",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    if not runs_dir.exists() or not runs_dir.is_dir():
        raise FileNotFoundError(f"runs_dir not found or not a directory: {runs_dir}")

    out_zip = Path(args.out_zip).resolve() if args.out_zip else runs_dir.with_suffix(".zip")
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    # overwrite defaults to true
    if out_zip.exists():
        out_zip.unlink()

    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "runs_dir": str(runs_dir),
        "out_zip": str(out_zip),
        "overwrite": True,
        "keep_original_directory_structure": True,
        "inputs": [],
    }

    total_files = 0
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zout:
        for rel_zip, output_prefix in REQUIRED_ZIPS:
            src = runs_dir / rel_zip
            item = {
                "source": str(src),
                "output_prefix": output_prefix,
                "exists": src.exists(),
                "files": 0,
            }
            if src.exists() and src.is_file():
                item["files"] = merge_zip_into(zout, src, output_prefix)
                total_files += int(item["files"])
            summary["inputs"].append(item)

        summary["total_files"] = total_files
        zout.writestr("merge_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))

    summary_path = out_zip.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("[OK] Merged run zip bundles")
    print(f"RUNS_DIR = {runs_dir}")
    print(f"OUT_ZIP  = {out_zip}")
    print("-" * 80)
    for item in summary["inputs"]:
        status = "OK" if item["exists"] else "MISSING"
        print(f"[{status}] {item['source']} -> {item['output_prefix']} | files={item['files']}")
    print(f"[OK] summary: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
