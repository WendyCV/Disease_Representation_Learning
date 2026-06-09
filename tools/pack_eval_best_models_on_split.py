# -*- coding: utf-8 -*-
"""
Pack files from:
1) runs_yolov8n_xxx/test_eval_best_models_xxx/
2) runs_yolov8n_xxx/glcp_stage1_yolo_det/use_pos_mask_rpd_hybrid_w020/config_used.yaml

Output:
runs_yolov8n_xxx/test_eval_best_models_xxx_timestamp.zip

Zip rule:
- Do not keep folders inside the zip.
- All files are placed flatly in the zip.
- If duplicated filenames exist, the script automatically adds suffixes.
"""

import os
import argparse
import zipfile
from pathlib import Path
from datetime import datetime
import shutil


def find_test_eval_dir(run_dir: Path, test_eval_name: str, out_suffix: str | None = None) -> Path:
    if out_suffix:
        test_eval_name = f"{test_eval_name}_{out_suffix}"
    test_eval_dir = run_dir / test_eval_name
    if not test_eval_dir.exists() or not test_eval_dir.is_dir():
        raise FileNotFoundError(f"Cannot find test_eval directory: {test_eval_dir}")
    return test_eval_dir


def make_unique_name(filename: str, used_names: set[str]) -> str:
    """
    Keep zip internal files flat.
    If duplicated basename exists, rename:
    results.csv -> results_dup01.csv
    """
    if filename not in used_names:
        used_names.add(filename)
        return filename

    path = Path(filename)
    stem = path.stem
    suffix = path.suffix

    idx = 1
    while True:
        new_name = f"{stem}_dup{idx:02d}{suffix}"
        if new_name not in used_names:
            used_names.add(new_name)
            return new_name
        idx += 1


def collect_files_flat(test_eval_dir: Path) -> list[Path]:
    """
    Only collect files directly under test_eval_dir.
    Do not collect files from subfolders.
    """
    files = []
    for p in sorted(test_eval_dir.iterdir()):
        if p.is_file():
            files.append(p)
    return files


def pack_results(run_dir: Path, test_eval_name: str, out_suffix: str | None = None) -> Path:
    run_dir = run_dir.resolve()

    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Cannot find run directory: {run_dir}")

    test_eval_dir = find_test_eval_dir(run_dir, test_eval_name, out_suffix)

    stage1_config = (
        run_dir
        / "glcp_stage1_yolo_det"
        / "use_pos_mask_rpd_hybrid_w020"
        / "config_used.yaml"
    )

    stage1_train_log = (
        run_dir
        / "glcp_stage1_yolo_det"
        / "use_pos_mask_rpd_hybrid_w020"
        / "train_log.csv"
    )

    timestamp = datetime.now().strftime("%H%M")
    zip_name = f"{test_eval_dir.name}_{timestamp}.zip"
    zip_path = run_dir / zip_name

    used_names = set()
    packed_count = 0

    # 先删除文件
    if test_eval_dir.joinpath("stage1_rpd_config_used.yaml").exists():
        os.remove(test_eval_dir.joinpath("stage1_rpd_config_used.yaml"))
    if test_eval_dir.joinpath("stage1_rpd_train_log.csv").exists():
        os.remove(test_eval_dir.joinpath("stage1_rpd_train_log.csv"))

    test_eval_files = collect_files_flat(test_eval_dir)

    if len(test_eval_files) == 0:
        raise FileNotFoundError(f"No files found under: {test_eval_dir}")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        print(f"[INFO] Packing files from: {test_eval_dir}")

        for file_path in test_eval_files:
            arcname = make_unique_name(file_path.name, used_names)
            zf.write(file_path, arcname=arcname)
            packed_count += 1
            print(f"  + {file_path.name}  ->  {arcname}")

        if stage1_config.exists():
            arcname = make_unique_name("stage1_rpd_config_used.yaml", used_names)
            zf.write(stage1_config, arcname=arcname)
            packed_count += 1
            print(f"  + {stage1_config}  ->  {arcname}")
            # copy here
            shutil.copy2(stage1_config, test_eval_dir.joinpath("stage1_rpd_config_used.yaml"))
        else:
            print(f"[WARNING] Stage1 config_used.yaml not found: {stage1_config}")

        if stage1_train_log.exists():
            arcname = make_unique_name("stage1_rpd_train_log.csv", used_names)
            zf.write(stage1_train_log, arcname=arcname)
            packed_count += 1
            print(f"  + {stage1_train_log}  ->  {arcname}")
            # copy here
            shutil.copy2(stage1_train_log, test_eval_dir.joinpath("stage1_rpd_train_log.csv"))
        else:
            print(f"[WARNING] Stage1 train_log.csv not found: {stage1_train_log}")

    print("\n[DONE]")
    print(f"Packed files: {packed_count}")
    print(f"Output zip: {zip_path}")

    return zip_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pack test_eval_best_models_xxx files and Stage1 RPD config_used.yaml."
    )

    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to runs_yolov8n_xxx directory.",
    )

    parser.add_argument(
        "--test_eval_name",
        type=str,
        default="test_eval_best_models",
        help=(
            "Optional. Exact test_eval_best_models_xxx folder name. "
            "Use this when multiple test_eval_best_models folders exist."
        ),
    )

    parser.add_argument(
        "--out_suffix",
        type=str,
        default=None,
        help=(
            "Optional. Exact test_eval_best_models_xxx folder suffix name. "
            "Use this when multiple test_eval_best_models folders exist."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pack_results(
        run_dir=Path(args.run_dir),
        test_eval_name=args.test_eval_name,
        out_suffix=args.out_suffix,
    )