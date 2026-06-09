# -*- coding: utf-8 -*-
"""
PlantSeg mask-to-box / YOLO-label dry-run audit.

Purpose
-------
This script DOES NOT modify labels. It inspects current YOLO labels and optionally
original masks, then reports whether the current mask-to-box conversion may be
creating problems such as:
  - very tiny fragmented boxes
  - many close boxes that may need merging
  - small boxes that may need padding
  - elongated / suspicious boxes
  - image-level over-fragmentation

Outputs
-------
- audit_boxes.csv
- image_summary.csv
- dryrun_merge_groups.csv
- dryrun_proposed_boxes.csv
- policy_summary.csv
- README_dryrun.md
- optional visualizations if --save-visuals is used

This is a dry-run diagnostic. It never overwrites existing labels.
"""

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[dict]):
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base: Path, p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (base / pp).resolve()


def parse_dataset_yaml(data_yaml: Path, split: str):
    data = load_yaml(data_yaml)
    root = Path(data.get("path", "."))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    split_value = data.get(split)
    if split_value is None:
        raise ValueError(f"Dataset yaml has no split: {split}")
    if isinstance(split_value, list):
        if len(split_value) != 1:
            raise ValueError(f"Expected one path for split {split}, got {split_value}")
        split_value = split_value[0]
    img_dir = resolve_path(root, str(split_value))
    parts = list(img_dir.parts)
    if "images" not in parts:
        raise ValueError(f"Cannot infer labels dir from image dir without 'images': {img_dir}")
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    label_dir = Path(*parts)
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    return root, img_dir, label_dir, list(names)


def collect_images(img_dir: Path) -> List[Path]:
    return sorted([p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


def label_for_image(img: Path, img_dir: Path, label_dir: Path) -> Path:
    return label_dir / img.relative_to(img_dir).with_suffix(".txt")


def yolo_to_xyxy(xc, yc, bw, bh, W, H):
    x1 = (xc - bw / 2) * W
    y1 = (yc - bh / 2) * H
    x2 = (xc + bw / 2) * W
    y2 = (yc + bh / 2) * H
    x1 = max(0.0, min(float(W), x1))
    y1 = max(0.0, min(float(H), y1))
    x2 = max(0.0, min(float(W), x2))
    y2 = max(0.0, min(float(H), y2))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def xyxy_to_yolo(box, W, H):
    x1, y1, x2, y2 = [float(x) for x in box]
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2
    yc = y1 + bh / 2
    return xc / W, yc / H, bw / W, bh / H


def box_area(box):
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def box_aspect(box):
    w = max(1e-9, float(box[2] - box[0]))
    h = max(1e-9, float(box[3] - box[1]))
    return max(w / h, h / w)


def pad_box(box, W, H, pad_ratio=0.10, min_pad_px=2):
    x1, y1, x2, y2 = [float(x) for x in box]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    px = max(float(min_pad_px), w * float(pad_ratio))
    py = max(float(min_pad_px), h * float(pad_ratio))
    return np.array([
        max(0.0, x1 - px),
        max(0.0, y1 - py),
        min(float(W), x2 + px),
        min(float(H), y2 + py),
    ], dtype=np.float32)


def union_box(boxes: List[np.ndarray]) -> np.ndarray:
    arr = np.vstack(boxes).astype(np.float32)
    return np.array([arr[:, 0].min(), arr[:, 1].min(), arr[:, 2].max(), arr[:, 3].max()], dtype=np.float32)


def rect_gap(b1, b2) -> Tuple[float, float]:
    gx = max(0.0, max(float(b1[0]), float(b2[0])) - min(float(b1[2]), float(b2[2])))
    gy = max(0.0, max(float(b1[1]), float(b2[1])) - min(float(b1[3]), float(b2[3])))
    return gx, gy


def read_yolo_labels(label_path: Path, W: int, H: int) -> List[dict]:
    if not label_path.exists():
        return []
    out = []
    for line_id, line in enumerate(label_path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:])
        except Exception:
            continue
        box = yolo_to_xyxy(xc, yc, bw, bh, W, H)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        out.append({
            "line_id": line_id,
            "cls": cls,
            "xyxy": box,
            "area_ratio": box_area(box) / float(W * H),
            "w_px": float(box[2] - box[0]),
            "h_px": float(box[3] - box[1]),
        })
    return out


class DSU:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def find_merge_groups(labels: List[dict], W: int, H: int, args) -> List[List[int]]:
    n = len(labels)
    if n <= 1:
        return []
    dsu = DSU(n)
    gap_thr = max(float(args.merge_gap_px), float(args.merge_gap_ratio) * min(W, H))
    for i in range(n):
        for j in range(i + 1, n):
            if int(labels[i]["cls"]) != int(labels[j]["cls"]):
                continue
            b1, b2 = labels[i]["xyxy"], labels[j]["xyxy"]
            gx, gy = rect_gap(b1, b2)
            if gx <= gap_thr and gy <= gap_thr:
                ub = union_box([b1, b2])
                if box_area(ub) / float(W * H) <= float(args.max_merge_area_ratio):
                    dsu.union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(dsu.find(i), []).append(i)
    return [g for g in groups.values() if len(g) >= 2]


def guess_mask_path(img_path: Path, img_dir: Path, mask_root: Optional[Path]) -> Optional[Path]:
    if mask_root is None:
        return None
    rel = img_path.relative_to(img_dir)
    for ext in MASK_EXTS:
        cand = mask_root / rel.with_suffix(ext)
        if cand.exists():
            return cand
    return None


def mask_components(mask_path: Optional[Path], min_component_area_px: int = 1) -> Tuple[int, int]:
    if mask_path is None or not mask_path.exists():
        return 0, 0
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0, 0
    bin_mask = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    total = 0
    kept = 0
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        total += 1
        if area >= int(min_component_area_px):
            kept += 1
    return total, kept


def draw_visual(img_path: Path, out_path: Path, labels: List[dict], merge_groups: List[List[int]], args):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    # current boxes: green
    for idx, lab in enumerate(labels):
        x1, y1, x2, y2 = [int(round(float(x))) for x in lab["xyxy"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(img, f"#{idx+1}", (x1, max(0, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 0), 1)
    # padded small boxes: blue
    H, W = img.shape[:2]
    for lab in labels:
        if lab["area_ratio"] < args.small_area_ratio:
            pb = pad_box(lab["xyxy"], W, H, args.pad_ratio, args.min_pad_px)
            x1, y1, x2, y2 = [int(round(float(x))) for x in pb]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    # merge groups union: yellow
    for gid, g in enumerate(merge_groups, 1):
        ub = union_box([labels[i]["xyxy"] for i in g])
        x1, y1, x2, y2 = [int(round(float(x))) for x in ub]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 220), 3)
        cv2.putText(img, f"MERGE{gid}", (x1, min(img.shape[0]-5, y2 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1)
    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="YOLO dataset yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mask-root", default="", help="Optional original mask root. If empty, only YOLO labels are audited.")
    ap.add_argument("--small-area-ratio", type=float, default=0.03)
    ap.add_argument("--tiny-area-ratio", type=float, default=0.001)
    ap.add_argument("--min-box-side-px", type=float, default=4.0)
    ap.add_argument("--max-aspect-ratio", type=float, default=6.0)
    ap.add_argument("--many-box-threshold", type=int, default=20)
    ap.add_argument("--merge-gap-px", type=float, default=12.0)
    ap.add_argument("--merge-gap-ratio", type=float, default=0.015)
    ap.add_argument("--max-merge-area-ratio", type=float, default=0.15)
    ap.add_argument("--pad-ratio", type=float, default=0.10)
    ap.add_argument("--min-pad-px", type=float, default=2.0)
    ap.add_argument("--save-visuals", action="store_true")
    ap.add_argument("--max-visuals", type=int, default=500)
    args = ap.parse_args()

    data_yaml = Path(args.data).resolve()
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    mask_root = Path(args.mask_root).resolve() if args.mask_root else None

    box_rows = []
    img_rows = []
    merge_rows = []
    proposed_rows = []
    visual_count = 0

    for split in splits:
        root, img_dir, label_dir, names = parse_dataset_yaml(data_yaml, split)
        images = collect_images(img_dir)
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]
            label_path = label_for_image(img_path, img_dir, label_dir)
            labs = read_yolo_labels(label_path, W, H)
            groups = find_merge_groups(labs, W, H, args)
            mask_path = guess_mask_path(img_path, img_dir, mask_root)
            mask_comp_total, mask_comp_kept = mask_components(mask_path, min_component_area_px=1)

            tiny_count = 0
            small_count = 0
            elongated_count = 0
            invalid_like_count = 0

            group_member_set = set()
            for g in groups:
                group_member_set.update(g)

            for idx, lab in enumerate(labs):
                box = lab["xyxy"]
                ar = lab["area_ratio"]
                asp = box_aspect(box)
                is_tiny = ar < args.tiny_area_ratio or lab["w_px"] < args.min_box_side_px or lab["h_px"] < args.min_box_side_px
                is_small = ar < args.small_area_ratio
                is_elongated = asp > args.max_aspect_ratio
                is_merge_candidate = idx in group_member_set
                if is_tiny:
                    tiny_count += 1
                if is_small:
                    small_count += 1
                if is_elongated:
                    elongated_count += 1
                if lab["w_px"] <= 0 or lab["h_px"] <= 0:
                    invalid_like_count += 1
                pb = pad_box(box, W, H, args.pad_ratio, args.min_pad_px) if is_small else box
                yolo_padded = xyxy_to_yolo(pb, W, H)
                flags = []
                if is_tiny:
                    flags.append("tiny_fragment_candidate")
                if is_small:
                    flags.append("small_box_padding_candidate")
                if is_elongated:
                    flags.append("elongated_box_candidate")
                if is_merge_candidate:
                    flags.append("nearby_merge_candidate")
                box_rows.append({
                    "split": split,
                    "image": str(img_path),
                    "label": str(label_path),
                    "label_exists": int(label_path.exists()),
                    "image_w": W,
                    "image_h": H,
                    "line_id": lab["line_id"],
                    "class_id": lab["cls"],
                    "class_name": names[lab["cls"]] if 0 <= lab["cls"] < len(names) else "",
                    "x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3]),
                    "w_px": lab["w_px"],
                    "h_px": lab["h_px"],
                    "area_ratio": ar,
                    "aspect_ratio_max": asp,
                    "is_tiny_fragment_candidate": int(is_tiny),
                    "is_small_box_padding_candidate": int(is_small),
                    "is_elongated_box_candidate": int(is_elongated),
                    "is_nearby_merge_candidate": int(is_merge_candidate),
                    "flags": ";".join(flags),
                    "dryrun_padded_x1": float(pb[0]), "dryrun_padded_y1": float(pb[1]),
                    "dryrun_padded_x2": float(pb[2]), "dryrun_padded_y2": float(pb[3]),
                    "dryrun_padded_yolo": " ".join([f"{v:.6f}" for v in yolo_padded]),
                })
                action = "keep"
                if is_tiny:
                    action = "review_tiny_do_not_auto_delete"
                elif is_small:
                    action = "candidate_pad"
                proposed_rows.append({
                    "split": split,
                    "image": str(img_path),
                    "source": "single_box_policy",
                    "action": action,
                    "class_id": lab["cls"],
                    "member_line_ids": str([lab["line_id"]]),
                    "proposed_xyxy": ",".join([f"{float(x):.2f}" for x in pb]),
                    "proposed_yolo": " ".join([f"{v:.6f}" for v in yolo_padded]),
                    "note": "dry-run only; not written to labels",
                })

            for gid, g in enumerate(groups, 1):
                ub = union_box([labs[i]["xyxy"] for i in g])
                yolo_u = xyxy_to_yolo(ub, W, H)
                merge_rows.append({
                    "split": split,
                    "image": str(img_path),
                    "group_id": gid,
                    "class_id": labs[g[0]]["cls"] if g else "",
                    "class_name": names[labs[g[0]]["cls"]] if g and 0 <= labs[g[0]]["cls"] < len(names) else "",
                    "member_count": len(g),
                    "member_line_ids": str([labs[i]["line_id"] for i in g]),
                    "union_area_ratio": box_area(ub) / float(W * H),
                    "union_xyxy": ",".join([f"{float(x):.2f}" for x in ub]),
                    "union_yolo": " ".join([f"{v:.6f}" for v in yolo_u]),
                    "dryrun_action": "candidate_merge_close_components",
                })
                proposed_rows.append({
                    "split": split,
                    "image": str(img_path),
                    "source": "merge_group_policy",
                    "action": "candidate_merge_close_components",
                    "class_id": labs[g[0]]["cls"] if g else "",
                    "member_line_ids": str([labs[i]["line_id"] for i in g]),
                    "proposed_xyxy": ",".join([f"{float(x):.2f}" for x in ub]),
                    "proposed_yolo": " ".join([f"{v:.6f}" for v in yolo_u]),
                    "note": "dry-run only; manually review before label conversion",
                })

            img_rows.append({
                "split": split,
                "image": str(img_path),
                "label": str(label_path),
                "label_exists": int(label_path.exists()),
                "box_count": len(labs),
                "small_box_count": small_count,
                "tiny_box_count": tiny_count,
                "elongated_box_count": elongated_count,
                "merge_group_count": len(groups),
                "merge_member_box_count": len(group_member_set),
                "many_boxes_flag": int(len(labs) >= args.many_box_threshold),
                "mask_path": str(mask_path) if mask_path else "",
                "mask_component_total": mask_comp_total,
                "mask_component_kept": mask_comp_kept,
            })

            if args.save_visuals and visual_count < args.max_visuals:
                if len(groups) > 0 or tiny_count > 0 or small_count > 0 or len(labs) >= args.many_box_threshold:
                    rel = img_path.relative_to(img_dir)
                    out_img = out_dir / "visual_dryrun" / split / rel.with_suffix(".jpg")
                    draw_visual(img_path, out_img, labs, groups, args)
                    visual_count += 1

    write_csv(out_dir / "audit_boxes.csv", box_rows)
    write_csv(out_dir / "image_summary.csv", img_rows)
    write_csv(out_dir / "dryrun_merge_groups.csv", merge_rows)
    write_csv(out_dir / "dryrun_proposed_boxes.csv", proposed_rows)

    def count_rows(rows, key):
        return sum(int(r.get(key, 0)) for r in rows)

    total_boxes = len(box_rows)
    total_images = len(img_rows)
    summary = [{
        "total_images": total_images,
        "total_boxes": total_boxes,
        "small_box_candidates": count_rows(box_rows, "is_small_box_padding_candidate"),
        "tiny_fragment_candidates": count_rows(box_rows, "is_tiny_fragment_candidate"),
        "elongated_box_candidates": count_rows(box_rows, "is_elongated_box_candidate"),
        "nearby_merge_candidate_boxes": count_rows(box_rows, "is_nearby_merge_candidate"),
        "merge_groups": len(merge_rows),
        "many_box_images": count_rows(img_rows, "many_boxes_flag"),
        "small_box_ratio": (count_rows(box_rows, "is_small_box_padding_candidate") / total_boxes) if total_boxes else 0.0,
        "tiny_box_ratio": (count_rows(box_rows, "is_tiny_fragment_candidate") / total_boxes) if total_boxes else 0.0,
        "merge_candidate_box_ratio": (count_rows(box_rows, "is_nearby_merge_candidate") / total_boxes) if total_boxes else 0.0,
    }]
    write_csv(out_dir / "policy_summary.csv", summary)

    readme = out_dir / "README_dryrun.md"
    readme.write_text(
        "# PlantSeg box-conversion dry-run audit\n\n"
        "This audit does not change labels. It flags possible conversion problems.\n\n"
        "## Interpretation\n\n"
        "- `tiny_fragment_candidate`: very tiny boxes. Do not auto-delete; visually review first.\n"
        "- `small_box_padding_candidate`: small boxes that may benefit from slight padding.\n"
        "- `nearby_merge_candidate`: close boxes that may represent fragmented mask components.\n"
        "- `elongated_box_candidate`: unusual long/thin boxes that may cause localization mismatch.\n"
        "- `many_boxes_flag`: image has many boxes; may be over-fragmented.\n\n"
        "## Recommended next step\n\n"
        "Open `policy_summary.csv` first, then review `dryrun_merge_groups.csv` and visualizations. "
        "Only after manual review should a real label-conversion script be applied.\n",
        encoding="utf-8",
    )

    print("[DONE] Dry-run audit finished.")
    print(f"[OUT] {out_dir}")
    print("[FILES] audit_boxes.csv, image_summary.csv, dryrun_merge_groups.csv, dryrun_proposed_boxes.csv, policy_summary.csv")


if __name__ == "__main__":
    main()
