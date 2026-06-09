# -*- coding: utf-8 -*-
"""
PlantSeg YOLO label policy dry-run / variant generator.

Purpose
-------
This script is designed for PlantSeg mask-to-box converted YOLO labels.
It does NOT overwrite the original dataset.

It supports two low-risk label policies:

1) padding_only
   - Keep the number of boxes unchanged.
   - Add small padding to small boxes to reduce overly strict IoU mismatch.

2) pad_merge_conservative
   - First identify nearby same-class small/medium components.
   - Only auto-merge very conservative groups:
       member_count <= --merge-max-members
       union_area_ratio <= --merge-max-union-area-ratio
   - Then apply small-box padding to the final boxes.
   - Large/risky merge groups are NOT merged; they are written to review CSV.

Default mode is dry-run:
   - It writes CSV summaries and optional visualizations.
   - It does NOT create new training labels unless --apply is used.

If --apply is used:
   - Default: creates a new variant dataset root:
       <out_root>/<policy_name>/
         images/<split>/...  (hardlinks/copies/symlinks to original images)
         labels/<split>/...  (new labels)
         data.yaml
   - With --overwrite-labels-in-place: overwrites the original labels/<split>/*.txt only.
     Images are not copied/linked/modified, and no new data.yaml is required.
   - Original data is untouched.

Recommended first run:
   python tools/plantseg_make_label_policy_variant.py ^
     --data data/plantseg_singleclass.yaml ^
     --splits train,val ^
     --out-root runs_audit/plantseg_label_policy_v2 ^
     --policy pad_merge_conservative ^
     --save-visuals

After checking CSV/visuals, apply:
   python tools/plantseg_make_label_policy_variant.py ^
     --data data/plantseg_singleclass.yaml ^
     --splits train,val ^
     --out-root data/PlantSeg_singleclass_policyB ^
     --policy pad_merge_conservative ^
     --apply ^
     --image-mode hardlink
"""

import argparse
import csv
import math
import os
import shutil
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import yaml
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# -----------------------------
# basic IO
# -----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def write_csv(path: Path, rows: List[dict]) -> None:
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


def safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def safe_name(x: str) -> str:
    s = str(x).replace("\\", "_").replace("/", "_").replace(":", "_")
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)[:180]


def resolve_dataset_path(dataset_yaml: Path, root_value: str) -> Path:
    p = Path(root_value)
    if p.is_absolute():
        return p.resolve()
    return (dataset_yaml.parent / p).resolve()


def parse_names(names_obj) -> List[str]:
    if isinstance(names_obj, list):
        return [str(x) for x in names_obj]
    if isinstance(names_obj, dict):
        return [str(names_obj[k]) for k in sorted(names_obj.keys(), key=lambda x: int(x))]
    raise ValueError("Unsupported names format in data yaml.")


def resolve_split_path(dataset_root: Path, split_value) -> Path:
    if isinstance(split_value, list):
        if len(split_value) != 1:
            raise ValueError(f"This script expects one image dir per split, got: {split_value}")
        split_value = split_value[0]
    p = Path(str(split_value))
    if p.is_absolute():
        return p.resolve()
    return (dataset_root / p).resolve()


def infer_label_dir_from_img_dir(img_dir: Path) -> Path:
    parts = list(img_dir.parts)
    if "images" not in parts:
        raise ValueError(f"Cannot infer labels dir because 'images' not in path: {img_dir}")
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts)


def collect_images(img_dir: Path) -> List[Path]:
    return sorted([p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


def label_for_image(img: Path, img_root: Path, label_root: Path) -> Path:
    return label_root / img.relative_to(img_root).with_suffix(".txt")


# -----------------------------
# box geometry
# -----------------------------

def xywhn_to_xyxy(xc, yc, bw, bh, w, h) -> np.ndarray:
    x1 = (xc - bw / 2.0) * w
    y1 = (yc - bh / 2.0) * h
    x2 = (xc + bw / 2.0) * w
    y2 = (yc + bh / 2.0) * h
    return clip_box(np.array([x1, y1, x2, y2], dtype=np.float32), w, h)


def xyxy_to_yolo(box: np.ndarray, w: int, h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return xc / w, yc / h, bw / w, bh / h


def clip_box(box: np.ndarray, w: int, h: int) -> np.ndarray:
    b = box.astype(np.float32).copy()
    b[0] = max(0.0, min(float(w), b[0]))
    b[1] = max(0.0, min(float(h), b[1]))
    b[2] = max(0.0, min(float(w), b[2]))
    b[3] = max(0.0, min(float(h), b[3]))
    return b


def box_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def area_ratio(box: np.ndarray, w: int, h: int) -> float:
    return box_area(box) / float(w * h)


def union_box(boxes: List[np.ndarray], w: int, h: int) -> np.ndarray:
    arr = np.stack(boxes, axis=0)
    return clip_box(
        np.array([arr[:, 0].min(), arr[:, 1].min(), arr[:, 2].max(), arr[:, 3].max()], dtype=np.float32),
        w,
        h,
    )


def box_gap_px(a: np.ndarray, b: np.ndarray) -> float:
    # 0 if boxes overlap/touch in both axes, otherwise Euclidean gap between rectangles.
    dx = max(float(max(a[0], b[0]) - min(a[2], b[2])), 0.0)
    dy = max(float(max(a[1], b[1]) - min(a[3], b[3])), 0.0)
    return math.sqrt(dx * dx + dy * dy)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    ua = box_area(a) + box_area(b) - inter
    return inter / max(ua, 1e-9)


def pad_box(box: np.ndarray, w: int, h: int, padding_ratio: float, min_pad_px: float, max_pad_px: float) -> np.ndarray:
    bw = max(1.0, float(box[2] - box[0]))
    bh = max(1.0, float(box[3] - box[1]))
    px = min(max(min_pad_px, bw * padding_ratio), max_pad_px)
    py = min(max(min_pad_px, bh * padding_ratio), max_pad_px)
    return clip_box(np.array([box[0] - px, box[1] - py, box[2] + px, box[3] + py], dtype=np.float32), w, h)


# -----------------------------
# labels
# -----------------------------

def read_yolo_boxes(label_path: Path, img_w: int, img_h: int) -> List[dict]:
    boxes = []
    if not label_path.exists():
        return boxes
    lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line_id, line in enumerate(lines, start=1):
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
        box = xywhn_to_xyxy(xc, yc, bw, bh, img_w, img_h)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        boxes.append({
            "orig_id": len(boxes),
            "line_id": line_id,
            "cls": cls,
            "box": box,
            "area_ratio": area_ratio(box, img_w, img_h),
        })
    return boxes


def write_yolo_boxes(label_path: Path, boxes: List[dict], img_w: int, img_h: int) -> None:
    ensure_dir(label_path.parent)
    lines = []
    for b in boxes:
        xc, yc, bw, bh = xyxy_to_yolo(b["box"], img_w, img_h)
        lines.append(f"{int(b['cls'])} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# -----------------------------
# policy
# -----------------------------

def connected_components_from_edges(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    graph = [[] for _ in range(n)]
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    seen = [False] * n
    comps = []
    for i in range(n):
        if seen[i]:
            continue
        q = deque([i])
        seen[i] = True
        comp = []
        while q:
            u = q.popleft()
            comp.append(u)
            for v in graph[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        comps.append(sorted(comp))
    return comps


def find_nearby_groups(
    boxes: List[dict],
    img_w: int,
    img_h: int,
    gap_ratio: float,
    max_pair_union_area_ratio: float,
) -> List[List[int]]:
    if len(boxes) <= 1:
        return []
    gap_px = gap_ratio * min(img_w, img_h)
    edges = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if int(boxes[i]["cls"]) != int(boxes[j]["cls"]):
                continue
            bi = boxes[i]["box"]
            bj = boxes[j]["box"]
            if box_gap_px(bi, bj) > gap_px:
                continue
            ub = union_box([bi, bj], img_w, img_h)
            if area_ratio(ub, img_w, img_h) > max_pair_union_area_ratio:
                continue
            edges.append((i, j))
    comps = connected_components_from_edges(len(boxes), edges)
    return [c for c in comps if len(c) >= 2]


def apply_policy_to_image(
    img_path: Path,
    label_path: Path,
    args,
    policy_name: str,
) -> Tuple[List[dict], List[dict], List[dict], List[dict], dict]:
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {img_path}")
    h, w = img.shape[:2]
    orig_boxes = read_yolo_boxes(label_path, w, h)

    original_count = len(orig_boxes)
    original_small = sum(1 for b in orig_boxes if b["area_ratio"] < args.small_area_ratio)
    original_tiny = sum(1 for b in orig_boxes if b["area_ratio"] < args.tiny_area_ratio)

    groups = find_nearby_groups(
        orig_boxes,
        w,
        h,
        gap_ratio=args.merge_gap_ratio,
        max_pair_union_area_ratio=args.merge_pair_max_union_area_ratio,
    )

    merge_group_rows = []
    review_group_rows = []
    merge_members = set()
    final_boxes = []
    transform_rows = []

    auto_merge_groups = []
    if policy_name == "pad_merge_conservative":
        for gid, comp in enumerate(groups):
            cls_set = sorted(set(int(orig_boxes[i]["cls"]) for i in comp))
            ub = union_box([orig_boxes[i]["box"] for i in comp], w, h)
            ub_area = area_ratio(ub, w, h)
            member_count = len(comp)

            can_merge = (
                member_count <= args.merge_max_members
                and ub_area <= args.merge_max_union_area_ratio
                and len(cls_set) == 1
            )

            row = {
                "image": str(img_path),
                "label": str(label_path),
                "group_id": gid,
                "member_count": member_count,
                "member_orig_ids": ",".join(str(orig_boxes[i]["orig_id"]) for i in comp),
                "class_ids": ",".join(str(x) for x in cls_set),
                "union_area_ratio": ub_area,
                "union_xyxy": ",".join(f"{float(x):.2f}" for x in ub),
                "auto_merge": int(can_merge),
                "reason": "auto_merge" if can_merge else "review_only_large_or_risky_group",
            }

            if can_merge:
                auto_merge_groups.append(comp)
                merge_group_rows.append(row)
            else:
                review_group_rows.append(row)

    # create merged boxes
    new_id = 0
    for comp in auto_merge_groups:
        for i in comp:
            merge_members.add(i)
        cls = int(orig_boxes[comp[0]]["cls"])
        ub = union_box([orig_boxes[i]["box"] for i in comp], w, h)
        final_boxes.append({
            "new_id": new_id,
            "cls": cls,
            "box": ub,
            "source": "merged",
            "source_orig_ids": [int(orig_boxes[i]["orig_id"]) for i in comp],
            "padded": 0,
        })
        for i in comp:
            transform_rows.append({
                "image": str(img_path),
                "orig_id": int(orig_boxes[i]["orig_id"]),
                "new_id": new_id,
                "transform": "merged",
                "orig_area_ratio": float(orig_boxes[i]["area_ratio"]),
                "new_area_ratio": float(area_ratio(ub, w, h)),
            })
        new_id += 1

    # keep non-merged original boxes
    for i, b in enumerate(orig_boxes):
        if i in merge_members:
            continue
        final_boxes.append({
            "new_id": new_id,
            "cls": int(b["cls"]),
            "box": b["box"].copy(),
            "source": "original",
            "source_orig_ids": [int(b["orig_id"])],
            "padded": 0,
        })
        transform_rows.append({
            "image": str(img_path),
            "orig_id": int(b["orig_id"]),
            "new_id": new_id,
            "transform": "kept",
            "orig_area_ratio": float(b["area_ratio"]),
            "new_area_ratio": float(b["area_ratio"]),
        })
        new_id += 1

    # padding final boxes if small
    max_pad_px = args.max_pad_ratio * min(w, h)
    for b in final_boxes:
        before = b["box"].copy()
        before_area = area_ratio(before, w, h)
        if before_area < args.small_area_ratio:
            b["box"] = pad_box(
                before,
                w,
                h,
                padding_ratio=args.padding_ratio,
                min_pad_px=args.min_pad_px,
                max_pad_px=max_pad_px,
            )
            b["padded"] = 1
            after_area = area_ratio(b["box"], w, h)
            # add one summary row for padded final box
            transform_rows.append({
                "image": str(img_path),
                "orig_id": ",".join(map(str, b["source_orig_ids"])),
                "new_id": int(b["new_id"]),
                "transform": "padded_final_box",
                "orig_area_ratio": float(before_area),
                "new_area_ratio": float(after_area),
            })

    # sort for stable labels
    final_boxes = sorted(final_boxes, key=lambda x: (int(x["cls"]), float(x["box"][1]), float(x["box"][0])))
    for nid, b in enumerate(final_boxes):
        b["new_id"] = nid
        b["area_ratio"] = area_ratio(b["box"], w, h)

    final_count = len(final_boxes)
    final_small = sum(1 for b in final_boxes if b["area_ratio"] < args.small_area_ratio)
    final_tiny = sum(1 for b in final_boxes if b["area_ratio"] < args.tiny_area_ratio)

    image_summary = {
        "image": str(img_path),
        "label": str(label_path),
        "policy": policy_name,
        "image_w": w,
        "image_h": h,
        "original_boxes": original_count,
        "final_boxes": final_count,
        "box_delta": final_count - original_count,
        "original_small_boxes": original_small,
        "final_small_boxes": final_small,
        "original_tiny_boxes": original_tiny,
        "final_tiny_boxes": final_tiny,
        "auto_merge_groups": len(auto_merge_groups),
        "auto_merged_member_boxes": len(merge_members),
        "review_groups": len(review_group_rows),
        "padded_final_boxes": sum(int(b["padded"]) for b in final_boxes),
        "many_box_before": int(original_count >= args.many_box_threshold),
        "many_box_after": int(final_count >= args.many_box_threshold),
    }
    return orig_boxes, final_boxes, merge_group_rows, review_group_rows, image_summary, transform_rows


# -----------------------------
# visuals and dataset materialization
# -----------------------------

def draw_box(img, box, color, text="", thickness=2):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        t = 1
        (tw, th), base = cv2.getTextSize(text, font, scale, t)
        y0 = max(0, y1 - th - base - 2)
        cv2.rectangle(img, (x1, y0), (x1 + tw + 4, y0 + th + base + 4), color, -1)
        cv2.putText(img, text, (x1 + 2, y0 + th + 1), font, scale, (255, 255, 255), t, cv2.LINE_AA)


def save_visual(img_path: Path, out_path: Path, orig_boxes: List[dict], final_boxes: List[dict]):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    vis = img.copy()
    # original: green
    for b in orig_boxes:
        draw_box(vis, b["box"], (0, 180, 0), f"O{b['orig_id']}", 1)
    # final: blue/purple
    for b in final_boxes:
        color = (255, 120, 0) if b.get("source") == "merged" else (180, 0, 180)
        draw_box(vis, b["box"], color, f"N{b['new_id']}", 2)
    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), vis)


def link_or_copy_image(src: Path, dst: Path, mode: str) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    if mode == "none":
        return
    if mode == "copy":
        shutil.copy2(str(src), str(dst))
        return
    if mode == "symlink":
        try:
            os.symlink(str(src), str(dst))
            return
        except Exception:
            shutil.copy2(str(src), str(dst))
            return
    # default hardlink
    try:
        os.link(str(src), str(dst))
    except Exception:
        shutil.copy2(str(src), str(dst))


def process_policy(dataset_yaml: Path, cfg: dict, args, policy_name: str) -> None:
    dataset_root = resolve_dataset_path(dataset_yaml, cfg["path"])
    names = parse_names(cfg["names"])

    policy_root = Path(args.out_root).resolve() / policy_name
    audit_root = policy_root / "audit"
    ensure_dir(audit_root)

    all_image_summary = []
    all_merge_groups = []
    all_review_groups = []
    all_transform_rows = []
    global_rows = []

    processed_splits = []

    for split in args.splits:
        if split not in cfg or cfg.get(split) is None:
            print(f"[WARN] split {split} not found in yaml; skipped.")
            continue
        img_dir = resolve_split_path(dataset_root, cfg[split])
        label_dir = infer_label_dir_from_img_dir(img_dir)
        images = collect_images(img_dir)
        if not images:
            print(f"[WARN] no images found for split={split}: {img_dir}")
            continue
        processed_splits.append(split)

        split_summary = {
            "policy": policy_name,
            "split": split,
            "images": 0,
            "original_boxes": 0,
            "final_boxes": 0,
            "box_delta": 0,
            "original_small_boxes": 0,
            "final_small_boxes": 0,
            "original_tiny_boxes": 0,
            "final_tiny_boxes": 0,
            "auto_merge_groups": 0,
            "auto_merged_member_boxes": 0,
            "review_groups": 0,
            "padded_final_boxes": 0,
            "many_box_before": 0,
            "many_box_after": 0,
        }

        for img_path in images:
            lab_path = label_for_image(img_path, img_dir, label_dir)
            try:
                orig_boxes, final_boxes, merge_rows, review_rows, image_summary, transform_rows = apply_policy_to_image(
                    img_path=img_path,
                    label_path=lab_path,
                    args=args,
                    policy_name=policy_name,
                )
            except Exception as e:
                print(f"[WARN] failed image {img_path}: {e}")
                continue

            rel_img = img_path.relative_to(img_dir)
            image_summary["split"] = split
            all_image_summary.append(image_summary)

            for r in merge_rows:
                r["split"] = split
            for r in review_rows:
                r["split"] = split
            for r in transform_rows:
                r["split"] = split
                r["policy"] = policy_name

            all_merge_groups.extend(merge_rows)
            all_review_groups.extend(review_rows)
            all_transform_rows.extend(transform_rows)

            for k in split_summary.keys():
                if k in ["policy", "split"]:
                    continue
                if k == "images":
                    continue
                split_summary[k] += int(image_summary.get(k, 0))
            split_summary["images"] += 1

            if args.save_visuals:
                # save only changed or all depending on flag
                changed = image_summary["original_boxes"] != image_summary["final_boxes"] or image_summary["padded_final_boxes"] > 0
                if changed or args.save_all_visuals:
                    vis_path = audit_root / "visual_compare" / split / rel_img.with_suffix(".jpg")
                    save_visual(img_path, vis_path, orig_boxes, final_boxes)

            if args.apply:
                # Write labels. In overwrite mode, only original label txt files are modified.
                # Images are never copied/linked/modified in overwrite mode.
                img = cv2.imread(str(img_path))
                if img is None:
                    raise RuntimeError(f"Cannot read image for writing labels: {img_path}")
                h, w = img.shape[:2]

                if args.overwrite_labels_in_place:
                    dst_lab = lab_path
                    if args.backup_labels and lab_path.exists():
                        backup_path = lab_path.with_name(lab_path.name + args.backup_suffix)
                        if not backup_path.exists():
                            ensure_dir(backup_path.parent)
                            shutil.copy2(str(lab_path), str(backup_path))
                    write_yolo_boxes(dst_lab, final_boxes, w, h)
                else:
                    # materialize images and labels under variant root
                    dst_img = policy_root / "images" / split / rel_img
                    dst_lab = policy_root / "labels" / split / rel_img.with_suffix(".txt")
                    link_or_copy_image(img_path, dst_img, args.image_mode)
                    write_yolo_boxes(dst_lab, final_boxes, w, h)

        global_rows.append(split_summary)

    # Save audit outputs
    write_csv(audit_root / "policy_summary.csv", global_rows)
    write_csv(audit_root / "image_summary.csv", all_image_summary)
    write_csv(audit_root / "auto_merge_groups.csv", all_merge_groups)
    write_csv(audit_root / "review_merge_groups.csv", all_review_groups)
    write_csv(audit_root / "box_transform_detail.csv", all_transform_rows)

    # Save data yaml only when creating a new variant dataset.
    # In-place overwrite keeps the original data.yaml unchanged.
    if args.apply and not args.overwrite_labels_in_place:
        out_yaml = {
            "path": str(policy_root.resolve()),
            "names": names,
        }
        for split in processed_splits:
            out_yaml[split] = f"images/{split}"
        # Preserve nc if present
        if "nc" in cfg:
            out_yaml["nc"] = cfg["nc"]
        else:
            out_yaml["nc"] = len(names)
        write_yaml(policy_root / "data.yaml", out_yaml)

    print("=" * 80)
    print(f"[DONE] policy={policy_name}")
    if args.apply and args.overwrite_labels_in_place:
        mode_msg = "APPLY original labels overwritten in place; images untouched"
    elif args.apply:
        mode_msg = "APPLY labels/images created under variant root"
    else:
        mode_msg = "DRY-RUN only; no labels/images written"
    print(f"[MODE] {mode_msg}")
    print(f"[OUT] {policy_root}")
    print("[FILES]")
    print(f"  {audit_root / 'policy_summary.csv'}")
    print(f"  {audit_root / 'image_summary.csv'}")
    print(f"  {audit_root / 'auto_merge_groups.csv'}")
    print(f"  {audit_root / 'review_merge_groups.csv'}")
    print(f"  {audit_root / 'box_transform_detail.csv'}")
    if args.apply and not args.overwrite_labels_in_place:
        print(f"  {policy_root / 'data.yaml'}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, help="Original YOLO dataset yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--policy",
        default="pad_merge_conservative",
        choices=["padding_only", "pad_merge_conservative", "all"],
    )

    # Dry-run by default. Use --apply to create new labels/images/yaml.
    parser.add_argument("--apply", action="store_true")

    # image materialization mode when --apply is used and a new variant dataset is created.
    # Ignored when --overwrite-labels-in-place is used.
    parser.add_argument("--image-mode", default="hardlink", choices=["hardlink", "copy", "symlink", "none"])

    # In-place overwrite mode: only original labels/<split>/*.txt files are rewritten.
    # Images are not copied/linked/modified. Use only after backing up labels.
    parser.add_argument("--overwrite-labels-in-place", action="store_true")
    parser.add_argument("--backup-labels", action="store_true", help="Before overwriting, save one .bak copy next to each original label file.")
    parser.add_argument("--backup-suffix", default=".bak_policy_original")

    # size thresholds
    parser.add_argument("--small-area-ratio", type=float, default=0.03)
    parser.add_argument("--tiny-area-ratio", type=float, default=0.001)
    parser.add_argument("--many-box-threshold", type=int, default=20)

    # padding policy
    parser.add_argument("--padding-ratio", type=float, default=0.12)
    parser.add_argument("--min-pad-px", type=float, default=3.0)
    parser.add_argument("--max-pad-ratio", type=float, default=0.02)

    # merge policy
    parser.add_argument("--merge-gap-ratio", type=float, default=0.02)
    parser.add_argument("--merge-pair-max-union-area-ratio", type=float, default=0.08)
    parser.add_argument("--merge-max-members", type=int, default=3)
    parser.add_argument("--merge-max-union-area-ratio", type=float, default=0.08)

    # visuals
    parser.add_argument("--save-visuals", action="store_true")
    parser.add_argument("--save-all-visuals", action="store_true")

    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    cfg = read_yaml(data_yaml)

    if "path" not in cfg or "names" not in cfg:
        raise ValueError("Dataset yaml must contain 'path' and 'names'.")

    if args.overwrite_labels_in_place and not args.apply:
        raise ValueError("--overwrite-labels-in-place requires --apply")
    if args.overwrite_labels_in_place and args.policy == "all":
        raise ValueError("--overwrite-labels-in-place does not support --policy all. Choose one policy only.")

    policies = ["padding_only", "pad_merge_conservative"] if args.policy == "all" else [args.policy]

    print("[INFO] data:", data_yaml)
    print("[INFO] splits:", args.splits)
    print("[INFO] out_root:", Path(args.out_root).resolve())
    print("[INFO] apply:", args.apply)
    print("[INFO] overwrite_labels_in_place:", args.overwrite_labels_in_place)
    print("[INFO] backup_labels:", args.backup_labels)
    print("[INFO] policy:", args.policy)
    print("[INFO] thresholds:")
    print("  small_area_ratio:", args.small_area_ratio)
    print("  tiny_area_ratio:", args.tiny_area_ratio)
    print("  padding_ratio:", args.padding_ratio)
    print("  min_pad_px:", args.min_pad_px)
    print("  max_pad_ratio:", args.max_pad_ratio)
    print("  merge_gap_ratio:", args.merge_gap_ratio)
    print("  merge_max_members:", args.merge_max_members)
    print("  merge_max_union_area_ratio:", args.merge_max_union_area_ratio)

    for policy_name in policies:
        process_policy(data_yaml, cfg, args, policy_name)


if __name__ == "__main__":
    main()
