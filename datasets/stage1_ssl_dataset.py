from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from datasets.augmentations import build_base_transform, build_geometric_transform, build_image_only_transform

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Stage1ContrastiveDataset(Dataset):
    """Stage-1 unlabeled dataset used by the SSL pretraining pipeline.

    Design goals:
    1. Keep the data path logic explicit and easy to audit.
    2. Isolate mask loading / post-processing from augmentation logic.
    3. Return two synchronized views plus masks for contrastive training.

    The dataset intentionally keeps *all* stage-1 assumptions in one place so that
    ablation experiments remain easy to trace.
    """

    def __init__(
        self,
        root_dir: str,
        image_size: int = 640,
        mask_root_dir: str | None = None,
        mask_mode: str = "sam2",
        edge_mask: bool = False,
        min_valid_pixels: int = 4096,
        use_cache: bool = True,
        mask_suffix: str = ".png",
        missing_mask_policy: str = "ones",
        external_mask_threshold: int = 127,
        external_mask_median_blur: int = 0,
        external_mask_open_kernel: int = 0,
        external_mask_close_kernel: int = 0,
        extra_image_prefixes: Sequence[str] | str | None = ("DSC_",),
        extra_max_images: int | None = None,
        extra_ratio_to_base: float | None = None,
        extra_sampling: str = "even",
        extra_random_seed: int = 2026,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.image_size = image_size
        self.mask_root_dir = mask_root_dir
        self.mask_mode = mask_mode
        self.edge_mask = edge_mask
        self.min_valid_pixels = min_valid_pixels
        self.use_cache = use_cache

        self.mask_suffix = mask_suffix
        self.missing_mask_policy = missing_mask_policy
        self.external_mask_threshold = external_mask_threshold
        self.external_mask_median_blur = external_mask_median_blur
        self.external_mask_open_kernel = external_mask_open_kernel
        self.external_mask_close_kernel = external_mask_close_kernel

        self.extra_image_prefixes = self._normalize_prefixes(extra_image_prefixes)
        self.extra_max_images = extra_max_images
        self.extra_ratio_to_base = extra_ratio_to_base
        self.extra_sampling = extra_sampling
        self.extra_random_seed = int(extra_random_seed)

        self.paths = self._scan_image_paths(
            root_dir,
            extra_image_prefixes=self.extra_image_prefixes,
            extra_max_images=self.extra_max_images,
            extra_ratio_to_base=self.extra_ratio_to_base,
            extra_sampling=self.extra_sampling,
            extra_random_seed=self.extra_random_seed,
        )
        self.cache: Dict[Tuple[str, str, str | None], Tuple[torch.Tensor, torch.Tensor]] = {}

        self.base_transform = build_base_transform(image_size=image_size)
        self.geometry_transform = build_geometric_transform(image_size=image_size)
        self.image_only_transform = build_image_only_transform()

    @staticmethod
    def _normalize_prefixes(prefixes: Sequence[str] | str | None) -> Tuple[str, ...]:
        if prefixes is None:
            return tuple()
        if isinstance(prefixes, str):
            prefixes = [p.strip() for p in prefixes.split(",")]
        return tuple(str(p).strip() for p in prefixes if str(p).strip())

    @staticmethod
    def _is_extra_image(path: Path, prefixes: Sequence[str]) -> bool:
        if not prefixes:
            return False
        name = path.name
        return any(name.startswith(prefix) for prefix in prefixes)

    @staticmethod
    def _select_extra_paths(
        extra_paths: List[Path],
        max_extra: int,
        sampling: str,
        seed: int,
    ) -> List[Path]:
        if max_extra < 0 or len(extra_paths) <= max_extra:
            return extra_paths
        max_extra = max(0, int(max_extra))
        if max_extra == 0:
            return []

        sampling = str(sampling).lower()
        if sampling == "first":
            return extra_paths[:max_extra]
        if sampling == "random":
            rng = np.random.default_rng(int(seed))
            idxs = sorted(rng.choice(len(extra_paths), size=max_extra, replace=False).tolist())
            return [extra_paths[i] for i in idxs]
        if sampling == "even":
            idxs = np.linspace(0, len(extra_paths) - 1, max_extra).round().astype(int).tolist()
            return [extra_paths[i] for i in idxs]
        raise ValueError(f"Unsupported extra_sampling: {sampling}. Use one of: even, first, random.")

    @classmethod
    def _scan_image_paths(
        cls,
        root_dir: str,
        extra_image_prefixes: Sequence[str] | str | None = ("DSC_",),
        extra_max_images: int | None = None,
        extra_ratio_to_base: float | None = None,
        extra_sampling: str = "even",
        extra_random_seed: int = 2026,
    ) -> List[str]:
        root = Path(root_dir)
        prefixes = cls._normalize_prefixes(extra_image_prefixes)

        base_paths: List[Path] = []
        extra_paths: List[Path] = []
        for path in sorted(root.rglob("*")):
            if not (path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
                continue
            if cls._is_extra_image(path, prefixes):
                extra_paths.append(path)
            else:
                base_paths.append(path)

        max_extra: int | None = None
        if extra_ratio_to_base is not None and float(extra_ratio_to_base) >= 0.0:
            max_extra = int(round(len(base_paths) * float(extra_ratio_to_base)))
        if extra_max_images is not None and int(extra_max_images) >= 0:
            max_extra = int(extra_max_images) if max_extra is None else min(max_extra, int(extra_max_images))

        if max_extra is None:
            selected_extra_paths = extra_paths
        else:
            selected_extra_paths = cls._select_extra_paths(
                extra_paths,
                max_extra=max_extra,
                sampling=extra_sampling,
                seed=extra_random_seed,
            )

        image_paths = [str(p) for p in sorted(base_paths + selected_extra_paths)]
        print(
            "[Stage1ContrastiveDataset] "
            f"base={len(base_paths)}, extra_total={len(extra_paths)}, "
            f"extra_used={len(selected_extra_paths)}, total={len(image_paths)}, "
            f"extra_prefixes={list(prefixes)}, extra_ratio_to_base={extra_ratio_to_base}, "
            f"extra_max_images={extra_max_images}, extra_sampling={extra_sampling}"
        )
        return image_paths

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def _build_edge_suppression_mask(image: Image.Image, edge: int = 64, threshold: int = 30) -> np.ndarray:
        """Mask out black borders often introduced by resize / padding artifacts."""
        image_np = np.array(image)
        height, width, _ = image_np.shape
        edge = int(min(height / 10, width / 10, edge))

        valid_mask = np.ones((height, width), dtype=np.uint8)
        dark_pixels = np.all(image_np < threshold, axis=2)

        valid_mask[:edge, :] &= ~dark_pixels[:edge, :]
        valid_mask[-edge:, :] &= ~dark_pixels[-edge:, :]
        valid_mask[:, :edge] &= ~dark_pixels[:, :edge]
        valid_mask[:, -edge:] &= ~dark_pixels[:, -edge:]
        return valid_mask

    @staticmethod
    def _build_all_one_mask(image: Image.Image) -> Image.Image:
        width, height = image.size
        mask_np = np.ones((height, width), dtype=np.uint8) * 255
        return Image.fromarray(mask_np, mode="L")

    def _resolve_mask_path(self, image_path: str) -> str:
        relative_path = os.path.relpath(image_path, self.root_dir)
        relative_path = os.path.splitext(relative_path)[0] + self.mask_suffix
        if self.mask_root_dir is None:
            raise RuntimeError("mask_root_dir is required when mask_mode != 'none'.")
        return os.path.join(self.mask_root_dir, relative_path)

    def _postprocess_external_mask(self, mask_np: np.ndarray) -> np.ndarray:
        mask_np = (mask_np > self.external_mask_threshold).astype(np.uint8) * 255

        if self.external_mask_median_blur and self.external_mask_median_blur >= 3:
            kernel = int(self.external_mask_median_blur)
            if kernel % 2 == 0:
                kernel += 1
            mask_np = cv2.medianBlur(mask_np, kernel)

        if self.external_mask_open_kernel and self.external_mask_open_kernel >= 3:
            kernel = int(self.external_mask_open_kernel)
            if kernel % 2 == 0:
                kernel += 1
            morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
            mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, morph_kernel)

        if self.external_mask_close_kernel and self.external_mask_close_kernel >= 3:
            kernel = int(self.external_mask_close_kernel)
            if kernel % 2 == 0:
                kernel += 1
            morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
            mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, morph_kernel)

        return (mask_np > 127).astype(np.uint8) * 255

    def _load_mask_for_image(self, image_path: str, image: Image.Image) -> Image.Image:
        if self.mask_mode == "none":
            mask = self._build_all_one_mask(image)
        elif self.mask_mode == "sam2":
            mask_path = self._resolve_mask_path(image_path)
            if not os.path.exists(mask_path):
                if self.missing_mask_policy == "ones":
                    mask = self._build_all_one_mask(image)
                else:
                    raise FileNotFoundError(f"Missing external mask: {mask_path}")
                print(f"Missing external mask: {mask_path}")
            else:
                mask = Image.open(mask_path).convert("L")
                if mask.size != image.size:
                    mask = mask.resize(image.size, Image.NEAREST)
                mask_np = self._postprocess_external_mask(np.array(mask, dtype=np.uint8))
                mask = Image.fromarray(mask_np, mode="L")
        else:
            raise ValueError(f"Unsupported mask_mode: {self.mask_mode}")

        # if self.edge_mask:
        #     edge_mask = self._build_edge_suppression_mask(image).astype(np.uint8)
        #     mask_np = np.array(mask, dtype=np.uint8)
        #     mask_np = (mask_np // 255) * edge_mask * 255
        #     mask = Image.fromarray(mask_np, mode="L")

        return mask
    
    @staticmethod
    def _center_pad_to_square(image: Image.Image, fill: int | Tuple[int, int, int] = 0) -> Image.Image:
        """
        将短边居中 padding 到长边，使图像变成正方形。

        例如：
            480 x 640 -> 640 x 640
            800 x 600 -> 800 x 800
            640 x 640 -> 不变
        """
        width, height = image.size
        target_size = max(width, height)

        pad_w = target_size - width
        pad_h = target_size - height

        if pad_w == 0 and pad_h == 0:
            return image

        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        return ImageOps.expand(image, border=(left, top, right, bottom), fill=fill)

    def _load_image_and_mask(self, image_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            mask = self._load_mask_for_image(image_path, image)
            
            # 短边 padding 到长边，先变成正方形
            image = self._center_pad_to_square(image, fill=(0, 0, 0))
            mask = self._center_pad_to_square(mask, fill=0)

            # padding 后再做 edge mask，更合理
            if self.edge_mask:
                edge_mask = self._build_edge_suppression_mask(image).astype(np.uint8)
                mask_np = np.array(mask, dtype=np.uint8)
                mask_np = (mask_np // 255) * edge_mask * 255
                mask = Image.fromarray(mask_np, mode="L")

            rgba_image = image.copy()
            rgba_image.putalpha(mask)
            rgba_tensor = self.base_transform(rgba_image)

            image_tensor = rgba_tensor[:3]
            mask_tensor = rgba_tensor[3:].float()
            mask_tensor = (mask_tensor > 0.5).float()

            if mask_tensor.sum().item() < self.min_valid_pixels:
                mask_tensor = torch.ones_like(mask_tensor)

        return image_tensor, mask_tensor

    def _build_augmented_view(self, image_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply synchronized geometric transforms, then image-only transforms."""
        rgba_tensor = torch.cat([image_tensor, mask_tensor], dim=0)
        rgba_tensor = self.geometry_transform(rgba_tensor)

        view_image = rgba_tensor[:3]
        view_mask = (rgba_tensor[3:] > 0.5).float()
        view_image = self.image_only_transform(view_image)
        return view_image, view_mask

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int | str]:
        image_path = self.paths[index]
        cache_key = (image_path, self.mask_mode, self.mask_root_dir)

        if self.use_cache and cache_key in self.cache:
            image_tensor, mask_tensor = self.cache[cache_key]
        else:
            image_tensor, mask_tensor = self._load_image_and_mask(image_path)
            if self.use_cache:
                self.cache[cache_key] = (image_tensor, mask_tensor)

        view_1, mask_1 = self._build_augmented_view(image_tensor.clone(), mask_tensor.clone())
        view_2, mask_2 = self._build_augmented_view(image_tensor.clone(), mask_tensor.clone())

        return {
            "x1": view_1,
            "x2": view_2,
            "m1": mask_1,
            "m2": mask_2,
            "index": index,
            "path": image_path,
        }


# Backward-compatible alias used by the existing project.
UnlabeledLeafContrastiveDataset = Stage1ContrastiveDataset
