from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Type

import torch
import torch.nn as nn
import yaml

from ultralytics.models.yolo.detect import DetectionTrainer  # type: ignore
from ultralytics.utils import LOGGER  # type: ignore
from ultralytics.utils.torch_utils import unwrap_model  # type: ignore

from losses.stage2_spatial_alignment_loss import SpatialMapAlignmentLoss
from models.stage1_ssl_model import Stage1SslModel
from utils.checkpoint import extract_stage1_state_dict, load_torch_checkpoint


def _safe_float(x) -> float:
    if x is None:
        return 0.0
    if torch.is_tensor(x):
        x = x.detach()
        if x.numel() == 1:
            return float(x.item())
        return float(x.mean().item())
    return float(x)


class _FeatureHook:
    """Top-level hook object so checkpoints remain pickle-able."""

    def __init__(self, cache: Dict[int, torch.Tensor], layer_idx: int):
        self.cache = cache
        self.layer_idx = int(layer_idx)

    def __call__(self, _module, _inputs, output):
        self.cache[self.layer_idx] = _extract_tensor(output)


class _SpatialAlignmentRuntime:
    """Runtime-only objects. Must be stripped before save/final_eval."""

    def __init__(
        self,
        teacher: nn.Module,
        teacher_branch: str,
        teacher_feature_indices: list[int],
        student_layer_indices: list[int],
        alpha: float,
        alignment_loss_fn: nn.Module,
        student_cache: Dict[int, torch.Tensor],
        hook_handles: list,
    ):
        self.teacher = teacher
        self.teacher_branch = str(teacher_branch)
        self.teacher_feature_indices = [int(x) for x in teacher_feature_indices]
        self.student_layer_indices = [int(x) for x in student_layer_indices]
        self.alpha = float(alpha)
        self.alignment_loss_fn = alignment_loss_fn
        self.student_cache = student_cache
        self.hook_handles = hook_handles


def resolve_local_path(path_str: str, project_root: Path) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p.resolve())
    return str((project_root / p).resolve())


def _extract_tensor(output):
    if isinstance(output, (list, tuple)):
        for item in reversed(output):
            if torch.is_tensor(item):
                return item
        raise TypeError("No tensor found in layer output list/tuple.")
    if not torch.is_tensor(output):
        raise TypeError(f"Unsupported hooked output type: {type(output)}")
    return output


def _build_stage1_teacher(spatial_cfg: dict, project_root: Path):
    ssl_cfg_path = resolve_local_path(spatial_cfg["teacher_ssl_config"], project_root)
    ckpt_path = resolve_local_path(spatial_cfg["teacher_ckpt_path"], project_root)

    with open(ssl_cfg_path, "r", encoding="utf-8") as f:
        ssl_cfg = yaml.safe_load(f)

    model_cfg = ssl_cfg.get("model", {})
    data_cfg = ssl_cfg.get("data", {})
    ablation_cfg = ssl_cfg.get("ablation", {})
    snapshot_cfg = model_cfg.get("snapshot_teacher", {})

    teacher_wrapper = Stage1SslModel(
        yolo_model=model_cfg.get("yolo_model", "yolov8n.pt"),
        nc=model_cfg.get("nc", None),
        layer_indices=tuple(model_cfg.get("layer_indices", [4, 6, 8])),
        image_size=data_cfg.get("image_size", 640),
        proj_dim=model_cfg.get("proj_dim", 256),
        local_dim=model_cfg.get("local_dim", 128),
        queue_size=model_cfg.get("queue_size", 4096),
        momentum=model_cfg.get("momentum", 0.999),
        sppf_indice=model_cfg.get("sppf_indice", 9),
        use_pos=ablation_cfg.get("use_pos", True),
        pos_pe_channels=model_cfg.get("pos_pe_channels", 64),
        pos_init_scales=model_cfg.get("pos_init_scales", [0.1, 0.5, 1.0]),
        pos_enable_fg_guidance=model_cfg.get("pos_enable_fg_guidance", True),
        pos_fg_gate_init=model_cfg.get("pos_fg_gate_init", 1.0),
        enable_raw_projection=model_cfg.get("enable_raw_projection", False),
        separate_projector=model_cfg.get("separate_projector", False),
        use_snapshot_teacher=snapshot_cfg.get("enabled", False),
        verbose=False,
    )

    ckpt = load_torch_checkpoint(ckpt_path, map_location="cpu")
    state_dict = extract_stage1_state_dict(ckpt)
    missing, unexpected = teacher_wrapper.load_state_dict(state_dict, strict=False)
    if missing:
        LOGGER.warning(f"[SpatialAlign] Missing keys when loading Stage1 teacher: {len(missing)}")
    if unexpected:
        LOGGER.warning(f"[SpatialAlign] Unexpected keys when loading Stage1 teacher: {len(unexpected)}")

    teacher = teacher_wrapper.online.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    LOGGER.info(f"[SpatialAlign] Teacher loaded from: {ckpt_path}")
    LOGGER.info(f"[SpatialAlign] Teacher SSL config: {ssl_cfg_path}")
    return teacher


def _register_student_hooks(det_model, student_layer_indices: Sequence[int]):
    cache: Dict[int, torch.Tensor] = {}
    handles = []

    if not hasattr(det_model, "model"):
        raise AttributeError("Detector model has no attribute 'model'; cannot register student hooks.")

    for layer_idx in student_layer_indices:
        try:
            module = det_model.model[layer_idx]
        except Exception as exc:
            raise IndexError(f"Cannot access detector layer model.{layer_idx} for spatial alignment") from exc

        hook = _FeatureHook(cache=cache, layer_idx=layer_idx)
        handles.append(module.register_forward_hook(hook))

    return cache, handles


def _get_runtime(det_model):
    return getattr(det_model, "_sa_runtime", None)


def loss_with_spatial_alignment(self, batch, preds=None):
    runtime = _get_runtime(self)
    det_loss, loss_items = self._sa_original_loss(batch, preds)

    if torch.is_tensor(det_loss):
        det_loss_detached = det_loss.detach()
        det_loss_scalar = float(det_loss_detached.sum().item()) if det_loss_detached.numel() > 1 else float(det_loss_detached.item())
    else:
        det_loss_scalar = float(det_loss)

    self.sa_aux_raw = 0.0
    self.sa_aux_weighted = 0.0
    self.sa_aux_ratio = 0.0
    self.sa_used = 0.0

    if runtime is None:
        return det_loss, loss_items

    if (not self.training) or (not isinstance(batch, dict)) or ("img" not in batch):
        runtime.student_cache.clear()
        return det_loss, loss_items

    missing_student = [idx for idx in runtime.student_layer_indices if idx not in runtime.student_cache]

    if missing_student:
        runtime.student_cache.clear()
        _ = self(batch["img"])
        missing_student = [idx for idx in runtime.student_layer_indices if idx not in runtime.student_cache]

    if missing_student:
        LOGGER.warning(
            f"[SpatialAlign] Missing student features for layers {missing_student}; "
            f"skip alignment for this batch. captured={sorted(runtime.student_cache.keys())}"
        )
        runtime.student_cache.clear()
        return det_loss, loss_items

    images = batch["img"]
    teacher_device = images.device
    teacher_dtype = next(runtime.teacher.parameters()).dtype
    runtime.teacher.to(device=teacher_device)
    teacher_images = images.to(device=teacher_device, dtype=teacher_dtype, non_blocking=True)

    with torch.no_grad():
        teacher_outputs = runtime.teacher(teacher_images, fg_mask=None)
        if runtime.teacher_branch not in teacher_outputs:
            raise KeyError(
                f"Teacher branch '{runtime.teacher_branch}' not found in teacher outputs. "
                f"Available keys: {list(teacher_outputs.keys())}"
            )
        teacher_feature_list = teacher_outputs[runtime.teacher_branch]

    teacher_features = [teacher_feature_list[index] for index in runtime.teacher_feature_indices]
    student_features = [runtime.student_cache[idx] for idx in runtime.student_layer_indices]

    align_loss, per_layer = runtime.alignment_loss_fn(student_features, teacher_features)
    weighted_align = runtime.alpha * align_loss
    total_loss = det_loss + weighted_align

    aux_raw = _safe_float(align_loss)
    aux_weighted = _safe_float(weighted_align)
    aux_ratio = aux_weighted / max(det_loss_scalar, 1e-12)

    self.sa_aux_raw = aux_raw
    self.sa_aux_weighted = aux_weighted
    self.sa_aux_ratio = aux_ratio
    self.sa_used = 1.0

    self.spatial_alignment_last = aux_raw
    self.spatial_alignment_per_layer = per_layer.detach().cpu().tolist()

    runtime.student_cache.clear()
    return total_loss, loss_items


def patch_detection_model_with_spatial_alignment(det_model, spatial_cfg: dict, project_root: Path):
    strip_spatial_alignment_runtime(det_model)

    teacher = _build_stage1_teacher(spatial_cfg, project_root)
    teacher_branch = str(spatial_cfg.get("teacher_branch", "pos_feats"))
    teacher_feature_indices = list(spatial_cfg.get("teacher_feature_indices", [0, 1, 2]))
    student_layer_indices = list(spatial_cfg.get("student_layer_indices", [4, 6, 8]))
    alpha = float(spatial_cfg.get("alpha", 0.05))
    layer_weights = list(spatial_cfg.get("layer_weights", [1.0, 1.0, 1.0]))

    if len(teacher_feature_indices) != len(student_layer_indices):
        raise ValueError(
            "spatial_alignment.teacher_feature_indices and student_layer_indices must have the same length. "
            f"Got {teacher_feature_indices} vs {student_layer_indices}"
        )

    alignment_loss_fn = SpatialMapAlignmentLoss(layer_weights=layer_weights)
    student_cache, hook_handles = _register_student_hooks(det_model, student_layer_indices)

    runtime = _SpatialAlignmentRuntime(
        teacher=teacher,
        teacher_branch=teacher_branch,
        teacher_feature_indices=teacher_feature_indices,
        student_layer_indices=student_layer_indices,
        alpha=alpha,
        alignment_loss_fn=alignment_loss_fn,
        student_cache=student_cache,
        hook_handles=hook_handles,
    )

    object.__setattr__(det_model, "_sa_runtime", runtime)
    object.__setattr__(det_model, "_sa_original_loss", det_model.loss)
    object.__setattr__(det_model, "_spatial_alignment_patched", True)
    object.__setattr__(
        det_model,
        "loss_with_spatial_alignment",
        loss_with_spatial_alignment.__get__(det_model, type(det_model)),
    )
    det_model.loss = det_model.loss_with_spatial_alignment

    LOGGER.info("=" * 80)
    LOGGER.info("[SpatialAlign] Enabled")
    LOGGER.info(f"[SpatialAlign] model_id             : {id(det_model)}")
    LOGGER.info(f"[SpatialAlign] runtime_id           : {id(runtime)}")
    LOGGER.info(f"[SpatialAlign] cache_id             : {id(student_cache)}")
    LOGGER.info(f"[SpatialAlign] teacher_branch       : {teacher_branch}")
    LOGGER.info(f"[SpatialAlign] teacher_feature_idx  : {teacher_feature_indices}")
    LOGGER.info(f"[SpatialAlign] student_layer_idx    : {student_layer_indices}")
    LOGGER.info(f"[SpatialAlign] layer_weights        : {layer_weights}")
    LOGGER.info(f"[SpatialAlign] alpha                : {alpha}")
    LOGGER.info("=" * 80)
    return det_model


def strip_spatial_alignment_runtime(det_model):
    runtime = getattr(det_model, "_sa_runtime", None)

    if hasattr(det_model, "_sa_original_loss"):
        det_model.loss = det_model._sa_original_loss

    if runtime is not None:
        handles = getattr(runtime, "hook_handles", None)
        if handles:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

    for name in [
        "_sa_runtime",
        "_sa_original_loss",
        "_spatial_alignment_patched",
        "loss_with_spatial_alignment",
        "spatial_alignment_last",
        "spatial_alignment_per_layer",
        "sa_aux_raw",
        "sa_aux_weighted",
        "sa_aux_ratio",
        "sa_used",
    ]:
        if hasattr(det_model, name):
            try:
                delattr(det_model, name)
            except Exception:
                pass

    return det_model


def _patch_sa_on_model(model, cfg, project_root):
    if model is None:
        return
    model = unwrap_model(model)
    if not getattr(model, "_spatial_alignment_patched", False):
        patch_detection_model_with_spatial_alignment(model, cfg, project_root)


def _strip_sa_from_model(model):
    if model is None:
        return
    model = unwrap_model(model)
    if getattr(model, "_spatial_alignment_patched", False):
        strip_spatial_alignment_runtime(model)


def _attach_sa_callback(trainer):
    _patch_sa_on_model(trainer.model, trainer._sa_cfg, trainer._sa_project_root)
    ema_model = getattr(getattr(trainer, "ema", None), "ema", None)
    if getattr(ema_model, "_spatial_alignment_patched", False):
        _patch_sa_on_model(ema_model, trainer._sa_cfg, trainer._sa_project_root)


def _log_sa_batch_callback(trainer):
    model = unwrap_model(trainer.model)
    if not getattr(model, "_spatial_alignment_patched", False):
        return

    trainer._sa_batch_counter += 1
    interval = int(getattr(trainer, "_sa_log_interval", 20))
    if trainer._sa_batch_counter > 0 and interval > 0 and (trainer._sa_batch_counter % interval == 0):
        LOGGER.info(
            "\n[SpatialAlignMonitor] "
            f"raw={getattr(model, 'sa_aux_raw', 0.0):.4e}, "
            f"weighted={getattr(model, 'sa_aux_weighted', 0.0):.4e}, "
            f"ratio={getattr(model, 'sa_aux_ratio', 0.0):.4e}, "
            f"used={getattr(model, 'sa_used', 0.0):.0f}"
        )


def _reset_sa_batch_counter(trainer):
    trainer._sa_batch_counter = 0


def build_spatial_alignment_trainer(
    spatial_cfg: dict,
    project_root: Path,
    base_trainer_det: Type[DetectionTrainer] = DetectionTrainer,
) -> Type[DetectionTrainer]:
    class SpatialAlignmentTrainer(base_trainer_det):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._sa_cfg = dict(spatial_cfg)
            self._sa_project_root = Path(project_root)
            self._sa_log_interval = int(spatial_cfg.get("log_interval", 20))
            self._sa_batch_counter = 0

            self.add_callback("on_train_start", _attach_sa_callback)
            self.add_callback("on_train_batch_end", _log_sa_batch_callback)
            self.add_callback("on_train_epoch_start", _reset_sa_batch_counter)

        def label_loss_items(self, loss_items=None, prefix="train"):
            items = super().label_loss_items(loss_items=loss_items, prefix=prefix)
            model = unwrap_model(self.model)

            if isinstance(items, dict) and prefix == "train":
                items[f"{prefix}/sa_aux_raw"] = float(getattr(model, "sa_aux_raw", 0.0))
                items[f"{prefix}/sa_aux_weighted"] = float(getattr(model, "sa_aux_weighted", 0.0))
                items[f"{prefix}/sa_aux_ratio"] = float(getattr(model, "sa_aux_ratio", 0.0))
                items[f"{prefix}/sa_used"] = float(getattr(model, "sa_used", 0.0))
            return items

        def save_model(self):
            model = unwrap_model(self.model)
            ema_model = getattr(getattr(self, "ema", None), "ema", None)
            ema_model = unwrap_model(ema_model) if ema_model is not None else None

            was_patched_model = getattr(model, "_spatial_alignment_patched", False)
            was_patched_ema = getattr(ema_model, "_spatial_alignment_patched", False) if ema_model is not None else False

            if was_patched_model:
                strip_spatial_alignment_runtime(model)
            if was_patched_ema:
                strip_spatial_alignment_runtime(ema_model)

            try:
                return super().save_model()
            finally:
                if was_patched_model:
                    patch_detection_model_with_spatial_alignment(model, self._sa_cfg, self._sa_project_root)
                if was_patched_ema:
                    patch_detection_model_with_spatial_alignment(ema_model, self._sa_cfg, self._sa_project_root)

        def final_eval(self):
            model = unwrap_model(self.model)
            ema_model = getattr(getattr(self, "ema", None), "ema", None)
            ema_model = unwrap_model(ema_model) if ema_model is not None else None

            was_patched_model = getattr(model, "_spatial_alignment_patched", False)
            was_patched_ema = getattr(ema_model, "_spatial_alignment_patched", False) if ema_model is not None else False

            if was_patched_model:
                strip_spatial_alignment_runtime(model)
            if was_patched_ema:
                strip_spatial_alignment_runtime(ema_model)

            return super().final_eval()

    SpatialAlignmentTrainer.__name__ = f"SpatialAlignmentTrainer_{base_trainer_det.__name__}"
    return SpatialAlignmentTrainer