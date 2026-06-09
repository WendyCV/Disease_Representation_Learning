from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Type

import torch
import torch.nn as nn
import yaml

from ultralytics.models.yolo.detect import DetectionTrainer  # type: ignore
from ultralytics.utils import LOGGER  # type: ignore
from ultralytics.utils.torch_utils import unwrap_model  # type: ignore

from losses.stage2_foreground_prior_distillation_loss import ForegroundPriorDistillationLoss
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
    def __init__(self, cache: Dict[int, torch.Tensor], layer_idx: int):
        self.cache = cache
        self.layer_idx = int(layer_idx)

    def __call__(self, _module, _inputs, output):
        self.cache[self.layer_idx] = _extract_tensor(output)


class _FPDRuntime:
    def __init__(
        self,
        teacher: nn.Module,
        teacher_branch: str,
        teacher_feature_indices: list[int],
        student_layer_indices: list[int],
        alpha: float,
        fpd_loss_fn: nn.Module,
        student_cache: Dict[int, torch.Tensor],
        hook_handles: list,
    ):
        self.teacher = teacher
        self.teacher_branch = str(teacher_branch)
        self.teacher_feature_indices = [int(x) for x in teacher_feature_indices]
        self.student_layer_indices = [int(x) for x in student_layer_indices]
        self.alpha = float(alpha)
        self.fpd_loss_fn = fpd_loss_fn
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


def _build_stage1_teacher(fpd_cfg: dict, project_root: Path, verbose=True):
    ssl_cfg_path = resolve_local_path(fpd_cfg["teacher_ssl_config"], project_root)
    ckpt_path = resolve_local_path(fpd_cfg["teacher_ckpt_path"], project_root)

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
        LOGGER.warning(f"[FPD] Missing keys when loading Stage1 teacher: {len(missing)}")
    if unexpected:
        LOGGER.warning(f"[FPD] Unexpected keys when loading Stage1 teacher: {len(unexpected)}")

    teacher = teacher_wrapper.online.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    if verbose:
        LOGGER.info(f"[FPD] Teacher loaded from: {ckpt_path}")
        LOGGER.info(f"[FPD] Teacher SSL config: {ssl_cfg_path}")
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
            raise IndexError(f"Cannot access detector layer model.{layer_idx} for FPD") from exc

        hook = _FeatureHook(cache=cache, layer_idx=layer_idx)
        handles.append(module.register_forward_hook(hook))

    return cache, handles


def _get_runtime(det_model):
    return getattr(det_model, "_fpd_runtime", None)


def loss_with_fpd(self, batch, preds=None):
    runtime = _get_runtime(self)
    det_loss, loss_items = self._fpd_original_loss(batch, preds)

    if torch.is_tensor(det_loss):
        det_loss_detached = det_loss.detach()
        det_loss_scalar = float(det_loss_detached.sum().item()) if det_loss_detached.numel() > 1 else float(det_loss_detached.item())
    else:
        det_loss_scalar = float(det_loss)

    self.fpd_aux_raw = 0.0
    self.fpd_aux_weighted = 0.0
    self.fpd_aux_ratio = 0.0
    self.fpd_used = 0.0

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
            f"[FPD] Missing student features for layers {missing_student}; "
            f"skip FPD for this batch. captured={sorted(runtime.student_cache.keys())}"
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
                f"Teacher branch '{runtime.teacher_branch}' not found. "
                f"Available keys: {list(teacher_outputs.keys())}"
            )
        teacher_feature_list = teacher_outputs[runtime.teacher_branch]

    teacher_features = [teacher_feature_list[index] for index in runtime.teacher_feature_indices]
    student_features = [runtime.student_cache[idx] for idx in runtime.student_layer_indices]

    fpd_loss, per_layer = runtime.fpd_loss_fn(student_features, teacher_features)
    weighted_fpd = runtime.alpha * fpd_loss
    total_loss = det_loss + weighted_fpd

    aux_raw = _safe_float(fpd_loss)
    aux_weighted = _safe_float(weighted_fpd)
    aux_ratio = aux_weighted / max(det_loss_scalar, 1e-12)

    self.fpd_aux_raw = aux_raw
    self.fpd_aux_weighted = aux_weighted
    self.fpd_aux_ratio = aux_ratio
    self.fpd_used = 1.0

    self.fpd_last = aux_raw
    self.fpd_per_layer = per_layer.detach().cpu().tolist()

    runtime.student_cache.clear()
    return total_loss, loss_items


def patch_detection_model_with_fpd(det_model, fpd_cfg: dict, project_root: Path, verbose=True):
    strip_fpd_runtime(det_model)

    teacher = _build_stage1_teacher(fpd_cfg, project_root, verbose=verbose)
    teacher_branch = str(fpd_cfg.get("teacher_branch", "pos_feats"))
    teacher_feature_indices = list(fpd_cfg.get("teacher_feature_indices", [0, 1, 2]))
    student_layer_indices = list(fpd_cfg.get("student_layer_indices", [4, 6, 8]))
    alpha = float(fpd_cfg.get("alpha", 0.05))
    layer_weights = list(fpd_cfg.get("layer_weights", [1.0, 1.0, 1.0]))

    fpd_loss_fn = ForegroundPriorDistillationLoss(
        layer_weights=layer_weights,
        fg_quantile=float(fpd_cfg.get("fg_quantile", 0.7)),
        bg_quantile=float(fpd_cfg.get("bg_quantile", 0.3)),
        lambda_fg=float(fpd_cfg.get("lambda_fg", 1.0)),
        lambda_bg=float(fpd_cfg.get("lambda_bg", 0.5)),
    )

    student_cache, hook_handles = _register_student_hooks(det_model, student_layer_indices)

    runtime = _FPDRuntime(
        teacher=teacher,
        teacher_branch=teacher_branch,
        teacher_feature_indices=teacher_feature_indices,
        student_layer_indices=student_layer_indices,
        alpha=alpha,
        fpd_loss_fn=fpd_loss_fn,
        student_cache=student_cache,
        hook_handles=hook_handles,
    )

    object.__setattr__(det_model, "_fpd_runtime", runtime)
    object.__setattr__(det_model, "_fpd_original_loss", det_model.loss)
    object.__setattr__(det_model, "_fpd_patched", True)
    object.__setattr__(det_model, "loss_with_fpd", loss_with_fpd.__get__(det_model, type(det_model)))
    det_model.loss = det_model.loss_with_fpd

    if verbose:
        LOGGER.info("=" * 80)
        LOGGER.info("[FPD] Enabled")
        LOGGER.info(f"[FPD] teacher_branch       : {teacher_branch}")
        LOGGER.info(f"[FPD] teacher_feature_idx  : {teacher_feature_indices}")
        LOGGER.info(f"[FPD] student_layer_idx    : {student_layer_indices}")
        LOGGER.info(f"[FPD] layer_weights        : {layer_weights}")
        LOGGER.info(f"[FPD] alpha                : {alpha}")
        LOGGER.info("=" * 80)
    return det_model


def strip_fpd_runtime(det_model):
    runtime = getattr(det_model, "_fpd_runtime", None)

    if hasattr(det_model, "_fpd_original_loss"):
        det_model.loss = det_model._fpd_original_loss

    if runtime is not None:
        handles = getattr(runtime, "hook_handles", None)
        if handles:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

    for name in [
        "_fpd_runtime",
        "_fpd_original_loss",
        "_fpd_patched",
        "loss_with_fpd",
        "fpd_last",
        "fpd_per_layer",
        "fpd_aux_raw",
        "fpd_aux_weighted",
        "fpd_aux_ratio",
        "fpd_used",
    ]:
        if hasattr(det_model, name):
            try:
                delattr(det_model, name)
            except Exception:
                pass

    return det_model


def _patch_fpd_on_model(model, cfg, project_root, verbose=False):
    if model is None:
        return
    model = unwrap_model(model)
    if not getattr(model, "_fpd_patched", False):
        patch_detection_model_with_fpd(model, cfg, project_root, verbose=verbose)


def _strip_fpd_from_model(model):
    if model is None:
        return
    model = unwrap_model(model)
    if getattr(model, "_fpd_patched", False):
        strip_fpd_runtime(model)


def _attach_fpd_callback(trainer):
    _patch_fpd_on_model(trainer.model, trainer._fpd_cfg, trainer._fpd_project_root, verbose=True)
    # EMA usually stays plain for FPD, but this keeps behavior robust across trainer/version changes.
    ema_model = getattr(getattr(trainer, "ema", None), "ema", None)
    if getattr(ema_model, "_fpd_patched", False):
        _patch_fpd_on_model(ema_model, trainer._fpd_cfg, trainer._fpd_project_root, verbose=False)


def _log_fpd_batch_callback(trainer):
    model = unwrap_model(trainer.model)
    if not getattr(model, "_fpd_patched", False):
        return

    trainer._fpd_batch_counter += 1
    interval = int(getattr(trainer, "_fpd_log_interval", 20))
    if trainer._fpd_batch_counter > 0 and interval > 0 and (trainer._fpd_batch_counter % interval == 0):
        LOGGER.info(
            "\n[FPDMonitor] "
            f"raw={getattr(model, 'fpd_aux_raw', 0.0):.4e}, "
            f"weighted={getattr(model, 'fpd_aux_weighted', 0.0):.4e}, "
            f"ratio={getattr(model, 'fpd_aux_ratio', 0.0):.4e}, "
            f"used={getattr(model, 'fpd_used', 0.0):.0f}"
        )


def _reset_fpd_batch_counter(trainer):
    trainer._fpd_batch_counter = 0


def build_fpd_trainer(
    fpd_cfg: dict,
    project_root: Path,
    base_trainer_cls: Type[DetectionTrainer] = DetectionTrainer,
) -> Type[DetectionTrainer]:
    class FPDTrainer(base_trainer_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._fpd_cfg = dict(fpd_cfg)
            self._fpd_project_root = Path(project_root)
            self._fpd_log_interval = int(fpd_cfg.get("log_interval", 20))
            self._fpd_batch_counter = 0

            self.add_callback("on_train_start", _attach_fpd_callback)
            self.add_callback("on_train_batch_end", _log_fpd_batch_callback)
            self.add_callback("on_train_epoch_start", _reset_fpd_batch_counter)

        def label_loss_items(self, loss_items=None, prefix="train"):
            items = super().label_loss_items(loss_items=loss_items, prefix=prefix)
            model = unwrap_model(self.model)

            if isinstance(items, dict) and prefix == "train":
                items[f"{prefix}/fpd_aux_raw"] = float(getattr(model, "fpd_aux_raw", 0.0))
                items[f"{prefix}/fpd_aux_weighted"] = float(getattr(model, "fpd_aux_weighted", 0.0))
                items[f"{prefix}/fpd_aux_ratio"] = float(getattr(model, "fpd_aux_ratio", 0.0))
                items[f"{prefix}/fpd_used"] = float(getattr(model, "fpd_used", 0.0))
            return items

        def save_model(self):
            model = unwrap_model(self.model)
            ema_model = getattr(getattr(self, "ema", None), "ema", None)
            ema_model = unwrap_model(ema_model) if ema_model is not None else None

            was_patched_model = getattr(model, "_fpd_patched", False)
            was_patched_ema = getattr(ema_model, "_fpd_patched", False) if ema_model is not None else False

            if was_patched_model:
                strip_fpd_runtime(model)
            if was_patched_ema:
                strip_fpd_runtime(ema_model)

            try:
                return super().save_model()
            finally:
                if was_patched_model:
                    patch_detection_model_with_fpd(model, self._fpd_cfg, self._fpd_project_root, verbose=False)
                if was_patched_ema:
                    patch_detection_model_with_fpd(ema_model, self._fpd_cfg, self._fpd_project_root, verbose=False)

        def final_eval(self):
            model = unwrap_model(self.model)
            ema_model = getattr(getattr(self, "ema", None), "ema", None)
            ema_model = unwrap_model(ema_model) if ema_model is not None else None

            was_patched_model = getattr(model, "_fpd_patched", False)
            was_patched_ema = getattr(ema_model, "_fpd_patched", False) if ema_model is not None else False

            if was_patched_model:
                strip_fpd_runtime(model)
            if was_patched_ema:
                strip_fpd_runtime(ema_model)

            return super().final_eval()

    FPDTrainer.__name__ = f"FPDTrainer_{base_trainer_cls.__name__}"
    return FPDTrainer