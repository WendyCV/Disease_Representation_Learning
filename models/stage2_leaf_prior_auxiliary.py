from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Type

import torch
import torch.nn as nn
import yaml

from ultralytics.models.yolo.detect import DetectionTrainer  # type: ignore
from ultralytics.utils import LOGGER  # type: ignore
from ultralytics.utils.torch_utils import unwrap_model  # type: ignore

from losses.stage2_leaf_prior_auxiliary_loss import LeafPriorAuxiliaryLoss
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


class LeafnessHead(nn.Module):
    """
    Channel-agnostic leafness head.

    Instead of learning a 1x1 conv that depends on exact input channels,
    we first reduce feature maps to a single spatial response map by channel mean,
    then apply a learnable affine transform:
        logit = scale * mean(abs(x), dim=1) + bias

    This avoids all channel-mismatch issues across different YOLO blocks.
    """

    def __init__(self, init_scale: float = 1.0, init_bias: float = 0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(float(init_bias), dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        # x: [B, C, H, W]
        pooled = x.abs().mean(dim=1, keepdim=True)   # [B,1,H,W]
        logit = self.scale * pooled + self.bias
        prob = torch.sigmoid(logit)
        return logit, prob


class ResidualLeafGate(nn.Module):
    def __init__(self, init_beta: float = 0.2):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))

    def forward(self, feat: torch.Tensor, prob: torch.Tensor):
        beta = self.beta.to(dtype=feat.dtype, device=feat.device)
        return feat * (1.0 + beta * prob.to(dtype=feat.dtype, device=feat.device))


@dataclass
class _LeafAuxRuntime:
    teacher: nn.Module
    teacher_branch: str
    teacher_feature_indices: List[int]
    student_layer_indices: List[int]
    alpha: float
    enable_gate: bool
    aux_loss_fn: nn.Module
    student_logits_cache: Dict[int, torch.Tensor]
    student_prob_cache: Dict[int, torch.Tensor]
    hook_handles: list
    structural_names: List[str]


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


def _build_stage1_teacher(leaf_cfg: dict, project_root: Path, verbose=True):
    ssl_cfg_path = resolve_local_path(leaf_cfg["teacher_ssl_config"], project_root)
    ckpt_path = resolve_local_path(leaf_cfg["teacher_ckpt_path"], project_root)

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
        verbose=False
    )

    ckpt = load_torch_checkpoint(ckpt_path, map_location="cpu")
    state_dict = extract_stage1_state_dict(ckpt)
    missing, unexpected = teacher_wrapper.load_state_dict(state_dict, strict=False)
    if missing:
        LOGGER.warning(f"[LeafAux] Missing keys when loading Stage1 teacher: {len(missing)}")
    if unexpected:
        LOGGER.warning(f"[LeafAux] Unexpected keys when loading Stage1 teacher: {len(unexpected)}")

    teacher = teacher_wrapper.online.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    if verbose:
        LOGGER.info(f"[LeafAux] Teacher loaded from: {ckpt_path}")
        LOGGER.info(f"[LeafAux] Teacher SSL config: {ssl_cfg_path}")
    return teacher


def _infer_module_out_channels(module: nn.Module) -> int:
    convs = [m for m in module.modules() if isinstance(m, nn.Conv2d)]
    if convs:
        return int(convs[-1].out_channels)
    # common ultralytics attrs
    for attr in ("out_channels", "c2", "cv2"):
        if hasattr(module, attr):
            obj = getattr(module, attr)
            if isinstance(obj, int):
                return int(obj)
            if isinstance(obj, nn.Conv2d):
                return int(obj.out_channels)
            convs2 = [m for m in obj.modules() if isinstance(m, nn.Conv2d)] if isinstance(obj, nn.Module) else []
            if convs2:
                return int(convs2[-1].out_channels)
    raise RuntimeError(f"Cannot infer output channels for module {module.__class__.__name__}")


class _LeafAuxHook:
    def __init__(
        self,
        layer_idx: int,
        head: nn.Module,
        gate: nn.Module,
        logits_cache: Dict[int, torch.Tensor],
        prob_cache: Dict[int, torch.Tensor],
        enable_gate: bool = True,
    ):
        self.layer_idx = int(layer_idx)
        self.head = head
        self.gate = gate
        self.logits_cache = logits_cache
        self.prob_cache = prob_cache
        self.enable_gate = bool(enable_gate)

    def __call__(self, _module, _inputs, output):
        feat = _extract_tensor(output)
        logits, prob = self.head(feat)
        self.logits_cache[self.layer_idx] = logits
        self.prob_cache[self.layer_idx] = prob
        if not self.enable_gate:
            return output
        gated = self.gate(feat, prob)
        return gated



def _attach_structural_modules(det_model: nn.Module, student_layer_indices: Sequence[int], gate_init_beta: float):
    names = []
    for idx in student_layer_indices:
        module = det_model.model[idx]
        channels = _infer_module_out_channels(module)
        head_name = f"_leafaux_head_{idx}"
        gate_name = f"_leafaux_gate_{idx}"
        if not hasattr(det_model, head_name):
            setattr(det_model, head_name, LeafnessHead())
        if not hasattr(det_model, gate_name):
            setattr(det_model, gate_name, ResidualLeafGate(init_beta=gate_init_beta))
        names.extend([head_name, gate_name])
    return names


def _register_leafaux_hooks(det_model: nn.Module, student_layer_indices: Sequence[int], enable_gate: bool = True):
    logits_cache: Dict[int, torch.Tensor] = {}
    prob_cache: Dict[int, torch.Tensor] = {}
    handles = []
    for idx in student_layer_indices:
        module = det_model.model[idx]
        head = getattr(det_model, f"_leafaux_head_{idx}")
        gate = getattr(det_model, f"_leafaux_gate_{idx}")
        hook = _LeafAuxHook(idx, head, gate, logits_cache, prob_cache, enable_gate=enable_gate)
        handles.append(module.register_forward_hook(hook))
    return logits_cache, prob_cache, handles


def _get_runtime(det_model):
    return getattr(det_model, "_leafaux_runtime", None)


def loss_with_leafaux(self, batch, preds=None):
    runtime = _get_runtime(self)
    det_loss, loss_items = self._leafaux_original_loss(batch, preds)

    if torch.is_tensor(det_loss):
        det_loss_detached = det_loss.detach()
        det_loss_scalar = float(det_loss_detached.sum().item()) if det_loss_detached.numel() > 1 else float(det_loss_detached.item())
    else:
        det_loss_scalar = float(det_loss)

    self.leafaux_aux_raw = 0.0
    self.leafaux_aux_weighted = 0.0
    self.leafaux_aux_ratio = 0.0
    self.leafaux_used = 0.0

    if runtime is None:
        return det_loss, loss_items

    if (not self.training) or (not isinstance(batch, dict)) or ("img" not in batch):
        runtime.student_logits_cache.clear()
        runtime.student_prob_cache.clear()
        return det_loss, loss_items

    missing_student = [idx for idx in runtime.student_layer_indices if idx not in runtime.student_logits_cache]
    if missing_student:
        runtime.student_logits_cache.clear()
        runtime.student_prob_cache.clear()
        _ = self(batch["img"])
        missing_student = [idx for idx in runtime.student_layer_indices if idx not in runtime.student_logits_cache]

    if missing_student:
        LOGGER.warning(
            f"[LeafAux] Missing student logits for layers {missing_student}; "
            f"skip leaf prior auxiliary loss for this batch. captured={sorted(runtime.student_logits_cache.keys())}"
        )
        runtime.student_logits_cache.clear()
        runtime.student_prob_cache.clear()
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
                f"Teacher branch '{runtime.teacher_branch}' not found. Available keys: {list(teacher_outputs.keys())}"
            )
        teacher_feature_list = teacher_outputs[runtime.teacher_branch]

    teacher_features = [teacher_feature_list[index] for index in runtime.teacher_feature_indices]
    student_logits = [runtime.student_logits_cache[idx] for idx in runtime.student_layer_indices]
    student_probs = [runtime.student_prob_cache[idx] for idx in runtime.student_layer_indices]

    aux_loss, per_layer = runtime.aux_loss_fn(student_logits, student_probs, teacher_features)
    weighted_aux = runtime.alpha * aux_loss
    total_loss = det_loss + weighted_aux

    aux_raw = _safe_float(aux_loss)
    aux_weighted = _safe_float(weighted_aux)
    aux_ratio = aux_weighted / max(det_loss_scalar, 1e-12)

    self.leafaux_aux_raw = aux_raw
    self.leafaux_aux_weighted = aux_weighted
    self.leafaux_aux_ratio = aux_ratio
    self.leafaux_used = 1.0
    self.leafaux_last = aux_raw
    self.leafaux_per_layer = per_layer.detach().cpu().tolist()

    runtime.student_logits_cache.clear()
    runtime.student_prob_cache.clear()
    return total_loss, loss_items


def patch_detection_model_with_leafaux(det_model: nn.Module, leaf_cfg: dict, project_root: Path, verbose=True):
    strip_leafaux_runtime(det_model)

    teacher = _build_stage1_teacher(leaf_cfg, project_root, verbose=verbose)
    teacher_branch = str(leaf_cfg.get("teacher_branch", "pos_feats"))
    teacher_feature_indices = list(leaf_cfg.get("teacher_feature_indices", [0, 1, 2]))
    student_layer_indices = list(leaf_cfg.get("student_layer_indices", [4, 6, 8]))
    alpha = float(leaf_cfg.get("alpha", 0.03))
    layer_weights = list(leaf_cfg.get("layer_weights", [1.0, 1.0, 1.0]))
    gate_init_beta = float(leaf_cfg.get("gate_init_beta", 0.2))
    enable_gate = bool(leaf_cfg.get("enable_gate", True))

    structural_names = _attach_structural_modules(det_model, student_layer_indices, gate_init_beta)
    aux_loss_fn = LeafPriorAuxiliaryLoss(
        layer_weights=layer_weights,
        bg_quantile=float(leaf_cfg.get("bg_quantile", 0.3)),
        gamma=float(leaf_cfg.get("gamma", 1.0)),
        lambda_bg=float(leaf_cfg.get("lambda_bg", 0.25)),
    )

    student_logits_cache, student_prob_cache, hook_handles = _register_leafaux_hooks(
        det_model, student_layer_indices, enable_gate=enable_gate
    )

    runtime = _LeafAuxRuntime(
        teacher=teacher,
        teacher_branch=teacher_branch,
        teacher_feature_indices=teacher_feature_indices,
        student_layer_indices=student_layer_indices,
        alpha=alpha,
        enable_gate=enable_gate,
        aux_loss_fn=aux_loss_fn,
        student_logits_cache=student_logits_cache,
        student_prob_cache=student_prob_cache,
        hook_handles=hook_handles,
        structural_names=structural_names,
    )

    object.__setattr__(det_model, "_leafaux_runtime", runtime)
    object.__setattr__(det_model, "_leafaux_original_loss", det_model.loss)
    object.__setattr__(det_model, "_leafaux_patched", True)
    object.__setattr__(det_model, "loss_with_leafaux", loss_with_leafaux.__get__(det_model, type(det_model)))

    det_model.loss = det_model.loss_with_leafaux

    if verbose:
        LOGGER.info("=" * 80)
        LOGGER.info("[LeafAux] Enabled")
        LOGGER.info(f"[LeafAux] teacher_branch       : {teacher_branch}")
        LOGGER.info(f"[LeafAux] teacher_feature_idx  : {teacher_feature_indices}")
        LOGGER.info(f"[LeafAux] student_layer_idx    : {student_layer_indices}")
        LOGGER.info(f"[LeafAux] layer_weights        : {layer_weights}")
        LOGGER.info(f"[LeafAux] alpha                : {alpha}")
        LOGGER.info(f"[LeafAux] gate_init_beta       : {gate_init_beta}")
        LOGGER.info(f"[LeafAux] enable_gate          : {enable_gate}")
        LOGGER.info("=" * 80)
    return det_model


def strip_leafaux_runtime(det_model):
    """
    Remove all runtime-only LeafAux objects before checkpoint save / final_eval.
    The saved checkpoint must remain a pure Ultralytics DetectionModel.
    """
    runtime = getattr(det_model, "_leafaux_runtime", None)

    # restore original loss first
    if hasattr(det_model, "_leafaux_original_loss"):
        det_model.loss = det_model._leafaux_original_loss

    # remove hook handles
    if runtime is not None:
        handles = getattr(runtime, "hook_handles", None)
        if handles:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

    # delete all runtime-only attrs that should never enter checkpoint
    for name in [
        "_leafaux_runtime",
        "_leafaux_original_loss",
        "_leafaux_patched",
        "loss_with_leafaux",
        "leafaux_aux_raw",
        "leafaux_aux_weighted",
        "leafaux_aux_ratio",
        "leafaux_used",
        "leafaux_last",
        "leafaux_per_layer",
    ]:
        if hasattr(det_model, name):
            try:
                delattr(det_model, name)
            except Exception:
                pass

    return det_model


def _log_leafaux_batch_callback(trainer):
    model = unwrap_model(trainer.model)
    if not getattr(model, "_leafaux_patched", False):
        return
    trainer._leafaux_batch_counter += 1
    interval = int(getattr(trainer, "_leafaux_log_interval", 20))
    if trainer._leafaux_batch_counter > 0 and interval > 0 and (trainer._leafaux_batch_counter % interval == 0):
        LOGGER.info(
            "\n[LeafAuxMonitor] "
            f"raw={getattr(model, 'leafaux_aux_raw', 0.0):.4e}, "
            f"weighted={getattr(model, 'leafaux_aux_weighted', 0.0):.4e}, "
            f"ratio={getattr(model, 'leafaux_aux_ratio', 0.0):.4e}, "
            f"used={getattr(model, 'leafaux_used', 0.0):.0f}"
        )


def _reset_leafaux_batch_counter(trainer):
    trainer._leafaux_batch_counter = 0


def build_leafaux_trainer(
    leafaux_cfg: dict,
    project_root: Path,
    base_trainer_det: Type[DetectionTrainer] = DetectionTrainer,
) -> Type[DetectionTrainer]:
    class LeafAuxTrainer(base_trainer_det):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._leafaux_cfg = dict(leafaux_cfg)
            self._leafaux_project_root = Path(project_root)
            self._leafaux_log_interval = int(leafaux_cfg.get("log_interval", 20))
            self._leafaux_batch_counter = 0

            self.add_callback("on_train_batch_end", _log_leafaux_batch_callback)
            self.add_callback("on_train_epoch_start", _reset_leafaux_batch_counter)

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            patch_detection_model_with_leafaux(model, self._leafaux_cfg, self._leafaux_project_root)
            return model

        def label_loss_items(self, loss_items=None, prefix="train"):
            items = super().label_loss_items(loss_items=loss_items, prefix=prefix)
            model = unwrap_model(self.model)
            if isinstance(items, dict) and prefix == "train":
                items[f"{prefix}/leafaux_aux_raw"] = float(getattr(model, "leafaux_aux_raw", 0.0))
                items[f"{prefix}/leafaux_aux_weighted"] = float(getattr(model, "leafaux_aux_weighted", 0.0))
                items[f"{prefix}/leafaux_aux_ratio"] = float(getattr(model, "leafaux_aux_ratio", 0.0))
                items[f"{prefix}/leafaux_used"] = float(getattr(model, "leafaux_used", 0.0))
            return items

        def save_model(self):
            model = unwrap_model(self.model)
            ema_model = getattr(getattr(self, "ema", None), "ema", None)
            ema_model = unwrap_model(ema_model) if ema_model is not None else None

            was_patched_model = getattr(model, "_leafaux_patched", False)
            was_patched_ema = getattr(ema_model, "_leafaux_patched", False) if ema_model is not None else False

            if was_patched_model:
                strip_leafaux_runtime(model)
            if was_patched_ema:
                strip_leafaux_runtime(ema_model)

            try:
                return super().save_model()
            finally:
                if was_patched_model:
                    patch_detection_model_with_leafaux(model, self._leafaux_cfg, self._leafaux_project_root, verbose=False)
                if was_patched_ema:
                    patch_detection_model_with_leafaux(ema_model, self._leafaux_cfg, self._leafaux_project_root, verbose=False)

        def final_eval(self):
            model = unwrap_model(self.model)
            ema_model = getattr(getattr(self, "ema", None), "ema", None)
            ema_model = unwrap_model(ema_model) if ema_model is not None else None

            was_patched_model = getattr(model, "_leafaux_patched", False)
            was_patched_ema = getattr(ema_model, "_leafaux_patched", False) if ema_model is not None else False

            if was_patched_model:
                strip_leafaux_runtime(model)
            if was_patched_ema:
                strip_leafaux_runtime(ema_model)

            return super().final_eval()

    LeafAuxTrainer.__name__ = f"LeafAuxTrainer_{base_trainer_det.__name__}"
    return LeafAuxTrainer
