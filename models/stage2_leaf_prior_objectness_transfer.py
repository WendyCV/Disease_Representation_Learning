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

from losses.stage2_leaf_prior_objectness_transfer_loss import build_lpot_loss_from_cfg
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
    def __init__(self, init_scale: float = 1.0, init_bias: float = 0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(float(init_bias), dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        # LPOT heads are lightweight scalar affine heads over channel-mean
        # activation maps. Under AMP or after a strong gate update, rare Inf/NaN
        # values in hooked features can propagate into the auxiliary loss and
        # trigger YOLO's NaN recovery. Keep this branch numerically bounded while
        # preserving the relative spatial response pattern.
        pooled = x.detach() if not torch.is_floating_point(x) else x
        pooled = pooled.abs().mean(dim=1, keepdim=True)
        pooled = torch.nan_to_num(pooled.float(), nan=0.0, posinf=1e4, neginf=0.0).clamp_(0.0, 1e4)

        scale = self.scale.to(device=pooled.device, dtype=pooled.dtype).clamp(-10.0, 10.0)
        bias = self.bias.to(device=pooled.device, dtype=pooled.dtype).clamp(-10.0, 10.0)
        logit = torch.nan_to_num(scale * pooled + bias, nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30.0, 30.0)
        prob = torch.sigmoid(logit)
        return logit, prob


class ResidualLeafGate(nn.Module):
    """
    Lightweight residual gate used by LPOT.

    For LPOT-v4 light PSP we must be able to keep the support branch as a
    loss-only signal. In that case the head still produces logits/probabilities
    for auxiliary losses and monitoring, but this gate returns the original
    feature unchanged.
    """

    def __init__(self, init_beta: float = 0.2, gain: float = 1.0, enabled: bool = True):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))
        self.register_buffer("gain", torch.tensor(float(gain), dtype=torch.float32), persistent=False)
        self.enabled = bool(enabled)

    def forward(self, feat: torch.Tensor, prob: torch.Tensor):
        if not self.enabled:
            return feat
        beta = self.beta.to(dtype=feat.dtype, device=feat.device).clamp(-2.0, 2.0)
        gain = self.gain.to(dtype=feat.dtype, device=feat.device).clamp(0.0, 2.0)
        if float(gain.detach().cpu()) == 0.0:
            return feat
        prob = torch.nan_to_num(
            prob.to(dtype=feat.dtype, device=feat.device),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        gate = torch.nan_to_num(1.0 + gain * beta * prob, nan=1.0, posinf=3.0, neginf=0.1).clamp(0.1, 3.0)
        return feat * gate


def _collect_lpot_runtime_state(det_model: nn.Module) -> dict:
    runtime = getattr(det_model, "_lpot_runtime", None)
    structural_names = list(getattr(runtime, "structural_names", []) or [])

    state = {}
    module_types = {}

    for name in structural_names:
        mod = getattr(det_model, name, None)
        if isinstance(mod, nn.Module):
            state[name] = {
                k: v.detach().cpu()
                for k, v in mod.state_dict().items()
            }
            module_types[name] = mod.__class__.__name__

    return {
        "structural_names": structural_names,
        "state": state,
        "module_types": module_types,
        "prior_layer_indices": list(getattr(runtime, "prior_layer_indices", []) or []),
        "proxy_layer_indices": list(getattr(runtime, "proxy_layer_indices", []) or []),
    }


def _save_lpot_runtime_sidecar(
    save_dir: Path,
    lpot_cfg: dict,
    model: nn.Module,
    ema_model: nn.Module | None = None,
) -> None:
    weights_dir = Path(save_dir) / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "format": "lpot_runtime_sidecar_v1",
        "lpot_cfg": dict(lpot_cfg or {}),
        "model": _collect_lpot_runtime_state(model) if model is not None else {},
        "ema": _collect_lpot_runtime_state(ema_model) if ema_model is not None else {},
    }

    torch.save(payload, weights_dir / "lpot_runtime_state.pt")


def _clear_lpot_caches(runtime) -> None:
    """Clear LPOT runtime caches safely after skipped/failed auxiliary steps."""
    if runtime is None:
        return
    for name in [
        "prior_logits_cache",
        "prior_prob_cache",
        "proxy_logits_cache",
        "proxy_prob_cache",
        "detect_pred_cache",
    ]:
        cache = getattr(runtime, name, None)
        if hasattr(cache, "clear"):
            cache.clear()


def _strip_lpot_runtime_only(det_model: nn.Module):
    """Remove non-serializable LPOT runtime objects but keep LPOT structural modules.

    The previous save path used strip_lpot_runtime(), which also removed
    _lpot_prior_head_*, _lpot_prior_gate_*, _lpot_proxy_head_* and
    _lpot_proxy_gate_* before writing last.pt. That made Ultralytics NaN recovery
    fail because the current training model still had these modules while the EMA
    state_dict loaded from last.pt did not.

    This function is intentionally different:
      - remove hooks, teacher, patched loss and monitor attributes;
      - keep structural head/gate modules registered in state_dict.
    Therefore last.pt remains compatible with the current LPOT model while the
    checkpoint does not pickle the Stage1 teacher/runtime caches.
    """
    runtime = getattr(det_model, "_lpot_runtime", None)

    if hasattr(det_model, "_lpot_original_loss"):
        det_model.loss = det_model._lpot_original_loss

    if runtime is not None:
        handles = getattr(runtime, "hook_handles", None)
        if handles:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass
        _clear_lpot_caches(runtime)

    for name in [
        "_lpot_runtime",
        "_lpot_original_loss",
        "_lpot_patched",
        "loss_with_lpot",
        "lpot_aux_raw",
        "lpot_aux_weighted",
        "lpot_aux_ratio",
        "lpot_used",
        "lpot_prior_raw",
        "lpot_proxy_raw",
        "lpot_bridge_raw",
        "lpot_prior_rank",
        "lpot_proxy_rank",
        "lpot_support_target_mean",
        "lpot_teacher_anchor_raw",
        "lpot_teacher_anchor_rank",
        "lpot_teacher_anchor_bg",
        "lpot_score_raw",
        "lpot_score_rank",
        "lpot_score_bg",
        "lpot_score_pos_mean",
        "lpot_score_neg_mean",
        "lpot_score_valid_ratio",
        "lpot_score_num_maps",
    ]:
        if hasattr(det_model, name):
            try:
                delattr(det_model, name)
            except Exception:
                pass
    return det_model


@dataclass
class _LPOTRuntime:
    teacher: nn.Module
    teacher_branch: str
    teacher_feature_indices: List[int]
    prior_layer_indices: List[int]
    proxy_layer_indices: List[int]
    alpha: float
    route_variant: str
    enable_feature_gate: bool
    enable_proxy_gate: bool
    lpot_loss_fn: nn.Module
    prior_logits_cache: Dict[int, torch.Tensor]
    prior_prob_cache: Dict[int, torch.Tensor]
    proxy_logits_cache: Dict[int, torch.Tensor]
    proxy_prob_cache: Dict[int, torch.Tensor]
    detect_pred_cache: dict
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


def _build_stage1_teacher(lpot_cfg: dict, project_root: Path, verbose=True):
    ssl_cfg_path = resolve_local_path(lpot_cfg["teacher_ssl_config"], project_root)
    ckpt_path = resolve_local_path(lpot_cfg["teacher_ckpt_path"], project_root)

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
        LOGGER.warning(f"[LPOT] Missing keys when loading Stage1 teacher: {len(missing)}")
    if unexpected:
        LOGGER.warning(f"[LPOT] Unexpected keys when loading Stage1 teacher: {len(unexpected)}")

    teacher = teacher_wrapper.online.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    if verbose:
        LOGGER.info(f"[LPOT] Teacher loaded from: {ckpt_path}")
        LOGGER.info(f"[LPOT] Teacher SSL config: {ssl_cfg_path}")
    return teacher


class _LPOTHook:
    def __init__(self, layer_idx: int, head: nn.Module, gate: nn.Module, logits_cache: Dict[int, torch.Tensor], prob_cache: Dict[int, torch.Tensor]):
        self.layer_idx = int(layer_idx)
        self.head = head
        self.gate = gate
        self.logits_cache = logits_cache
        self.prob_cache = prob_cache

    def __call__(self, _module, _inputs, output):
        feat = _extract_tensor(output)
        logits, prob = self.head(feat)
        self.logits_cache[self.layer_idx] = logits
        self.prob_cache[self.layer_idx] = prob
        gated = self.gate(feat, prob)
        return gated


class _DetectPredHook:
    """Cache raw YOLO Detect outputs produced during the normal training forward.

    LPOT-v4.1 needs the actual Detect head outputs to compute score-level
    proposal-support regularization. In Ultralytics, the original loss often
    performs the forward pass internally, so the optional `preds` argument of
    model.loss() may be None. This hook captures the Detect output from that
    normal forward pass without changing the detector.
    """

    def __init__(self, cache: dict):
        self.cache = cache

    def __call__(self, _module, _inputs, output):
        self.cache["preds"] = output
        return None


def _register_detect_output_hook(det_model: nn.Module):
    cache: dict = {}
    try:
        handle = det_model.model[-1].register_forward_hook(_DetectPredHook(cache))
        return cache, handle
    except Exception as e:
        LOGGER.warning(f"[LPOT-v4.1] Failed to register Detect output hook: {e}")
        return cache, None


def _attach_structural_modules(
    det_model: nn.Module,
    layer_indices: Sequence[int],
    gate_init_beta: float,
    gate_gain: float,
    gate_enabled: bool,
    prefix: str,
):
    names = []
    for idx in layer_indices:
        head_name = f"_{prefix}_head_{idx}"
        gate_name = f"_{prefix}_gate_{idx}"
        if not hasattr(det_model, head_name):
            setattr(det_model, head_name, LeafnessHead())
        if not hasattr(det_model, gate_name):
            setattr(
                det_model,
                gate_name,
                ResidualLeafGate(init_beta=gate_init_beta, gain=gate_gain, enabled=gate_enabled),
            )
        names.extend([head_name, gate_name])
    return names


def _register_hooks(det_model: nn.Module, layer_indices: Sequence[int], prefix: str):
    logits_cache: Dict[int, torch.Tensor] = {}
    prob_cache: Dict[int, torch.Tensor] = {}
    handles = []
    for idx in layer_indices:
        module = det_model.model[idx]
        head = getattr(det_model, f"_{prefix}_head_{idx}")
        gate = getattr(det_model, f"_{prefix}_gate_{idx}")
        hook = _LPOTHook(idx, head, gate, logits_cache, prob_cache)
        handles.append(module.register_forward_hook(hook))
    return logits_cache, prob_cache, handles


def _get_runtime(det_model):
    return getattr(det_model, "_lpot_runtime", None)


def _infer_num_classes(det_model) -> int | None:
    try:
        detect_head = det_model.model[-1]
        nc = getattr(detect_head, "nc", None)
        if nc is not None:
            return int(nc)
    except Exception:
        pass
    nc = getattr(det_model, "nc", None)
    return int(nc) if nc is not None else None


def _lpot_score_regularization_enabled(runtime) -> bool:
    fn = getattr(runtime, "lpot_loss_fn", None)
    return bool(getattr(fn, "score_regularization_enabled", False))


def loss_with_lpot(self, batch, preds=None):
    runtime = _get_runtime(self)
    det_loss, loss_items = self._lpot_original_loss(batch, preds)

    if torch.is_tensor(det_loss):
        det_loss_detached = det_loss.detach()
        det_loss_scalar = float(det_loss_detached.sum().item()) if det_loss_detached.numel() > 1 else float(det_loss_detached.item())
    else:
        det_loss_scalar = float(det_loss)

    self.lpot_aux_raw = 0.0
    self.lpot_aux_weighted = 0.0
    self.lpot_aux_ratio = 0.0
    self.lpot_used = 0.0
    self.lpot_prior_raw = 0.0
    self.lpot_proxy_raw = 0.0
    self.lpot_bridge_raw = 0.0
    self.lpot_prior_rank = 0.0
    self.lpot_proxy_rank = 0.0
    self.lpot_support_target_mean = 0.0
    self.lpot_teacher_anchor_raw = 0.0
    self.lpot_teacher_anchor_rank = 0.0
    self.lpot_teacher_anchor_bg = 0.0
    self.lpot_score_raw = 0.0
    self.lpot_score_rank = 0.0
    self.lpot_score_bg = 0.0
    self.lpot_score_pos_mean = 0.0
    self.lpot_score_neg_mean = 0.0
    self.lpot_score_valid_ratio = 0.0
    self.lpot_score_num_maps = 0.0

    if runtime is None:
        return det_loss, loss_items

    if (not self.training) or (not isinstance(batch, dict)) or ("img" not in batch):
        runtime.prior_logits_cache.clear()
        runtime.prior_prob_cache.clear()
        runtime.proxy_logits_cache.clear()
        runtime.proxy_prob_cache.clear()
        return det_loss, loss_items

    missing_prior = [idx for idx in runtime.prior_layer_indices if idx not in runtime.prior_logits_cache]
    missing_proxy = [idx for idx in runtime.proxy_layer_indices if idx not in runtime.proxy_logits_cache]
    if missing_prior or missing_proxy:
        runtime.prior_logits_cache.clear()
        runtime.prior_prob_cache.clear()
        runtime.proxy_logits_cache.clear()
        runtime.proxy_prob_cache.clear()
        _ = self(batch["img"])
        missing_prior = [idx for idx in runtime.prior_layer_indices if idx not in runtime.prior_logits_cache]
        missing_proxy = [idx for idx in runtime.proxy_layer_indices if idx not in runtime.proxy_logits_cache]

    if missing_prior or missing_proxy:
        LOGGER.warning(
            f"[LPOT] Missing cached maps. prior_missing={missing_prior}, proxy_missing={missing_proxy}; skip LPOT for this batch. "
            f"prior={sorted(runtime.prior_logits_cache.keys())}, proxy={sorted(runtime.proxy_logits_cache.keys())}"
        )
        runtime.prior_logits_cache.clear()
        runtime.prior_prob_cache.clear()
        runtime.proxy_logits_cache.clear()
        runtime.proxy_prob_cache.clear()
        return det_loss, loss_items

    preds_for_score = preds
    # v4.1 score-level proposal-support regularization needs YOLO Detect
    # feature maps. If Ultralytics did not pass preds into this wrapper, do one
    # extra forward only when the score-level loss is explicitly enabled.
    if _lpot_score_regularization_enabled(runtime) and preds_for_score is None:
        runtime.prior_logits_cache.clear()
        runtime.prior_prob_cache.clear()
        runtime.proxy_logits_cache.clear()
        runtime.proxy_prob_cache.clear()
        preds_for_score = self(batch["img"])
        missing_prior = [idx for idx in runtime.prior_layer_indices if idx not in runtime.prior_logits_cache]
        missing_proxy = [idx for idx in runtime.proxy_layer_indices if idx not in runtime.proxy_logits_cache]
        if missing_prior or missing_proxy:
            LOGGER.warning(
                f"[LPOT-v4.1] Missing cached maps after score forward. "
                f"prior_missing={missing_prior}, proxy_missing={missing_proxy}; skip LPOT for this batch."
            )
            runtime.prior_logits_cache.clear()
            runtime.prior_prob_cache.clear()
            runtime.proxy_logits_cache.clear()
            runtime.proxy_prob_cache.clear()
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
    prior_logits = [runtime.prior_logits_cache[idx] for idx in runtime.prior_layer_indices]
    prior_probs = [runtime.prior_prob_cache[idx] for idx in runtime.prior_layer_indices]
    proxy_logits = [runtime.proxy_logits_cache[idx] for idx in runtime.proxy_layer_indices]
    proxy_probs = [runtime.proxy_prob_cache[idx] for idx in runtime.proxy_layer_indices]

    lpot_loss, details = runtime.lpot_loss_fn(prior_logits, prior_probs, proxy_logits, proxy_probs, teacher_features)

    if (not torch.is_tensor(lpot_loss)) or (not torch.isfinite(lpot_loss).all()):
        LOGGER.warning("[LPOT] Non-finite LPOT auxiliary loss detected; skip LPOT for this batch.")
        _clear_lpot_caches(runtime)
        return det_loss, loss_items

    score_loss = torch.tensor(0.0, device=lpot_loss.device)
    score_details = {}
    if hasattr(runtime.lpot_loss_fn, "score_level_regularization"):
        # Try multiple sources for YOLO head score maps:
        #   1) preds argument passed by Ultralytics,
        #   2) raw Detect output captured by our forward hook during original loss,
        #   3) a guarded extra forward as a last fallback.
        score_candidates = []
        if preds_for_score is not None:
            score_candidates.append(preds_for_score)
        cached_preds = getattr(runtime, "detect_pred_cache", {}).get("preds", None)
        if cached_preds is not None and cached_preds is not preds_for_score:
            score_candidates.append(cached_preds)

        for candidate in score_candidates:
            score_loss, score_details = runtime.lpot_loss_fn.score_level_regularization(
                candidate,
                batch,
                teacher_features,
                num_classes=_infer_num_classes(self),
            )
            if _safe_float(score_details.get("score_num_maps")) > 0.0:
                break

        if _lpot_score_regularization_enabled(runtime) and _safe_float(score_details.get("score_num_maps")) <= 0.0:
            try:
                runtime.detect_pred_cache.clear()
                extra_preds = self(batch["img"])
                cached_extra = runtime.detect_pred_cache.get("preds", None)
                candidate = cached_extra if cached_extra is not None else extra_preds
                score_loss, score_details = runtime.lpot_loss_fn.score_level_regularization(
                    candidate,
                    batch,
                    teacher_features,
                    num_classes=_infer_num_classes(self),
                )
            except Exception as e:
                LOGGER.warning(f"[LPOT-v4.1] score-level extra forward failed: {e}")

    if (not torch.is_tensor(score_loss)) or (not torch.isfinite(score_loss).all()):
        LOGGER.warning("[LPOT] Non-finite score-level LPOT loss detected; disable score loss for this batch.")
        score_loss = torch.zeros((), device=lpot_loss.device, dtype=lpot_loss.dtype)

    lpot_total_raw = lpot_loss + score_loss
    weighted_lpot = runtime.alpha * lpot_total_raw
    total_loss = det_loss + weighted_lpot

    if (not torch.is_tensor(total_loss)) or (not torch.isfinite(total_loss).all()):
        LOGGER.warning("[LPOT] Non-finite total loss after adding LPOT; fallback to detection loss for this batch.")
        _clear_lpot_caches(runtime)
        return det_loss, loss_items

    aux_raw = _safe_float(lpot_total_raw)
    aux_weighted = _safe_float(weighted_lpot)
    aux_ratio = aux_weighted / max(det_loss_scalar, 1e-12)

    self.lpot_aux_raw = aux_raw
    self.lpot_aux_weighted = aux_weighted
    self.lpot_aux_ratio = aux_ratio
    self.lpot_used = 1.0
    self.lpot_prior_raw = _safe_float(details.get("prior_total"))
    self.lpot_proxy_raw = _safe_float(details.get("proxy_total"))
    self.lpot_bridge_raw = _safe_float(details.get("bridge_total"))
    self.lpot_prior_rank = _safe_float(details.get("prior_rank_mean"))
    self.lpot_proxy_rank = _safe_float(details.get("proxy_rank_mean"))
    self.lpot_support_target_mean = _safe_float(details.get("proxy_target_mean"))
    self.lpot_teacher_anchor_raw = _safe_float(details.get("teacher_anchor_total"))
    self.lpot_teacher_anchor_rank = _safe_float(details.get("teacher_anchor_rank"))
    self.lpot_teacher_anchor_bg = _safe_float(details.get("teacher_anchor_bg"))
    self.lpot_score_raw = _safe_float(score_details.get("score_total"))
    self.lpot_score_rank = _safe_float(score_details.get("score_rank"))
    self.lpot_score_bg = _safe_float(score_details.get("score_bg"))
    self.lpot_score_pos_mean = _safe_float(score_details.get("score_pos_mean"))
    self.lpot_score_neg_mean = _safe_float(score_details.get("score_neg_mean"))
    self.lpot_score_valid_ratio = _safe_float(score_details.get("score_valid_ratio"))
    self.lpot_score_num_maps = _safe_float(score_details.get("score_num_maps"))

    _clear_lpot_caches(runtime)
    return total_loss, loss_items


def patch_detection_model_with_lpot(det_model, lpot_cfg: dict, project_root: Path, verbose=True):
    strip_lpot_runtime(det_model)

    teacher = _build_stage1_teacher(lpot_cfg, project_root, verbose=verbose)
    teacher_branch = str(lpot_cfg.get("teacher_branch", "pos_feats"))
    teacher_feature_indices = list(lpot_cfg.get("teacher_feature_indices", [0, 1, 2]))
    prior_layer_indices = list(lpot_cfg.get("student_layer_indices", [4, 6, 8]))
    proxy_layer_indices = list(lpot_cfg.get("proxy_layer_indices", [15, 18, 21]))
    alpha = float(lpot_cfg.get("alpha", 1.0))
    route_variant, lpot_loss_fn = build_lpot_loss_from_cfg(lpot_cfg)
    enable_feature_gate = bool(lpot_cfg.get("enable_feature_gate", True))
    enable_proxy_gate = bool(lpot_cfg.get("enable_proxy_gate", True))

    structural_names = []
    structural_names += _attach_structural_modules(
        det_model,
        prior_layer_indices,
        float(lpot_cfg.get("gate_init_beta", 0.2)),
        float(lpot_cfg.get("support_gate_gain", 1.0)),
        enable_feature_gate,
        prefix="lpot_prior",
    )
    structural_names += _attach_structural_modules(
        det_model,
        proxy_layer_indices,
        float(lpot_cfg.get("proxy_gate_init_beta", 0.05)),
        float(lpot_cfg.get("proxy_support_gate_gain", lpot_cfg.get("support_gate_gain", 1.0))),
        enable_proxy_gate,
        prefix="lpot_proxy",
    )

    prior_logits_cache, prior_prob_cache, prior_handles = _register_hooks(det_model, prior_layer_indices, prefix="lpot_prior")
    proxy_logits_cache, proxy_prob_cache, proxy_handles = _register_hooks(det_model, proxy_layer_indices, prefix="lpot_proxy")
    detect_pred_cache, detect_handle = _register_detect_output_hook(det_model)
    detect_handles = [detect_handle] if detect_handle is not None else []

    runtime = _LPOTRuntime(
        teacher=teacher,
        teacher_branch=teacher_branch,
        teacher_feature_indices=teacher_feature_indices,
        prior_layer_indices=prior_layer_indices,
        proxy_layer_indices=proxy_layer_indices,
        alpha=alpha,
        route_variant=route_variant,
        enable_feature_gate=enable_feature_gate,
        enable_proxy_gate=enable_proxy_gate,
        lpot_loss_fn=lpot_loss_fn,
        prior_logits_cache=prior_logits_cache,
        prior_prob_cache=prior_prob_cache,
        proxy_logits_cache=proxy_logits_cache,
        proxy_prob_cache=proxy_prob_cache,
        detect_pred_cache=detect_pred_cache,
        hook_handles=prior_handles + proxy_handles + detect_handles,
        structural_names=structural_names,
    )

    object.__setattr__(det_model, "_lpot_runtime", runtime)
    object.__setattr__(det_model, "_lpot_original_loss", det_model.loss)
    object.__setattr__(det_model, "_lpot_patched", True)
    object.__setattr__(det_model, "loss_with_lpot", loss_with_lpot.__get__(det_model, type(det_model)))
    det_model.loss = det_model.loss_with_lpot

    if verbose:
        LOGGER.info("=" * 80)
        LOGGER.info("[LPOT] Enabled")
        LOGGER.info(f"[LPOT] route_variant        : {route_variant}")
        LOGGER.info(f"[LPOT] teacher_branch        : {teacher_branch}")
        LOGGER.info(f"[LPOT] teacher_feature_idx   : {teacher_feature_indices}")
        LOGGER.info(f"[LPOT] prior_layer_idx      : {prior_layer_indices}")
        LOGGER.info(f"[LPOT] proxy_layer_idx      : {proxy_layer_indices}")
        LOGGER.info(f"[LPOT] prior_weights        : {list(lpot_cfg.get('layer_weights', [1.0, 1.0, 1.0]))}")
        LOGGER.info(f"[LPOT] proxy_weights        : {list(lpot_cfg.get('proxy_layer_weights', [1.0, 1.0, 1.0]))}")
        LOGGER.info(f"[LPOT] alpha                : {alpha}")
        LOGGER.info(f"[LPOT] lambda_prior/proxy   : {lpot_cfg.get('lambda_prior', 1.0)} / {lpot_cfg.get('lambda_proxy', 0.5)}")
        LOGGER.info(f"[LPOT] lambda_bridge        : {lpot_cfg.get('lambda_bridge', 0.6)}")
        LOGGER.info(f"[LPOT] objectness_blend     : {lpot_cfg.get('objectness_student_blend', 0.75)}")
        LOGGER.info(f"[LPOT] objectness_smooth    : k={lpot_cfg.get('objectness_smooth_kernel', 7)}, iters={lpot_cfg.get('objectness_smooth_iters', 2)}")
        LOGGER.info(f"[LPOT] lambda_rank/align    : {lpot_cfg.get('lambda_rank', 0.20)} / {lpot_cfg.get('lambda_align', 1.0)}")
        LOGGER.info(f"[LPOT] teacher_anchor      : rank={lpot_cfg.get('lambda_teacher_rank', 0.0)}, bg={lpot_cfg.get('lambda_teacher_bg', 0.0)}, margin={lpot_cfg.get('teacher_anchor_margin', lpot_cfg.get('support_margin', 0.10))}, weights={lpot_cfg.get('teacher_anchor_layer_weights', [])}")
        LOGGER.info(f"[LPOT-v4.1] score_rank/bg   : {lpot_cfg.get('lambda_score_rank', 0.0)} / {lpot_cfg.get('lambda_score_bg', 0.0)}, margin={lpot_cfg.get('score_rank_margin', 0.10)}, weights={lpot_cfg.get('score_rank_layer_weights', [])}, gt_expand={lpot_cfg.get('score_gt_expand_ratio', 1.5)}")
        LOGGER.info(f"[LPOT] enable gates         : prior={enable_feature_gate}, proxy={enable_proxy_gate}")
        LOGGER.info(f"[LPOT] prior/proxy gate gain : {lpot_cfg.get('support_gate_gain', 1.0)} / {lpot_cfg.get('proxy_support_gate_gain', lpot_cfg.get('support_gate_gain', 1.0))}")
        LOGGER.info("=" * 80)
    return det_model


def strip_lpot_runtime(det_model):
    runtime = getattr(det_model, "_lpot_runtime", None)

    if hasattr(det_model, "_lpot_original_loss"):
        det_model.loss = det_model._lpot_original_loss

    if runtime is not None:
        handles = getattr(runtime, "hook_handles", None)
        if handles:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

    structural_names = getattr(runtime, "structural_names", []) if runtime is not None else []
    for name in structural_names:
        if hasattr(det_model, name):
            try:
                delattr(det_model, name)
            except Exception:
                pass

    for name in [
        "_lpot_runtime",
        "_lpot_original_loss",
        "_lpot_patched",
        "loss_with_lpot",
        "lpot_aux_raw",
        "lpot_aux_weighted",
        "lpot_aux_ratio",
        "lpot_used",
        "lpot_prior_raw",
        "lpot_proxy_raw",
        "lpot_bridge_raw",
        "lpot_prior_rank",
        "lpot_proxy_rank",
        "lpot_support_target_mean",
        "lpot_teacher_anchor_raw",
        "lpot_teacher_anchor_rank",
        "lpot_teacher_anchor_bg",
        "lpot_score_raw",
        "lpot_score_rank",
        "lpot_score_bg",
        "lpot_score_pos_mean",
        "lpot_score_neg_mean",
        "lpot_score_valid_ratio",
        "lpot_score_num_maps",
    ]:
        if hasattr(det_model, name):
            try:
                delattr(det_model, name)
            except Exception:
                pass
    return det_model


def _log_lpot_batch_callback(trainer):
    model = unwrap_model(trainer.model)
    if not getattr(model, "_lpot_patched", False):
        return
    trainer._lpot_batch_counter += 1
    interval = int(getattr(trainer, "_lpot_log_interval", 20))
    if trainer._lpot_batch_counter > 0 and interval > 0 and (trainer._lpot_batch_counter % interval == 0):
        LOGGER.info(
            "\n[LPOTMonitor] "
            f"raw={getattr(model, 'lpot_aux_raw', 0.0):.4e}, "
            f"weighted={getattr(model, 'lpot_aux_weighted', 0.0):.4e}, "
            f"ratio={getattr(model, 'lpot_aux_ratio', 0.0):.4e}, "
            f"prior={getattr(model, 'lpot_prior_raw', 0.0):.4e}, "
            f"proxy={getattr(model, 'lpot_proxy_raw', 0.0):.4e}, "
            f"bridge={getattr(model, 'lpot_bridge_raw', 0.0):.4e}, "
            f"prior_rank={getattr(model, 'lpot_prior_rank', 0.0):.4e}, "
            f"proxy_rank={getattr(model, 'lpot_proxy_rank', 0.0):.4e}, "
            f"target_mean={getattr(model, 'lpot_support_target_mean', 0.0):.4e}, "
            f"t_anchor={getattr(model, 'lpot_teacher_anchor_raw', 0.0):.4e}, "
            f"t_rank={getattr(model, 'lpot_teacher_anchor_rank', 0.0):.4e}, "
            f"t_bg={getattr(model, 'lpot_teacher_anchor_bg', 0.0):.4e}, "
            f"score={getattr(model, 'lpot_score_raw', 0.0):.4e}, "
            f"score_rank={getattr(model, 'lpot_score_rank', 0.0):.4e}, "
            f"score_bg={getattr(model, 'lpot_score_bg', 0.0):.4e}, "
            f"score_pos={getattr(model, 'lpot_score_pos_mean', 0.0):.4e}, "
            f"score_neg={getattr(model, 'lpot_score_neg_mean', 0.0):.4e}, "
            f"score_maps={getattr(model, 'lpot_score_num_maps', 0.0):.0f}, "
            f"used={getattr(model, 'lpot_used', 0.0):.0f}"
        )


def _reset_lpot_batch_counter(trainer):
    trainer._lpot_batch_counter = 0


def build_lpot_trainer(
    lpot_cfg: dict,
    project_root: Path,
    base_trainer_det: Type[DetectionTrainer] = DetectionTrainer,
) -> Type[DetectionTrainer]:
    class LPOTTrainer(base_trainer_det):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._lpot_cfg = dict(lpot_cfg)
            self._lpot_project_root = Path(project_root)
            self._lpot_log_interval = int(lpot_cfg.get("log_interval", 20))
            self._lpot_batch_counter = 0

            self.add_callback("on_train_batch_end", _log_lpot_batch_callback)
            self.add_callback("on_train_epoch_start", _reset_lpot_batch_counter)

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            patch_detection_model_with_lpot(model, self._lpot_cfg, self._lpot_project_root)
            return model

        def label_loss_items(self, loss_items=None, prefix="train"):
            items = super().label_loss_items(loss_items=loss_items, prefix=prefix)
            model = unwrap_model(self.model)
            if isinstance(items, dict) and prefix == "train":
                items[f"{prefix}/lpot_aux_raw"] = float(getattr(model, "lpot_aux_raw", 0.0))
                items[f"{prefix}/lpot_aux_weighted"] = float(getattr(model, "lpot_aux_weighted", 0.0))
                items[f"{prefix}/lpot_aux_ratio"] = float(getattr(model, "lpot_aux_ratio", 0.0))
                items[f"{prefix}/lpot_prior_raw"] = float(getattr(model, "lpot_prior_raw", 0.0))
                items[f"{prefix}/lpot_proxy_raw"] = float(getattr(model, "lpot_proxy_raw", 0.0))
                items[f"{prefix}/lpot_bridge_raw"] = float(getattr(model, "lpot_bridge_raw", 0.0))
                items[f"{prefix}/lpot_prior_rank"] = float(getattr(model, "lpot_prior_rank", 0.0))
                items[f"{prefix}/lpot_proxy_rank"] = float(getattr(model, "lpot_proxy_rank", 0.0))
                items[f"{prefix}/lpot_support_target_mean"] = float(getattr(model, "lpot_support_target_mean", 0.0))
                items[f"{prefix}/lpot_teacher_anchor_raw"] = float(getattr(model, "lpot_teacher_anchor_raw", 0.0))
                items[f"{prefix}/lpot_teacher_anchor_rank"] = float(getattr(model, "lpot_teacher_anchor_rank", 0.0))
                items[f"{prefix}/lpot_teacher_anchor_bg"] = float(getattr(model, "lpot_teacher_anchor_bg", 0.0))
                items[f"{prefix}/lpot_score_raw"] = float(getattr(model, "lpot_score_raw", 0.0))
                items[f"{prefix}/lpot_score_rank"] = float(getattr(model, "lpot_score_rank", 0.0))
                items[f"{prefix}/lpot_score_bg"] = float(getattr(model, "lpot_score_bg", 0.0))
                items[f"{prefix}/lpot_score_pos_mean"] = float(getattr(model, "lpot_score_pos_mean", 0.0))
                items[f"{prefix}/lpot_score_neg_mean"] = float(getattr(model, "lpot_score_neg_mean", 0.0))
                items[f"{prefix}/lpot_score_valid_ratio"] = float(getattr(model, "lpot_score_valid_ratio", 0.0))
                items[f"{prefix}/lpot_score_num_maps"] = float(getattr(model, "lpot_score_num_maps", 0.0))
                items[f"{prefix}/lpot_used"] = float(getattr(model, "lpot_used", 0.0))
            return items

        def save_model(self):
            model = unwrap_model(self.model)
            ema_model = getattr(getattr(self, "ema", None), "ema", None)
            ema_model = unwrap_model(ema_model) if ema_model is not None else None

            was_patched_model = getattr(model, "_lpot_patched", False)
            was_patched_ema = getattr(ema_model, "_lpot_patched", False) if ema_model is not None else False

            if was_patched_model or was_patched_ema:
                try:
                    _save_lpot_runtime_sidecar(
                        Path(self.save_dir),
                        self._lpot_cfg,
                        model,
                        ema_model,
                    )
                except Exception as e:
                    LOGGER.warning(f"[LPOT] Failed to save runtime sidecar: {e}")

            if was_patched_model:
                _strip_lpot_runtime_only(model)
            if was_patched_ema:
                _strip_lpot_runtime_only(ema_model)
            try:
                return super().save_model()
            finally:
                if was_patched_model:
                    patch_detection_model_with_lpot(model, self._lpot_cfg, self._lpot_project_root, verbose=False)
                if was_patched_ema:
                    patch_detection_model_with_lpot(ema_model, self._lpot_cfg, self._lpot_project_root, verbose=False)

        def final_eval(self):
            model = unwrap_model(self.model)
            ema_model = getattr(getattr(self, "ema", None), "ema", None)
            ema_model = unwrap_model(ema_model) if ema_model is not None else None

            was_patched_model = getattr(model, "_lpot_patched", False)
            was_patched_ema = getattr(ema_model, "_lpot_patched", False) if ema_model is not None else False

            if was_patched_model:
                strip_lpot_runtime(model)
            if was_patched_ema:
                strip_lpot_runtime(ema_model)
            return super().final_eval()

    LPOTTrainer.__name__ = f"LPOTv2Trainer_{base_trainer_det.__name__}"
    return LPOTTrainer
