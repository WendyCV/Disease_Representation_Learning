from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn

from .stage1_position_aware_module import PositionAwareModule
from .stage1_ssl_projector import MultiScaleProjector
from .stage1_memory_queue import MultiScaleMemoryQueue
from .yolo_backbone import MultiScaleYOLOv8Backbone


def _resolve_position_scale_init(init_values, num_scales: int):
    if init_values is None:
        init_values = [0.1, 0.5, 1.0]
    init_values = list(init_values)
    if len(init_values) == num_scales:
        return init_values
    if len(init_values) < num_scales:
        return init_values + [init_values[-1]] * (num_scales - len(init_values))
    return init_values[:num_scales]


def set_module_trainable(module: nn.Module, is_trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = is_trainable


def count_module_parameters(module: nn.Module):
    total = 0
    trainable = 0
    for parameter in module.parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
    return total, trainable


class Stage1MultiScaleEncoder(nn.Module):
    """Online / momentum / snapshot encoder used in stage-1 SSL.

    The encoder returns a dictionary instead of a dataclass on purpose. This keeps
    checkpoint compatibility with the existing training and visualization scripts.
    """

    def __init__(
        self,
        yolo_model: str = "yolov8n.pt",
        nc: Optional[int] = None,
        layer_indices=(4, 6, 8),
        image_size: int = 640,
        proj_dim: int = 256,
        local_dim: int = 128,
        sppf_indice: int = 9,
        use_pos: bool = True,
        pos_pe_channels: int = 64,
        pos_pe_spans = [1, 1, 1],
        scale_weight_before = True,
        pos_init_scales=None,
        pos_enable_fg_guidance: bool = True,
        pos_fg_gate_init: float = 1.0,
        enable_raw_projection: bool = False,
        separate_projector: bool = False,
        verbose=True,
    ):
        super().__init__()
        self.backbone = MultiScaleYOLOv8Backbone(
            model_path=yolo_model,
            layer_indices=layer_indices,
            sppf_indice=sppf_indice,
            nc=nc,
            verbose=verbose,
        )
        set_module_trainable(self.backbone, True)

        backbone_output_dims = self.backbone.infer_out_dims(image_size=image_size, device="cpu")
        self.out_dims = backbone_output_dims
        self.use_position_module = use_pos
        self.enable_raw_projection = bool(enable_raw_projection)
        self.use_separate_projector = self.enable_raw_projection and separate_projector

        pos_init_scales = _resolve_position_scale_init(pos_init_scales, len(backbone_output_dims))
        if self.use_position_module:
            self.position_modules = nn.ModuleList([
                PositionAwareModule(
                    in_channels=channels,
                    pe_channels=pos_pe_channels,
                    pe_span=pos_pe_spans[index],
                    init_scale=pos_init_scales[index],
                    scale_weight_before=scale_weight_before,
                    enable_fg_guidance=pos_enable_fg_guidance,
                    fg_gate_init=pos_fg_gate_init,
                )
                for index, channels in enumerate(backbone_output_dims)
            ])
        else:
            self.position_modules = None

        self.position_projector = MultiScaleProjector(
            in_channels=backbone_output_dims,
            proj_dim=proj_dim,
            local_dim=local_dim,
            dropout_r=0.1,
        )
        self.raw_projector = (
            MultiScaleProjector(
                in_channels=backbone_output_dims,
                proj_dim=proj_dim,
                local_dim=local_dim,
                dropout_r=0.1,
            )
            if self.use_separate_projector
            else self.position_projector
        )

    def forward(self, images: torch.Tensor, fg_mask: Optional[torch.Tensor] = None) -> Dict[str, object]:
        raw_features, sppf_feat = self.backbone(images)
        if self.use_position_module:
            position_features = [
                position_module(feature_map, fg_mask=fg_mask)
                for feature_map, position_module in zip(raw_features, self.position_modules)
            ]
        else:
            position_features = raw_features

        global_embeddings, local_embeddings = self.position_projector(position_features)

        if self.enable_raw_projection:
            raw_global_embeddings, raw_local_embeddings = self.raw_projector(raw_features)
        else:
            raw_global_embeddings, raw_local_embeddings = None, None

        return {
            "sppf_feat": sppf_feat,
            "raw_feats": raw_features,
            "pos_feats": position_features,
            "global_embs": global_embeddings,
            "local_embs": local_embeddings,
            "raw_global_embs": raw_global_embeddings,
            "raw_local_embs": raw_local_embeddings,
        }


class Stage1SslModel(nn.Module):
    """Full stage-1 SSL model with online, momentum and optional snapshot teacher.

    The implementation keeps the original key names (`q1`, `k1`, `t1`, ...) because
    many of the project's analysis scripts already depend on them.
    """

    def __init__(
        self,
        yolo_model: str = "yolov8n.pt",
        nc: Optional[int] = None,
        layer_indices=(4, 6, 8),
        image_size: int = 640,
        proj_dim: int = 256,
        local_dim: int = 128,
        queue_size: int = 4096,
        momentum: float = 0.999,
        sppf_indice: int = 9,
        use_pos: bool = True,
        pos_pe_channels: int = 64,
        pos_pe_spans=[1, 1, 1],
        scale_weight_before=True,
        pos_init_scales=None,
        pos_enable_fg_guidance: bool = True,
        pos_fg_gate_init: float = 1.0,
        enable_raw_projection: bool = False,
        separate_projector: bool = False,
        use_snapshot_teacher: bool = False,
        verbose=True,
    ):
        super().__init__()
        encoder_kwargs = dict(
            yolo_model=yolo_model,
            nc=nc,
            layer_indices=layer_indices,
            image_size=image_size,
            proj_dim=proj_dim,
            local_dim=local_dim,
            sppf_indice=sppf_indice,
            use_pos=use_pos,
            pos_pe_channels=pos_pe_channels,
            pos_pe_spans=pos_pe_spans,
            scale_weight_before=scale_weight_before,
            pos_init_scales=pos_init_scales,
            pos_enable_fg_guidance=pos_enable_fg_guidance,
            pos_fg_gate_init=pos_fg_gate_init,
            enable_raw_projection=enable_raw_projection,
            separate_projector=separate_projector,
            verbose=verbose
        )

        self.online = Stage1MultiScaleEncoder(**encoder_kwargs)
        set_module_trainable(self.online, True)

        self.momentum = copy.deepcopy(self.online)
        set_module_trainable(self.momentum, False)

        self.use_snapshot_teacher = bool(use_snapshot_teacher)
        if self.use_snapshot_teacher:
            self.snapshot_teacher = Stage1MultiScaleEncoder(**encoder_kwargs)
            set_module_trainable(self.snapshot_teacher, False)
            self.snapshot_teacher.eval()
            self.snapshot_ready = False
        else:
            self.snapshot_teacher = None
            self.snapshot_ready = False

        feature_dims = [proj_dim for _ in self.online.out_dims]
        self.queue = MultiScaleMemoryQueue(feature_dims=feature_dims, queue_size=queue_size)
        self.momentum_decay = momentum
        self.use_position_module = use_pos

        if verbose:
            self._print_parameter_summary()

    def _print_parameter_summary(self) -> None:
        online_total, online_trainable = count_module_parameters(self.online)
        backbone_total, backbone_trainable = count_module_parameters(self.online.backbone)
        momentum_total, momentum_trainable = count_module_parameters(self.momentum)
        print("[Stage1SslModel] Parameter status:")
        print(f"  online_total/trainable      : {online_total} / {online_trainable}")
        print(f"  backbone_total/trainable    : {backbone_total} / {backbone_trainable}")
        print(f"  momentum_total/trainable    : {momentum_total} / {momentum_trainable}")
        if self.use_snapshot_teacher:
            teacher_total, teacher_trainable = count_module_parameters(self.snapshot_teacher)
            print(f"  snapshot_total/trainable    : {teacher_total} / {teacher_trainable}")

    def assert_backbone_trainable(self, raise_if_false: bool = True) -> Dict[str, object]:
        total, trainable = count_module_parameters(self.online.backbone)
        is_valid = trainable > 0
        if not is_valid and raise_if_false:
            raise RuntimeError(
                "The online backbone is frozen. "
                f"backbone_total={total}, backbone_trainable={trainable}. "
                "Training is aborted because raw-supervision losses would become ineffective."
            )
        return {
            "backbone_total": total,
            "backbone_trainable": trainable,
            "ok": is_valid,
        }

    @torch.no_grad()
    def capture_snapshot_teacher(self) -> None:
        if not self.use_snapshot_teacher:
            raise RuntimeError("Snapshot teacher is disabled in this model instance.")
        self.snapshot_teacher.load_state_dict(self.online.state_dict(), strict=True)
        self.snapshot_teacher.eval()
        set_module_trainable(self.snapshot_teacher, False)
        self.snapshot_ready = True
        print("[Stage1SslModel] Snapshot teacher captured from the current online encoder.")

    @torch.no_grad()
    def update_momentum_encoder(self) -> None:
        for online_parameter, momentum_parameter in zip(self.online.parameters(), self.momentum.parameters()):
            momentum_parameter.data.mul_(self.momentum_decay).add_(online_parameter.data, alpha=1.0 - self.momentum_decay)

    def _assert_raw_features_require_grad(self, view_1_output: Dict[str, object], view_2_output: Dict[str, object]) -> None:
        if not (self.training and torch.is_grad_enabled()):
            return
        invalid_tensor_names = []
        for index, feature_map in enumerate(view_1_output["raw_feats"], start=1):
            if not feature_map.requires_grad:
                invalid_tensor_names.append(f"q1.raw_feats[{index}]")
        for index, feature_map in enumerate(view_2_output["raw_feats"], start=1):
            if not feature_map.requires_grad:
                invalid_tensor_names.append(f"q2.raw_feats[{index}]")
        if invalid_tensor_names:
            raise RuntimeError(
                "Online raw feature maps do not require gradients. "
                "This usually means the backbone has been detached or frozen. "
                f"Invalid tensors: {invalid_tensor_names}"
            )

    def forward(
        self,
        view_1_images: torch.Tensor,
        view_2_images: torch.Tensor,
        view_1_mask: Optional[torch.Tensor] = None,
        view_2_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        self.assert_backbone_trainable(raise_if_false=True)

        query_view_1 = self.online(view_1_images, fg_mask=view_1_mask)
        query_view_2 = self.online(view_2_images, fg_mask=view_2_mask)
        self._assert_raw_features_require_grad(query_view_1, query_view_2)

        with torch.no_grad():
            self.update_momentum_encoder()
            key_view_1 = self.momentum(view_1_images, fg_mask=view_1_mask)
            key_view_2 = self.momentum(view_2_images, fg_mask=view_2_mask)

            if self.use_snapshot_teacher and self.snapshot_ready:
                snapshot_view_1 = self.snapshot_teacher(view_1_images, fg_mask=view_1_mask)
                snapshot_view_2 = self.snapshot_teacher(view_2_images, fg_mask=view_2_mask)
            else:
                snapshot_view_1 = None
                snapshot_view_2 = None

        return {
            "q1": query_view_1,
            "q2": query_view_2,
            "k1": key_view_1,
            "k2": key_view_2,
            "t1": snapshot_view_1,
            "t2": snapshot_view_2,
            "queues": self.queue.all_queues(),
        }


# Backward-compatible aliases.
GLCPMultiScaleEncoder = Stage1MultiScaleEncoder
GLCPStage1Model = Stage1SslModel
