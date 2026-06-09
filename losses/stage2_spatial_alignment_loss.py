from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialMapAlignmentLoss(nn.Module):
    """
    Scale-strengthened Spatial Map Alignment Loss

    Compared with the weak old version:
      1) use mean-normalized maps instead of sum-normalized maps
      2) keep whole-map alignment, but avoid collapsing the map magnitude
         to ~1/(H*W), which previously made the raw loss extremely tiny

    Goal:
      make raw alignment loss large enough to influence training,
      while still preserving the original spatial-alignment idea.
    """

    def __init__(
        self,
        layer_weights: Iterable[float],
        loss_type: str = "mse",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.layer_weights = [float(x) for x in layer_weights]
        self.loss_type = str(loss_type)
        self.eps = float(eps)

    def feature_to_map(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Input:
            feat: [B, C, H, W]
        Output:
            map:  [B, 1, H, W]

        IMPORTANT:
        Old weak version often used sum-normalization:
            m = m / m.sum(...)
        which pushed each pixel to extremely small magnitude.

        New version uses mean-normalization:
            m = m / m.mean(...)
        so the map keeps meaningful scale.
        """
        if feat.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(feat.shape)}")

        m = feat.abs().mean(dim=1, keepdim=True)  # [B,1,H,W]

        mean_denom = m.mean(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        m = m / mean_denom

        return m

    def _single_layer_loss(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
            teacher_feat = F.interpolate(
                teacher_feat,
                size=student_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        s_map = self.feature_to_map(student_feat).float()
        t_map = self.feature_to_map(teacher_feat).float()

        if self.loss_type == "mse":
            return F.mse_loss(s_map, t_map)
        elif self.loss_type == "l1":
            return F.l1_loss(s_map, t_map)
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

    def forward(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(student_features) != len(teacher_features):
            raise ValueError(
                f"student_features len {len(student_features)} != teacher_features len {len(teacher_features)}"
            )
        if len(student_features) != len(self.layer_weights):
            raise ValueError(
                f"feature len {len(student_features)} != layer_weights len {len(self.layer_weights)}"
            )

        total = None
        per_layer = []

        for weight, s_feat, t_feat in zip(self.layer_weights, student_features, teacher_features):
            layer_loss = float(weight) * self._single_layer_loss(s_feat, t_feat)
            per_layer.append(layer_loss.detach())
            total = layer_loss if total is None else total + layer_loss

        return total, torch.stack(per_layer)