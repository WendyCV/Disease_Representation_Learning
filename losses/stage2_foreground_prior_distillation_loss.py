from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForegroundPriorDistillationLoss(nn.Module):
    """
    Foreground Prior Distillation (FPD)

    Teacher provides a soft foreground prior:
      - high-response region => likely leaf foreground
      - low-response region  => likely background

    This version is a scale-strengthened patch:
      1) use mean-normalized maps instead of sum-normalized maps
      2) use masked SUM-style region losses instead of overly weak region means

    Goal:
      make raw loss much larger than ~3e-4, ideally reaching ~1e-2 ~ 1e-1
      so that alpha-scaled aux loss can meaningfully affect optimization.
    """

    def __init__(
        self,
        layer_weights: Iterable[float],
        fg_quantile: float = 0.7,
        bg_quantile: float = 0.3,
        lambda_fg: float = 1.0,
        lambda_bg: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.layer_weights = [float(x) for x in layer_weights]
        self.fg_quantile = float(fg_quantile)
        self.bg_quantile = float(bg_quantile)
        self.lambda_fg = float(lambda_fg)
        self.lambda_bg = float(lambda_bg)
        self.eps = float(eps)

        if not (0.0 < self.bg_quantile < self.fg_quantile < 1.0):
            raise ValueError(
                f"Require 0 < bg_quantile < fg_quantile < 1, "
                f"but got bg={self.bg_quantile}, fg={self.fg_quantile}"
            )

    def feature_to_map(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Input:
            feat: [B, C, H, W]
        Output:
            map : [B, 1, H, W]

        IMPORTANT:
        Old weak version used sum-normalization:
            m = m / m.sum(...)
        which made each pixel extremely small.

        New stronger version uses mean-normalization:
            m = m / m.mean(...)
        so the map keeps meaningful magnitude.
        """
        if feat.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(feat.shape)}")

        # basic activation map
        m = feat.abs().mean(dim=1, keepdim=True)  # [B,1,H,W]

        # mean-normalize instead of sum-normalize
        mean_denom = m.mean(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        m = m / mean_denom

        return m

    def _make_fg_bg_masks(self, teacher_map: torch.Tensor):
        """
        teacher_map: [B,1,H,W]
        Use teacher's high-response area as foreground prior,
        low-response area as background prior.
        """
        b, _, h, w = teacher_map.shape

        # quantile requires float32/float64, especially under AMP where teacher_map may be fp16
        flat = teacher_map.float().flatten(2)  # [B,1,HW]

        fg_thresh = torch.quantile(flat, self.fg_quantile, dim=2, keepdim=True)
        bg_thresh = torch.quantile(flat, self.bg_quantile, dim=2, keepdim=True)

        fg_mask = (flat >= fg_thresh).to(dtype=teacher_map.dtype).view(b, 1, h, w)
        bg_mask = (flat <= bg_thresh).to(dtype=teacher_map.dtype).view(b, 1, h, w)

        return fg_mask, bg_mask

    def _masked_sum(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Stronger than masked mean:
        sum over spatial region, then average over batch.
        """
        return (value * mask).sum(dim=(-2, -1)).mean()

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
            if s_feat.shape[-2:] != t_feat.shape[-2:]:
                t_feat = F.interpolate(t_feat, size=s_feat.shape[-2:], mode="bilinear", align_corners=False)

            s_map = self.feature_to_map(s_feat).float()  # [B,1,H,W]
            t_map = self.feature_to_map(t_feat).float()  # [B,1,H,W]

            fg_mask, bg_mask = self._make_fg_bg_masks(t_map)

            # Foreground consistency:
            # student should match teacher on teacher-high-response (leaf foreground) regions
            fg_diff = (s_map - t_map).pow(2)
            l_fg = self._masked_sum(fg_diff, fg_mask)

            # Background suppression:
            # teacher-low-response regions should stay low for student
            l_bg = self._masked_sum(s_map, bg_mask)

            layer_loss = weight * (self.lambda_fg * l_fg + self.lambda_bg * l_bg)
            per_layer.append(layer_loss.detach())

            total = layer_loss if total is None else total + layer_loss

        return total, torch.stack(per_layer)