from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _nan_to_num(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


class LeafPriorAuxiliaryLoss(nn.Module):
    """
    Stage1 teacher provides soft leaf foreground prior maps.
    Student predicts leafness logits on detector feature maps.

    Loss = weighted soft BCE + optional background suppression penalty.

    Stability patch:
      1) all intermediate tensors are cast to float32
      2) all critical tensors pass through nan_to_num
      3) probabilities are clamped before reduction
      4) final total/per-layer losses are sanitized before return
    """

    def __init__(
        self,
        layer_weights: Iterable[float],
        bg_quantile: float = 0.3,
        gamma: float = 1.0,
        lambda_bg: float = 0.25,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.layer_weights = [float(x) for x in layer_weights]
        self.bg_quantile = float(bg_quantile)
        self.gamma = float(gamma)
        self.lambda_bg = float(lambda_bg)
        self.eps = float(eps)

        if not (0.0 < self.bg_quantile < 1.0):
            raise ValueError(f"bg_quantile must be in (0,1), got {self.bg_quantile}")

    def teacher_feature_to_prior(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(feat.shape)}")

        # force float32 for numerical stability
        m = feat.float().abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
        m = _nan_to_num(m)

        # mean-normalize to preserve useful scale
        mean_denom = m.mean(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        m = m / mean_denom
        m = _nan_to_num(m)

        # min-max to [0,1] per image, yielding a soft leafness prior
        flat = m.flatten(2)
        flat = _nan_to_num(flat)

        vmin = flat.min(dim=2, keepdim=True).values.view(m.shape[0], 1, 1, 1)
        vmax = flat.max(dim=2, keepdim=True).values.view(m.shape[0], 1, 1, 1)

        # extra clamp to avoid zero-range pathological cases
        scale = (vmax - vmin).clamp_min(self.eps)
        prior = (m - vmin) / scale
        prior = _nan_to_num(prior).clamp(0.0, 1.0)
        return prior

    def _weighted_soft_bce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = _nan_to_num(logits.float())
        target = _nan_to_num(target.float()).clamp(0.0, 1.0)

        # higher-confidence regions (closer to 0/1) get larger weight
        weight = 1.0 + self.gamma * (target - 0.5).abs() * 2.0
        weight = _nan_to_num(weight)

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        bce = _nan_to_num(bce)

        return _nan_to_num(weight * bce).mean()

    def _background_suppression(self, prob: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        prior = _nan_to_num(prior.float()).clamp(0.0, 1.0)
        prob = _nan_to_num(prob.float()).clamp(0.0, 1.0)

        b, _, h, w = prior.shape
        flat = prior.flatten(2)
        flat = _nan_to_num(flat)

        # quantile requires float/double and can be sensitive to invalid values
        bg_thresh = torch.quantile(flat, self.bg_quantile, dim=2, keepdim=True)
        bg_thresh = _nan_to_num(bg_thresh)

        bg_mask = (flat <= bg_thresh).to(dtype=prob.dtype).view(b, 1, h, w)
        bg_mask = _nan_to_num(bg_mask)

        denom = bg_mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        loss = ((prob * bg_mask).sum(dim=(-2, -1), keepdim=True) / denom).mean()
        return _nan_to_num(loss)

    def forward(
        self,
        student_logits: List[torch.Tensor],
        student_probs: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not (len(student_logits) == len(student_probs) == len(teacher_features) == len(self.layer_weights)):
            raise ValueError(
                "Lengths of student_logits, student_probs, teacher_features, and layer_weights must match."
            )

        total = None
        per_layer = []

        for weight, logit, prob, t_feat in zip(self.layer_weights, student_logits, student_probs, teacher_features):
            prior = self.teacher_feature_to_prior(t_feat)

            if prior.shape[-2:] != logit.shape[-2:]:
                prior = F.interpolate(prior, size=logit.shape[-2:], mode="bilinear", align_corners=False)
                prior = _nan_to_num(prior).clamp(0.0, 1.0)

            logit = _nan_to_num(logit.float())
            prob = _nan_to_num(prob.float()).clamp(0.0, 1.0)

            l_leaf = self._weighted_soft_bce(logit, prior)
            l_bg = self._background_suppression(prob, prior)

            layer_loss = float(weight) * (l_leaf + self.lambda_bg * l_bg)
            layer_loss = _nan_to_num(layer_loss)

            per_layer.append(layer_loss.detach())
            total = layer_loss if total is None else total + layer_loss

        total = _nan_to_num(total)
        per_layer = _nan_to_num(torch.stack(per_layer))
        return total, per_layer