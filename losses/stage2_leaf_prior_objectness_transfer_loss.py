from __future__ import annotations

from typing import Any, Iterable, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _nan_to_num(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


class LegacyLeafPriorObjectnessTransferLoss(nn.Module):
    """
    LPOT v2 (minimal revision)
    -------------------------
    Core fixes over v1:
      1) proxy branch is no longer supervised by fused teacher prior directly.
         Instead it is supervised by an OBJECTNESS target built mainly from
         detached backbone prior predictions, with only light teacher mixing.
      2) bridge loss is strengthened from weak MSE-on-probabilities to BCE on
         proxy logits against the detached objectness target.
      3) objectness target is spatially smoothed so that the proxy branch learns
         broader, detection-friendly support rather than narrow hotspot stripes.

    This keeps the architecture unchanged and only modifies the target/loss form.
    """

    def __init__(
        self,
        prior_layer_weights: Iterable[float],
        proxy_layer_weights: Iterable[float],
        teacher_fuse_weights: Iterable[float] | None = None,
        gamma: float = 1.0,
        bg_quantile: float = 0.3,
        lambda_bg: float = 0.25,
        lambda_prior: float = 1.0,
        lambda_proxy: float = 0.5,
        lambda_bridge: float = 0.6,
        objectness_student_blend: float = 0.75,
        objectness_smooth_kernel: int = 7,
        objectness_smooth_iters: int = 2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.prior_layer_weights = [float(x) for x in prior_layer_weights]
        self.proxy_layer_weights = [float(x) for x in proxy_layer_weights]
        self.teacher_fuse_weights = [float(x) for x in (teacher_fuse_weights or self.prior_layer_weights)]
        self.gamma = float(gamma)
        self.bg_quantile = float(bg_quantile)
        self.lambda_bg = float(lambda_bg)
        self.lambda_prior = float(lambda_prior)
        self.lambda_proxy = float(lambda_proxy)
        self.lambda_bridge = float(lambda_bridge)
        self.objectness_student_blend = float(objectness_student_blend)
        self.objectness_smooth_kernel = int(objectness_smooth_kernel)
        self.objectness_smooth_iters = int(objectness_smooth_iters)
        self.eps = float(eps)

        if not (0.0 < self.bg_quantile < 1.0):
            raise ValueError(f"bg_quantile must be in (0,1), got {self.bg_quantile}")
        if not (0.0 <= self.objectness_student_blend <= 1.0):
            raise ValueError(f"objectness_student_blend must be in [0,1], got {self.objectness_student_blend}")
        if self.objectness_smooth_kernel < 1 or self.objectness_smooth_kernel % 2 == 0:
            raise ValueError("objectness_smooth_kernel must be an odd positive integer")

    def teacher_feature_to_prior(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(feat.shape)}")

        m = feat.float().abs().mean(dim=1, keepdim=True)
        m = _nan_to_num(m)
        mean_denom = m.mean(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        m = _nan_to_num(m / mean_denom)

        flat = _nan_to_num(m.flatten(2))
        vmin = flat.min(dim=2, keepdim=True).values.view(m.shape[0], 1, 1, 1)
        vmax = flat.max(dim=2, keepdim=True).values.view(m.shape[0], 1, 1, 1)
        scale = (vmax - vmin).clamp_min(self.eps)
        prior = _nan_to_num((m - vmin) / scale).clamp(0.0, 1.0)
        return prior

    def _resize(self, x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == size_hw:
            return x
        return _nan_to_num(F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False))

    def _renorm01(self, x: torch.Tensor) -> torch.Tensor:
        x = _nan_to_num(x.float())
        flat = x.flatten(2)
        vmin = flat.min(dim=2, keepdim=True).values.view(x.shape[0], 1, 1, 1)
        vmax = flat.max(dim=2, keepdim=True).values.view(x.shape[0], 1, 1, 1)
        scale = (vmax - vmin).clamp_min(self.eps)
        return _nan_to_num((x - vmin) / scale).clamp(0.0, 1.0)

    def _smooth_map(self, x: torch.Tensor) -> torch.Tensor:
        x = _nan_to_num(x.float())
        k = self.objectness_smooth_kernel
        p = k // 2
        for _ in range(max(self.objectness_smooth_iters, 0)):
            avg = F.avg_pool2d(x, kernel_size=k, stride=1, padding=p)
            mx = F.max_pool2d(x, kernel_size=k, stride=1, padding=p)
            x = 0.5 * avg + 0.5 * mx
            x = self._renorm01(x)
        return x

    def _weighted_soft_bce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = _nan_to_num(logits.float())
        target = _nan_to_num(target.float()).clamp(0.0, 1.0)
        weight = _nan_to_num(1.0 + self.gamma * (target - 0.5).abs() * 2.0)
        bce = _nan_to_num(F.binary_cross_entropy_with_logits(logits, target, reduction="none"))
        return _nan_to_num(weight * bce).mean()

    def _background_suppression(self, prob: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        prior = _nan_to_num(prior.float()).clamp(0.0, 1.0)
        prob = _nan_to_num(prob.float()).clamp(0.0, 1.0)

        b, _, h, w = prior.shape
        flat = _nan_to_num(prior.flatten(2))
        bg_thresh = _nan_to_num(torch.quantile(flat, self.bg_quantile, dim=2, keepdim=True))
        bg_mask = _nan_to_num((flat <= bg_thresh).to(dtype=prob.dtype).view(b, 1, h, w))
        denom = bg_mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        loss = ((prob * bg_mask).sum(dim=(-2, -1), keepdim=True) / denom).mean()
        return _nan_to_num(loss)

    def _single_prior_loss(self, logits: torch.Tensor, prob: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        prior = self._resize(prior, logits.shape[-2:]).clamp(0.0, 1.0)
        logits = _nan_to_num(logits.float())
        prob = _nan_to_num(prob.float()).clamp(0.0, 1.0)
        l_leaf = self._weighted_soft_bce(logits, prior)
        l_bg = self._background_suppression(prob, prior)
        return _nan_to_num(l_leaf + self.lambda_bg * l_bg)

    def _fuse_teacher_prior(self, teacher_features: List[torch.Tensor], out_size: Tuple[int, int]) -> torch.Tensor:
        if len(teacher_features) != len(self.teacher_fuse_weights):
            raise ValueError(
                f"teacher_features length {len(teacher_features)} != teacher_fuse_weights length {len(self.teacher_fuse_weights)}"
            )
        weight_sum = 0.0
        fused = None
        for feat, w in zip(teacher_features, self.teacher_fuse_weights):
            prior = self.teacher_feature_to_prior(feat)
            prior = self._resize(prior, out_size).clamp(0.0, 1.0)
            w = float(w)
            weight_sum += w
            fused = prior * w if fused is None else fused + prior * w
        fused = fused / max(weight_sum, self.eps)
        fused = self._renorm01(fused)
        return fused

    def _build_student_objectness_target(self, prior_probs: List[torch.Tensor], out_size: Tuple[int, int]) -> torch.Tensor:
        if len(prior_probs) == 0:
            raise ValueError("prior_probs must be non-empty")
        fused = None
        for p in prior_probs:
            p = _nan_to_num(p.detach().float()).clamp(0.0, 1.0)
            p = self._resize(p, out_size)
            fused = p if fused is None else fused + p
        fused = fused / max(len(prior_probs), 1)
        fused = self._renorm01(fused)
        fused = self._smooth_map(fused)
        return fused

    def _build_proxy_target(self, teacher_features: List[torch.Tensor], prior_probs: List[torch.Tensor], out_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        teacher_fused = self._fuse_teacher_prior(teacher_features, out_size)
        student_obj = self._build_student_objectness_target(prior_probs, out_size)
        target = (
            self.objectness_student_blend * student_obj
            + (1.0 - self.objectness_student_blend) * teacher_fused
        )
        target = self._renorm01(target)
        return target, student_obj

    def _bridge_consistency(self, proxy_logits: List[torch.Tensor], detached_objectness_target: torch.Tensor) -> torch.Tensor:
        if len(proxy_logits) == 0:
            return torch.tensor(0.0, device=detached_objectness_target.device)
        losses = []
        for logit in proxy_logits:
            tgt = self._resize(detached_objectness_target, logit.shape[-2:]).clamp(0.0, 1.0)
            losses.append(self._weighted_soft_bce(logit, tgt))
        return _nan_to_num(torch.stack(losses).mean())

    def forward(
        self,
        prior_logits: List[torch.Tensor],
        prior_probs: List[torch.Tensor],
        proxy_logits: List[torch.Tensor],
        proxy_probs: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if len(prior_logits) != len(self.prior_layer_weights) or len(prior_probs) != len(self.prior_layer_weights):
            raise ValueError("prior branch lengths do not match prior_layer_weights")
        if len(proxy_logits) != len(self.proxy_layer_weights) or len(proxy_probs) != len(self.proxy_layer_weights):
            raise ValueError("proxy branch lengths do not match proxy_layer_weights")

        prior_per_layer = []
        for w, logit, prob, t_feat in zip(self.prior_layer_weights, prior_logits, prior_probs, teacher_features):
            target_prior = self.teacher_feature_to_prior(t_feat)
            loss = self._single_prior_loss(logit, prob, target_prior)
            prior_per_layer.append(_nan_to_num(loss * float(w)))
        prior_total = _nan_to_num(torch.stack(prior_per_layer).sum()) if prior_per_layer else torch.tensor(0.0)

        if proxy_logits:
            proxy_target, detached_objectness_target = self._build_proxy_target(
                teacher_features,
                prior_probs,
                proxy_logits[0].shape[-2:],
            )
            proxy_per_layer = []
            for w, logit, prob in zip(self.proxy_layer_weights, proxy_logits, proxy_probs):
                loss = self._single_prior_loss(logit, prob, proxy_target)
                proxy_per_layer.append(_nan_to_num(loss * float(w)))
            proxy_total = _nan_to_num(torch.stack(proxy_per_layer).sum())
            bridge_total = _nan_to_num(self._bridge_consistency(proxy_logits, detached_objectness_target)) if self.lambda_bridge > 0 else torch.tensor(0.0, device=prior_total.device)
        else:
            proxy_target = torch.zeros(1, device=prior_total.device)
            detached_objectness_target = torch.zeros(1, device=prior_total.device)
            proxy_per_layer = []
            proxy_total = torch.tensor(0.0, device=prior_total.device)
            bridge_total = torch.tensor(0.0, device=prior_total.device)

        total = _nan_to_num(
            self.lambda_prior * prior_total
            + self.lambda_proxy * proxy_total
            + self.lambda_bridge * bridge_total
        )

        details = {
            "prior_total": _nan_to_num(prior_total.detach()),
            "proxy_total": _nan_to_num(proxy_total.detach()),
            "bridge_total": _nan_to_num(bridge_total.detach()),
            "prior_per_layer": _nan_to_num(torch.stack(prior_per_layer).detach()) if prior_per_layer else torch.zeros(0),
            "proxy_per_layer": _nan_to_num(torch.stack(proxy_per_layer).detach()) if proxy_per_layer else torch.zeros(0),
            "proxy_target_mean": _nan_to_num(proxy_target.detach()).mean(),
            "student_objectness_mean": _nan_to_num(detached_objectness_target.detach()).mean(),
        }
        return total, details


class ProposalSupportPriorLeafPriorObjectnessTransferLoss(nn.Module):
    """
    LPOT-v4: lightweight Proposal-Support Prior (PSP).

    Narrative:
      Stage1 high-response regions are treated as foreground/non-background
      proposal-support prior. The loss should raise support in high-prior
      regions, suppress support in background-dominant regions, and optionally
      keep a very soft bridge from backbone support to pre-head support.

    Difference from heavier PSP:
      - lambda_align controls dense heatmap matching.
      - setting lambda_align=0.0 gives ranking/background-only supervision.
      - this better matches "proposal support" rather than pixel-level heatmap imitation.
    """

    def __init__(
        self,
        prior_layer_weights: Iterable[float],
        proxy_layer_weights: Iterable[float],
        teacher_fuse_weights: Iterable[float] | None = None,
        gamma: float = 1.0,
        fg_quantile: float = 0.7,
        bg_quantile: float = 0.3,
        lambda_bg: float = 0.25,
        lambda_rank: float = 0.05,
        lambda_align: float = 0.0,
        support_margin: float = 0.10,
        lambda_prior: float = 0.5,
        lambda_proxy: float = 0.25,
        lambda_bridge: float = 0.2,
        objectness_student_blend: float = 0.75,
        objectness_smooth_kernel: int = 7,
        objectness_smooth_iters: int = 2,
        # LPOT-v5 teacher-anchored proxy support terms.
        lambda_teacher_rank: float = 0.0,
        lambda_teacher_bg: float = 0.0,
        teacher_anchor_margin: float | None = None,
        teacher_fg_quantile: float | None = None,
        teacher_bg_quantile: float | None = None,
        teacher_anchor_layer_weights: Iterable[float] | None = None,
        # LPOT-v4.1 score-level proposal-support regularization.
        # Disabled by default. It calibrates YOLO candidate scores rather than
        # forcing proxy/feature heatmaps to imitate the teacher prior.
        lambda_score_rank: float = 0.0,
        lambda_score_bg: float = 0.0,
        score_rank_margin: float = 0.10,
        score_fg_quantile: float = 0.70,
        score_bg_quantile: float = 0.30,
        score_rank_layer_weights: Iterable[float] | None = None,
        score_gt_expand_ratio: float = 1.5,
        score_use_gt_pos: bool = True,
        score_use_teacher_fg_pos: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.prior_layer_weights = [float(x) for x in prior_layer_weights]
        self.proxy_layer_weights = [float(x) for x in proxy_layer_weights]
        self.teacher_fuse_weights = [float(x) for x in (teacher_fuse_weights or self.prior_layer_weights)]
        self.gamma = float(gamma)
        self.fg_quantile = float(fg_quantile)
        self.bg_quantile = float(bg_quantile)
        self.lambda_bg = float(lambda_bg)
        self.lambda_rank = float(lambda_rank)
        self.lambda_align = float(lambda_align)
        self.support_margin = float(support_margin)
        self.lambda_prior = float(lambda_prior)
        self.lambda_proxy = float(lambda_proxy)
        self.lambda_bridge = float(lambda_bridge)
        self.objectness_student_blend = float(objectness_student_blend)
        self.objectness_smooth_kernel = int(objectness_smooth_kernel)
        self.objectness_smooth_iters = int(objectness_smooth_iters)
        self.lambda_teacher_rank = float(lambda_teacher_rank)
        self.lambda_teacher_bg = float(lambda_teacher_bg)
        self.teacher_anchor_margin = float(teacher_anchor_margin) if teacher_anchor_margin is not None else float(support_margin)
        self.teacher_fg_quantile = float(teacher_fg_quantile) if teacher_fg_quantile is not None else float(fg_quantile)
        self.teacher_bg_quantile = float(teacher_bg_quantile) if teacher_bg_quantile is not None else float(bg_quantile)
        self.teacher_anchor_layer_weights = ([float(x) for x in teacher_anchor_layer_weights]
                                             if teacher_anchor_layer_weights is not None else None)
        self.lambda_score_rank = float(lambda_score_rank)
        self.lambda_score_bg = float(lambda_score_bg)
        self.score_rank_margin = float(score_rank_margin)
        self.score_fg_quantile = float(score_fg_quantile)
        self.score_bg_quantile = float(score_bg_quantile)
        self.score_rank_layer_weights = ([float(x) for x in score_rank_layer_weights]
                                         if score_rank_layer_weights is not None else None)
        self.score_gt_expand_ratio = float(score_gt_expand_ratio)
        self.score_use_gt_pos = bool(score_use_gt_pos)
        self.score_use_teacher_fg_pos = bool(score_use_teacher_fg_pos)
        self.eps = float(eps)

        if not (0.0 < self.fg_quantile < 1.0):
            raise ValueError(f"fg_quantile must be in (0,1), got {self.fg_quantile}")
        if not (0.0 < self.bg_quantile < 1.0):
            raise ValueError(f"bg_quantile must be in (0,1), got {self.bg_quantile}")
        if self.fg_quantile <= self.bg_quantile:
            raise ValueError(f"fg_quantile must be > bg_quantile, got {self.fg_quantile} <= {self.bg_quantile}")
        if not (0.0 <= self.objectness_student_blend <= 1.0):
            raise ValueError(f"objectness_student_blend must be in [0,1], got {self.objectness_student_blend}")
        if self.objectness_smooth_kernel < 1 or self.objectness_smooth_kernel % 2 == 0:
            raise ValueError("objectness_smooth_kernel must be an odd positive integer")
        if self.lambda_align < 0:
            raise ValueError(f"lambda_align must be >= 0, got {self.lambda_align}")
        if self.lambda_teacher_rank < 0:
            raise ValueError(f"lambda_teacher_rank must be >= 0, got {self.lambda_teacher_rank}")
        if self.lambda_teacher_bg < 0:
            raise ValueError(f"lambda_teacher_bg must be >= 0, got {self.lambda_teacher_bg}")
        if not (0.0 < self.teacher_fg_quantile < 1.0):
            raise ValueError(f"teacher_fg_quantile must be in (0,1), got {self.teacher_fg_quantile}")
        if not (0.0 < self.teacher_bg_quantile < 1.0):
            raise ValueError(f"teacher_bg_quantile must be in (0,1), got {self.teacher_bg_quantile}")
        if self.teacher_fg_quantile <= self.teacher_bg_quantile:
            raise ValueError(
                f"teacher_fg_quantile must be > teacher_bg_quantile, "
                f"got {self.teacher_fg_quantile} <= {self.teacher_bg_quantile}"
            )
        if self.teacher_anchor_layer_weights is not None and len(self.teacher_anchor_layer_weights) == 0:
            raise ValueError("teacher_anchor_layer_weights cannot be empty when provided")
        if self.lambda_score_rank < 0:
            raise ValueError(f"lambda_score_rank must be >= 0, got {self.lambda_score_rank}")
        if self.lambda_score_bg < 0:
            raise ValueError(f"lambda_score_bg must be >= 0, got {self.lambda_score_bg}")
        if not (0.0 < self.score_fg_quantile < 1.0):
            raise ValueError(f"score_fg_quantile must be in (0,1), got {self.score_fg_quantile}")
        if not (0.0 < self.score_bg_quantile < 1.0):
            raise ValueError(f"score_bg_quantile must be in (0,1), got {self.score_bg_quantile}")
        if self.score_fg_quantile <= self.score_bg_quantile:
            raise ValueError(
                f"score_fg_quantile must be > score_bg_quantile, "
                f"got {self.score_fg_quantile} <= {self.score_bg_quantile}"
            )
        if self.score_rank_layer_weights is not None and len(self.score_rank_layer_weights) == 0:
            raise ValueError("score_rank_layer_weights cannot be empty when provided")
        if self.score_gt_expand_ratio <= 0:
            raise ValueError(f"score_gt_expand_ratio must be > 0, got {self.score_gt_expand_ratio}")

    @property
    def score_regularization_enabled(self) -> bool:
        return bool(self.lambda_score_rank > 0.0 or self.lambda_score_bg > 0.0)

    def teacher_feature_to_prior(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(feat.shape)}")
        m = feat.float().abs().mean(dim=1, keepdim=True)
        m = _nan_to_num(m)
        mean_denom = m.mean(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        m = _nan_to_num(m / mean_denom)
        flat = _nan_to_num(m.flatten(2))
        vmin = flat.min(dim=2, keepdim=True).values.view(m.shape[0], 1, 1, 1)
        vmax = flat.max(dim=2, keepdim=True).values.view(m.shape[0], 1, 1, 1)
        scale = (vmax - vmin).clamp_min(self.eps)
        return _nan_to_num((m - vmin) / scale).clamp(0.0, 1.0)

    def _resize(self, x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == size_hw:
            return x
        return _nan_to_num(F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False))

    def _renorm01(self, x: torch.Tensor) -> torch.Tensor:
        x = _nan_to_num(x.float())
        flat = x.flatten(2)
        vmin = flat.min(dim=2, keepdim=True).values.view(x.shape[0], 1, 1, 1)
        vmax = flat.max(dim=2, keepdim=True).values.view(x.shape[0], 1, 1, 1)
        scale = (vmax - vmin).clamp_min(self.eps)
        return _nan_to_num((x - vmin) / scale).clamp(0.0, 1.0)

    def _smooth_map(self, x: torch.Tensor) -> torch.Tensor:
        x = _nan_to_num(x.float())
        k = self.objectness_smooth_kernel
        p = k // 2
        for _ in range(max(self.objectness_smooth_iters, 0)):
            avg = F.avg_pool2d(x, kernel_size=k, stride=1, padding=p)
            mx = F.max_pool2d(x, kernel_size=k, stride=1, padding=p)
            x = 0.5 * avg + 0.5 * mx
            x = self._renorm01(x)
        return x

    def _weighted_soft_bce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = _nan_to_num(logits.float())
        target = _nan_to_num(target.float()).clamp(0.0, 1.0)
        weight = _nan_to_num(1.0 + self.gamma * (target - 0.5).abs() * 2.0)
        bce = _nan_to_num(F.binary_cross_entropy_with_logits(logits, target, reduction="none"))
        return _nan_to_num(weight * bce).mean()

    def _background_suppression(self, prob: torch.Tensor, support_prior: torch.Tensor) -> torch.Tensor:
        support_prior = _nan_to_num(support_prior.float()).clamp(0.0, 1.0)
        prob = _nan_to_num(prob.float()).clamp(0.0, 1.0)

        b, _, h, w = support_prior.shape
        flat = _nan_to_num(support_prior.flatten(2))
        bg_thresh = _nan_to_num(torch.quantile(flat, self.bg_quantile, dim=2, keepdim=True))
        bg_mask = _nan_to_num((flat <= bg_thresh).to(dtype=prob.dtype).view(b, 1, h, w))
        denom = bg_mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        return _nan_to_num(((prob * bg_mask).sum(dim=(-2, -1), keepdim=True) / denom).mean())

    def _support_ranking(self, prob: torch.Tensor, support_prior: torch.Tensor) -> torch.Tensor:
        support_prior = _nan_to_num(support_prior.float()).clamp(0.0, 1.0)
        prob = _nan_to_num(prob.float()).clamp(0.0, 1.0)

        b, _, h, w = support_prior.shape
        flat = _nan_to_num(support_prior.flatten(2))
        fg_thresh = _nan_to_num(torch.quantile(flat, self.fg_quantile, dim=2, keepdim=True))
        bg_thresh = _nan_to_num(torch.quantile(flat, self.bg_quantile, dim=2, keepdim=True))
        fg_mask = _nan_to_num((flat >= fg_thresh).to(dtype=prob.dtype).view(b, 1, h, w))
        bg_mask = _nan_to_num((flat <= bg_thresh).to(dtype=prob.dtype).view(b, 1, h, w))

        fg_denom = fg_mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        bg_denom = bg_mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        fg_mean = (prob * fg_mask).sum(dim=(-2, -1), keepdim=True) / fg_denom
        bg_mean = (prob * bg_mask).sum(dim=(-2, -1), keepdim=True) / bg_denom
        margin_gap = _nan_to_num(fg_mean - bg_mean)
        return _nan_to_num(F.softplus(self.support_margin - margin_gap)).mean()

    def _quantile_ranking(
        self,
        prob: torch.Tensor,
        support_prior: torch.Tensor,
        *,
        fg_quantile: float,
        bg_quantile: float,
        margin: float,
    ) -> torch.Tensor:
        """Rank support probabilities using high/low teacher-prior regions.

        LPOT-v5 uses this as a teacher-anchor term: high Stage1-prior regions
        should receive higher pre-head proxy support than low Stage1-prior
        regions. This avoids dense heatmap imitation and matches proposal
        support transfer.
        """
        support_prior = _nan_to_num(support_prior.float()).clamp(0.0, 1.0)
        prob = _nan_to_num(prob.float()).clamp(0.0, 1.0)

        b, _, h, w = support_prior.shape
        flat = _nan_to_num(support_prior.flatten(2))
        fg_thresh = _nan_to_num(torch.quantile(flat, float(fg_quantile), dim=2, keepdim=True))
        bg_thresh = _nan_to_num(torch.quantile(flat, float(bg_quantile), dim=2, keepdim=True))
        fg_mask = _nan_to_num((flat >= fg_thresh).to(dtype=prob.dtype).view(b, 1, h, w))
        bg_mask = _nan_to_num((flat <= bg_thresh).to(dtype=prob.dtype).view(b, 1, h, w))

        fg_denom = fg_mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        bg_denom = bg_mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        fg_mean = (prob * fg_mask).sum(dim=(-2, -1), keepdim=True) / fg_denom
        bg_mean = (prob * bg_mask).sum(dim=(-2, -1), keepdim=True) / bg_denom
        margin_gap = _nan_to_num(fg_mean - bg_mean)
        return _nan_to_num(F.softplus(float(margin) - margin_gap)).mean()

    def _teacher_anchor_loss(
        self,
        proxy_probs: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """LPOT-v5 teacher-anchor over proxy support probabilities.

        It explicitly transfers the Stage1 teacher prior into objectness-like
        pre-head proxy support, while keeping the detector student-adaptive.
        """
        if len(proxy_probs) == 0 or (self.lambda_teacher_rank <= 0 and self.lambda_teacher_bg <= 0):
            device = proxy_probs[0].device if proxy_probs else teacher_features[0].device
            z = torch.tensor(0.0, device=device)
            return z, {
                "teacher_rank": z.detach(),
                "teacher_bg": z.detach(),
                "teacher_anchor_per_layer": torch.zeros(0, device=device),
            }

        weights = self.teacher_anchor_layer_weights
        if weights is None:
            weights = [1.0 for _ in proxy_probs]
        if len(weights) != len(proxy_probs):
            raise ValueError(
                f"teacher_anchor_layer_weights length {len(weights)} must match proxy_probs length {len(proxy_probs)}"
            )

        per_layer = []
        rank_items = []
        bg_items = []
        weight_sum = 0.0
        for prob, w in zip(proxy_probs, weights):
            w = float(w)
            if w == 0.0:
                continue

            teacher_prior = self._fuse_teacher_prior(teacher_features, prob.shape[-2:]).detach().clamp(0.0, 1.0)
            prob = _nan_to_num(prob.float()).clamp(0.0, 1.0)

            l_rank = self._quantile_ranking(
                prob,
                teacher_prior,
                fg_quantile=self.teacher_fg_quantile,
                bg_quantile=self.teacher_bg_quantile,
                margin=self.teacher_anchor_margin,
            )
            l_bg = self._background_suppression(prob, teacher_prior)
            l_total = _nan_to_num(self.lambda_teacher_rank * l_rank + self.lambda_teacher_bg * l_bg)

            per_layer.append(l_total * w)
            rank_items.append(l_rank.detach())
            bg_items.append(l_bg.detach())
            weight_sum += w

        if not per_layer:
            device = proxy_probs[0].device
            z = torch.tensor(0.0, device=device)
            return z, {
                "teacher_rank": z.detach(),
                "teacher_bg": z.detach(),
                "teacher_anchor_per_layer": torch.zeros(0, device=device),
            }

        total = _nan_to_num(torch.stack(per_layer).sum() / max(weight_sum, self.eps))
        details = {
            "teacher_rank": _nan_to_num(torch.stack(rank_items).mean().detach()) if rank_items else total.detach() * 0,
            "teacher_bg": _nan_to_num(torch.stack(bg_items).mean().detach()) if bg_items else total.detach() * 0,
            "teacher_anchor_per_layer": _nan_to_num(torch.stack(per_layer).detach()),
        }
        return total, details

    def _single_support_loss(
        self,
        logits: torch.Tensor,
        prob: torch.Tensor,
        support_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        support_prior = self._resize(support_prior, logits.shape[-2:]).clamp(0.0, 1.0)
        logits = _nan_to_num(logits.float())
        prob = _nan_to_num(prob.float()).clamp(0.0, 1.0)

        l_align = self._weighted_soft_bce(logits, support_prior)
        l_bg = self._background_suppression(prob, support_prior)
        l_rank = self._support_ranking(prob, support_prior)

        total = _nan_to_num(
            self.lambda_align * l_align
            + self.lambda_bg * l_bg
            + self.lambda_rank * l_rank
        )
        details = {
            "align": _nan_to_num(l_align.detach()),
            "bg": _nan_to_num(l_bg.detach()),
            "rank": _nan_to_num(l_rank.detach()),
        }
        return total, details

    def _fuse_teacher_prior(self, teacher_features: List[torch.Tensor], out_size: Tuple[int, int]) -> torch.Tensor:
        if len(teacher_features) != len(self.teacher_fuse_weights):
            raise ValueError(
                f"teacher_features length {len(teacher_features)} != teacher_fuse_weights length {len(self.teacher_fuse_weights)}"
            )
        weight_sum = 0.0
        fused = None
        for feat, w in zip(teacher_features, self.teacher_fuse_weights):
            prior = self.teacher_feature_to_prior(feat)
            prior = self._resize(prior, out_size).clamp(0.0, 1.0)
            w = float(w)
            weight_sum += w
            fused = prior * w if fused is None else fused + prior * w
        fused = fused / max(weight_sum, self.eps)
        return self._renorm01(fused)

    def _build_student_support_target(self, prior_probs: List[torch.Tensor], out_size: Tuple[int, int]) -> torch.Tensor:
        if len(prior_probs) == 0:
            raise ValueError("prior_probs must be non-empty")
        fused = None
        for p in prior_probs:
            p = _nan_to_num(p.detach().float()).clamp(0.0, 1.0)
            p = self._resize(p, out_size)
            fused = p if fused is None else fused + p
        fused = fused / max(len(prior_probs), 1)
        fused = self._renorm01(fused)
        return self._smooth_map(fused)

    def _build_proxy_target(
        self,
        teacher_features: List[torch.Tensor],
        prior_probs: List[torch.Tensor],
        out_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        teacher_support = self._fuse_teacher_prior(teacher_features, out_size)
        student_support = self._build_student_support_target(prior_probs, out_size)
        target = (
            self.objectness_student_blend * student_support
            + (1.0 - self.objectness_student_blend) * teacher_support
        )
        target = self._renorm01(target)
        target = self._smooth_map(target)
        return target, teacher_support, student_support

    def _bridge_consistency(self, proxy_logits: List[torch.Tensor], detached_support_target: torch.Tensor) -> torch.Tensor:
        if len(proxy_logits) == 0:
            return torch.tensor(0.0, device=detached_support_target.device)
        losses = []
        for logit in proxy_logits:
            tgt = self._resize(detached_support_target, logit.shape[-2:]).clamp(0.0, 1.0)
            losses.append(self._weighted_soft_bce(logit, tgt))
        return _nan_to_num(torch.stack(losses).mean())

    def _extract_yolo_feature_maps(self, yolo_preds: Any) -> List[torch.Tensor]:
        """Return YOLO Detect training maps [B, no, H, W] across Ultralytics versions.

        Ultralytics may wrap Detect outputs in tuples/lists. In some versions,
        the outer object contains both inference tensors and raw multi-scale
        Detect maps. We recursively search for groups of 4D tensors and prefer
        the group that looks like multi-scale Detect maps.
        """
        groups: List[List[torch.Tensor]] = []

        def visit(obj: Any):
            if torch.is_tensor(obj):
                return
            if isinstance(obj, (list, tuple)):
                tensors4 = [x for x in obj if torch.is_tensor(x) and x.ndim == 4]
                if tensors4:
                    groups.append(tensors4)
                for x in obj:
                    visit(x)
            elif isinstance(obj, dict):
                for x in obj.values():
                    visit(x)

        visit(yolo_preds)
        if not groups:
            return []

        groups = sorted(groups, key=lambda g: (len(g), max(int(t.shape[1]) for t in g)), reverse=True)
        return [x for x in groups[0]]

    def _score_maps_from_yolo_preds(self, yolo_preds: Any, num_classes: int | None) -> List[torch.Tensor]:
        feats = self._extract_yolo_feature_maps(yolo_preds)
        if not feats or num_classes is None or int(num_classes) <= 0:
            return []
        nc = int(num_classes)
        score_maps: List[torch.Tensor] = []
        for feat in feats:
            if not torch.is_tensor(feat) or feat.ndim != 4 or feat.shape[1] <= nc:
                continue
            cls_logits = _nan_to_num(feat[:, -nc:, :, :].float())
            cls_score = torch.sigmoid(cls_logits).amax(dim=1, keepdim=True)
            score_maps.append(_nan_to_num(cls_score).clamp(0.0, 1.0))
        return score_maps

    def _build_gt_mask_from_batch(
        self,
        batch: Dict[str, torch.Tensor],
        out_size: Tuple[int, int],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        h, w = int(out_size[0]), int(out_size[1])
        mask = torch.zeros((batch_size, 1, h, w), device=device, dtype=torch.float32)
        if not isinstance(batch, dict) or "bboxes" not in batch or "batch_idx" not in batch:
            return mask
        bboxes = batch.get("bboxes")
        batch_idx = batch.get("batch_idx")
        if bboxes is None or batch_idx is None or bboxes.numel() == 0:
            return mask
        bboxes_cpu = bboxes.detach().float().cpu()
        batch_idx_cpu = batch_idx.detach().long().view(-1).cpu()
        expand = float(self.score_gt_expand_ratio)
        for bi, box in zip(batch_idx_cpu.tolist(), bboxes_cpu):
            if bi < 0 or bi >= batch_size or box.numel() < 4:
                continue
            xc, yc, bw, bh = [float(v) for v in box[:4].tolist()]
            bw *= expand
            bh *= expand
            x1 = max(0, int((xc - bw / 2.0) * w))
            y1 = max(0, int((yc - bh / 2.0) * h))
            x2 = min(w, int((xc + bw / 2.0) * w) + 1)
            y2 = min(h, int((yc + bh / 2.0) * h) + 1)
            if x2 > x1 and y2 > y1:
                mask[bi, 0, y1:y2, x1:x2] = 1.0
        return mask

    def _score_level_single_loss(
        self,
        score_map: torch.Tensor,
        teacher_prior: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        score_map = _nan_to_num(score_map.float()).clamp(0.0, 1.0)
        teacher_prior = self._resize(teacher_prior.detach().float(), score_map.shape[-2:]).clamp(0.0, 1.0)
        gt_mask = self._resize(gt_mask.detach().float(), score_map.shape[-2:]).clamp(0.0, 1.0)
        gt_mask = (gt_mask > 0.5).to(dtype=score_map.dtype)

        b, _, h, w = teacher_prior.shape
        flat = _nan_to_num(teacher_prior.flatten(2))
        fg_thresh = _nan_to_num(torch.quantile(flat, self.score_fg_quantile, dim=2, keepdim=True))
        bg_thresh = _nan_to_num(torch.quantile(flat, self.score_bg_quantile, dim=2, keepdim=True))
        teacher_fg = (flat >= fg_thresh).to(dtype=score_map.dtype).view(b, 1, h, w)
        teacher_bg = (flat <= bg_thresh).to(dtype=score_map.dtype).view(b, 1, h, w)

        if self.score_use_gt_pos:
            pos_mask = gt_mask
            if self.score_use_teacher_fg_pos:
                pos_mask = torch.clamp(pos_mask + teacher_fg, 0.0, 1.0)
        else:
            pos_mask = teacher_fg
        neg_mask = teacher_bg * (1.0 - gt_mask)

        pos_count = pos_mask.sum(dim=(-2, -1), keepdim=True)
        neg_count = neg_mask.sum(dim=(-2, -1), keepdim=True)
        pos_denom = pos_count.clamp_min(1.0)
        neg_denom = neg_count.clamp_min(1.0)
        pos_mean = (score_map * pos_mask).sum(dim=(-2, -1), keepdim=True) / pos_denom
        neg_mean = (score_map * neg_mask).sum(dim=(-2, -1), keepdim=True) / neg_denom
        has_pos = (pos_count > 0).to(score_map.dtype)
        has_neg = (neg_count > 0).to(score_map.dtype)
        valid = has_pos * has_neg

        rank_per_img = F.softplus(float(self.score_rank_margin) - (pos_mean - neg_mean))
        rank = (rank_per_img * valid).sum() / valid.sum().clamp_min(1.0)
        bg_per_img = ((score_map.square() * neg_mask).sum(dim=(-2, -1), keepdim=True) / neg_denom)
        bg = (bg_per_img * has_neg).sum() / has_neg.sum().clamp_min(1.0)
        total = _nan_to_num(self.lambda_score_rank * rank + self.lambda_score_bg * bg)
        return total, {
            "score_rank": _nan_to_num(rank.detach()),
            "score_bg": _nan_to_num(bg.detach()),
            "score_pos_mean": _nan_to_num(pos_mean.detach()).mean(),
            "score_neg_mean": _nan_to_num(neg_mean.detach()).mean(),
            "score_valid_ratio": _nan_to_num(valid.detach()).mean(),
        }

    def score_level_regularization(
        self,
        yolo_preds: Any,
        batch: Dict[str, torch.Tensor],
        teacher_features: List[torch.Tensor],
        num_classes: int | None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = teacher_features[0].device if teacher_features else torch.device("cpu")
        z = torch.tensor(0.0, device=device)
        empty = {
            "score_total": z.detach(),
            "score_rank": z.detach(),
            "score_bg": z.detach(),
            "score_pos_mean": z.detach(),
            "score_neg_mean": z.detach(),
            "score_valid_ratio": z.detach(),
            "score_num_maps": z.detach(),
        }
        if not self.score_regularization_enabled:
            return z, empty
        score_maps = self._score_maps_from_yolo_preds(yolo_preds, num_classes)
        if not score_maps:
            return z, empty
        weights = self.score_rank_layer_weights or [1.0 for _ in score_maps]
        if len(weights) != len(score_maps):
            raise ValueError(
                f"score_rank_layer_weights length {len(weights)} must match number of score maps {len(score_maps)}"
            )
        per_layer = []
        rank_items = []
        bg_items = []
        pos_items = []
        neg_items = []
        valid_items = []
        weight_sum = 0.0
        batch_size = int(score_maps[0].shape[0])
        for score_map, w in zip(score_maps, weights):
            w = float(w)
            if w == 0.0:
                continue
            teacher_prior = self._fuse_teacher_prior(teacher_features, score_map.shape[-2:]).detach().clamp(0.0, 1.0)
            gt_mask = self._build_gt_mask_from_batch(batch, score_map.shape[-2:], batch_size, score_map.device)
            loss, det = self._score_level_single_loss(score_map, teacher_prior, gt_mask)
            per_layer.append(_nan_to_num(loss * w))
            rank_items.append(det["score_rank"])
            bg_items.append(det["score_bg"])
            pos_items.append(det["score_pos_mean"])
            neg_items.append(det["score_neg_mean"])
            valid_items.append(det["score_valid_ratio"])
            weight_sum += w
        if not per_layer:
            return z, empty
        total = _nan_to_num(torch.stack(per_layer).sum() / max(weight_sum, self.eps))
        return total, {
            "score_total": total.detach(),
            "score_rank": _nan_to_num(torch.stack(rank_items).mean().detach()),
            "score_bg": _nan_to_num(torch.stack(bg_items).mean().detach()),
            "score_pos_mean": _nan_to_num(torch.stack(pos_items).mean().detach()),
            "score_neg_mean": _nan_to_num(torch.stack(neg_items).mean().detach()),
            "score_valid_ratio": _nan_to_num(torch.stack(valid_items).mean().detach()),
            "score_num_maps": torch.tensor(float(len(per_layer)), device=total.device),
        }

    def forward(
        self,
        prior_logits: List[torch.Tensor],
        prior_probs: List[torch.Tensor],
        proxy_logits: List[torch.Tensor],
        proxy_probs: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if len(prior_logits) != len(self.prior_layer_weights) or len(prior_probs) != len(self.prior_layer_weights):
            raise ValueError("prior branch lengths do not match prior_layer_weights")
        if len(proxy_logits) != len(self.proxy_layer_weights) or len(proxy_probs) != len(self.proxy_layer_weights):
            raise ValueError("proxy branch lengths do not match proxy_layer_weights")

        prior_per_layer = []
        prior_rank_per_layer = []
        prior_align_per_layer = []
        for w, logit, prob, t_feat in zip(self.prior_layer_weights, prior_logits, prior_probs, teacher_features):
            target_prior = self.teacher_feature_to_prior(t_feat)
            loss, item_details = self._single_support_loss(logit, prob, target_prior)
            prior_per_layer.append(_nan_to_num(loss * float(w)))
            prior_rank_per_layer.append(item_details["rank"])
            prior_align_per_layer.append(item_details["align"])
        prior_total = _nan_to_num(torch.stack(prior_per_layer).sum()) if prior_per_layer else torch.tensor(0.0)

        if proxy_logits:
            proxy_target, teacher_support, student_support = self._build_proxy_target(
                teacher_features,
                prior_probs,
                proxy_logits[0].shape[-2:],
            )
            proxy_per_layer = []
            proxy_rank_per_layer = []
            proxy_align_per_layer = []
            for w, logit, prob in zip(self.proxy_layer_weights, proxy_logits, proxy_probs):
                loss, item_details = self._single_support_loss(logit, prob, proxy_target)
                proxy_per_layer.append(_nan_to_num(loss * float(w)))
                proxy_rank_per_layer.append(item_details["rank"])
                proxy_align_per_layer.append(item_details["align"])
            proxy_total = _nan_to_num(torch.stack(proxy_per_layer).sum())
            bridge_total = (
                _nan_to_num(self._bridge_consistency(proxy_logits, student_support))
                if self.lambda_bridge > 0
                else torch.tensor(0.0, device=prior_total.device)
            )
        else:
            proxy_target = torch.zeros(1, device=prior_total.device)
            teacher_support = torch.zeros(1, device=prior_total.device)
            student_support = torch.zeros(1, device=prior_total.device)
            proxy_per_layer = []
            proxy_rank_per_layer = []
            proxy_align_per_layer = []
            proxy_total = torch.tensor(0.0, device=prior_total.device)
            bridge_total = torch.tensor(0.0, device=prior_total.device)

        teacher_anchor_total, teacher_anchor_details = self._teacher_anchor_loss(proxy_probs, teacher_features)

        total = _nan_to_num(
            self.lambda_prior * prior_total
            + self.lambda_proxy * proxy_total
            + self.lambda_bridge * bridge_total
            + teacher_anchor_total
        )

        details = {
            "prior_total": _nan_to_num(prior_total.detach()),
            "proxy_total": _nan_to_num(proxy_total.detach()),
            "bridge_total": _nan_to_num(bridge_total.detach()),
            "prior_rank_mean": _nan_to_num(torch.stack(prior_rank_per_layer).mean().detach()) if prior_rank_per_layer else torch.tensor(0.0, device=prior_total.device),
            "proxy_rank_mean": _nan_to_num(torch.stack(proxy_rank_per_layer).mean().detach()) if proxy_rank_per_layer else torch.tensor(0.0, device=prior_total.device),
            "prior_align_mean": _nan_to_num(torch.stack(prior_align_per_layer).mean().detach()) if prior_align_per_layer else torch.tensor(0.0, device=prior_total.device),
            "proxy_align_mean": _nan_to_num(torch.stack(proxy_align_per_layer).mean().detach()) if proxy_align_per_layer else torch.tensor(0.0, device=prior_total.device),
            "prior_per_layer": _nan_to_num(torch.stack(prior_per_layer).detach()) if prior_per_layer else torch.zeros(0, device=prior_total.device),
            "proxy_per_layer": _nan_to_num(torch.stack(proxy_per_layer).detach()) if proxy_per_layer else torch.zeros(0, device=prior_total.device),
            "teacher_anchor_total": _nan_to_num(teacher_anchor_total.detach()),
            "teacher_anchor_rank": _nan_to_num(teacher_anchor_details.get("teacher_rank", torch.tensor(0.0, device=prior_total.device)).detach()),
            "teacher_anchor_bg": _nan_to_num(teacher_anchor_details.get("teacher_bg", torch.tensor(0.0, device=prior_total.device)).detach()),
            "teacher_anchor_per_layer": _nan_to_num(teacher_anchor_details.get("teacher_anchor_per_layer", torch.zeros(0, device=prior_total.device)).detach()),
            "proxy_target_mean": _nan_to_num(proxy_target.detach()).mean(),
            "teacher_support_mean": _nan_to_num(teacher_support.detach()).mean(),
            "student_support_mean": _nan_to_num(student_support.detach()).mean(),
        }
        return total, details


# Backward-compatible alias for old imports, if any.
LeafPriorObjectnessTransferLoss = LegacyLeafPriorObjectnessTransferLoss


def _resolve_variant_name(lpot_cfg: dict) -> str:
    implementation = str(lpot_cfg.get("implementation", "")).strip()
    route_variant = str(lpot_cfg.get("route_variant", "")).strip()
    return implementation or route_variant or "proposal_support_prior_v1"


def build_lpot_loss_from_cfg(lpot_cfg: dict) -> tuple[str, nn.Module]:
    variant = _resolve_variant_name(lpot_cfg)

    common_kwargs = dict(
        prior_layer_weights=list(lpot_cfg.get("layer_weights", [1.0, 1.0, 1.0])),
        proxy_layer_weights=list(lpot_cfg.get("proxy_layer_weights", [1.0, 1.0, 1.0])),
        teacher_fuse_weights=list(lpot_cfg.get("teacher_fuse_weights", lpot_cfg.get("layer_weights", [1.0, 1.0, 1.0]))),
        gamma=float(lpot_cfg.get("gamma", 1.0)),
        bg_quantile=float(lpot_cfg.get("bg_quantile", 0.3)),
        lambda_bg=float(lpot_cfg.get("lambda_bg", 0.25)),
        lambda_prior=float(lpot_cfg.get("lambda_prior", 1.0)),
        lambda_proxy=float(lpot_cfg.get("lambda_proxy", 0.5)),
        lambda_bridge=float(lpot_cfg.get("lambda_bridge", 0.4)),
        objectness_student_blend=float(lpot_cfg.get("objectness_student_blend", 0.60)),
        objectness_smooth_kernel=int(lpot_cfg.get("objectness_smooth_kernel", 7)),
        objectness_smooth_iters=int(lpot_cfg.get("objectness_smooth_iters", 2)),
    )

    if variant in {
        "proposal_support_prior_v1",
        "psp",
        "psp_v1",
        "lpotv4_psp_light",
        "teacher_anchored_proxy_support_v1",
        "lpotv5_teacher_anchor",
        "score_level_proposal_support_v1",
        "lpotv41_score_level",
    }:
        canonical_name = (
            "teacher_anchored_proxy_support_v1"
            if variant in {"teacher_anchored_proxy_support_v1", "lpotv5_teacher_anchor"}
            else ("score_level_proposal_support_v1" if variant in {"score_level_proposal_support_v1", "lpotv41_score_level"} else "proposal_support_prior_v1")
        )
        return canonical_name, ProposalSupportPriorLeafPriorObjectnessTransferLoss(
            **common_kwargs,
            fg_quantile=float(lpot_cfg.get("fg_quantile", 0.7)),
            lambda_rank=float(lpot_cfg.get("lambda_rank", 0.05)),
            lambda_align=float(lpot_cfg.get("lambda_align", 0.0)),
            support_margin=float(lpot_cfg.get("support_margin", 0.10)),
            lambda_teacher_rank=float(lpot_cfg.get("lambda_teacher_rank", 0.0)),
            lambda_teacher_bg=float(lpot_cfg.get("lambda_teacher_bg", 0.0)),
            teacher_anchor_margin=(
                float(lpot_cfg["teacher_anchor_margin"])
                if lpot_cfg.get("teacher_anchor_margin", None) is not None
                else None
            ),
            teacher_fg_quantile=(
                float(lpot_cfg["teacher_fg_quantile"])
                if lpot_cfg.get("teacher_fg_quantile", None) is not None
                else None
            ),
            teacher_bg_quantile=(
                float(lpot_cfg["teacher_bg_quantile"])
                if lpot_cfg.get("teacher_bg_quantile", None) is not None
                else None
            ),
            teacher_anchor_layer_weights=(list(lpot_cfg.get("teacher_anchor_layer_weights", [])) or None),
            lambda_score_rank=float(lpot_cfg.get("lambda_score_rank", 0.0)),
            lambda_score_bg=float(lpot_cfg.get("lambda_score_bg", 0.0)),
            score_rank_margin=float(lpot_cfg.get("score_rank_margin", 0.10)),
            score_fg_quantile=float(lpot_cfg.get("score_fg_quantile", lpot_cfg.get("fg_quantile", 0.70))),
            score_bg_quantile=float(lpot_cfg.get("score_bg_quantile", lpot_cfg.get("bg_quantile", 0.30))),
            score_rank_layer_weights=(list(lpot_cfg.get("score_rank_layer_weights", [])) or None),
            score_gt_expand_ratio=float(lpot_cfg.get("score_gt_expand_ratio", 1.5)),
            score_use_gt_pos=bool(lpot_cfg.get("score_use_gt_pos", True)),
            score_use_teacher_fg_pos=bool(lpot_cfg.get("score_use_teacher_fg_pos", False)),
        )

    if variant in {"legacy_lpot_v2", "legacy", "lpot_v2_legacy"}:
        return "legacy_lpot_v2", LegacyLeafPriorObjectnessTransferLoss(**common_kwargs)

    raise ValueError(
        f"Unknown LPOT route_variant/implementation: {variant}. "
        f"Supported: proposal_support_prior_v1, teacher_anchored_proxy_support_v1, legacy_lpot_v2"
    )
