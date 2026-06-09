import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPE2D(nn.Module):
    """
    2D sinusoidal positional encoding with optional scale-aware span.

    pe_span controls the spatial granularity of PE:
        pe_span = 1: original behavior, one PE point per feature-map cell.
        pe_span = 2: generate PE on H/2 x W/2 grid, then upsample.
        pe_span = 4: generate PE on H/4 x W/4 grid, then upsample.

    Recommended for YOLO-style multi-scale features:
        L1: pe_span = 4
        L2: pe_span = 2
        L3: pe_span = 1

    This avoids overly dense position injection in shallow high-resolution layers.
    """

    def __init__(self, pe_channels: int, pe_span: int = 1):
        super().__init__()
        self.pe_channels = int(pe_channels)
        self.pe_span = max(1, int(pe_span))

    def _build_pe(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        c = self.pe_channels
        if c % 4 != 0:
            raise ValueError("pe_channels must be divisible by 4")

        y_pos = torch.arange(h, device=device).float().unsqueeze(1)  # [H, 1]
        x_pos = torch.arange(w, device=device).float().unsqueeze(1)  # [W, 1]

        div_term = torch.exp(
            torch.arange(0, c // 4, device=device).float() * 
            -(math.log(10000.0) / (c // 4))
        )

        pe_y = torch.zeros(h, c // 2, device=device)
        pe_y[:, 0::2] = torch.sin(y_pos * div_term)
        pe_y[:, 1::2] = torch.cos(y_pos * div_term)

        pe_x = torch.zeros(w, c // 2, device=device)
        pe_x[:, 0::2] = torch.sin(x_pos * div_term)
        pe_x[:, 1::2] = torch.cos(x_pos * div_term)

        pe = torch.zeros(c, h, w, device=device)
        pe[: c // 2] = pe_y.T.unsqueeze(2).repeat(1, 1, w)
        pe[c // 2 :] = pe_x.T.unsqueeze(1).repeat(1, h, 1)

        return pe.unsqueeze(0)  # [1, C, H, W]

    def forward(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """
        Return:
            PE tensor with shape [1, C, H, W].

        If pe_span > 1:
            PE is first generated on a coarser grid and then upsampled to H x W.
        """
        span = max(1, int(self.pe_span))

        if span == 1:
            return self._build_pe(h, w, device)

        low_h = max(1, math.ceil(h / span))
        low_w = max(1, math.ceil(w / span))

        pe_low = self._build_pe(low_h, low_w, device)  # [1, C, low_H, low_W]

        pe = F.interpolate(
            pe_low,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        return pe


class PositionAwareModule(nn.Module):
    """
    Position-aware representation module with three components:

    1. Scale-aware PE span
       - pe_span controls the spatial granularity of positional encoding.
       - Larger pe_span means coarser PE.
       - Recommended:
           L1: pe_span = 4
           L2: pe_span = 2
           L3: pe_span = 1

    2. Scale-aware weighting
       - Each scale can have an independent learnable scale_weight.
       - It controls the strength of position refinement.

    3. Foreground-guided position refinement
       - fg_mask is downsampled to the current feature resolution.
       - Foreground regions are enhanced by a learnable gate.
    """

    def __init__(
        self,
        in_channels: int,
        pe_channels: int = 64,
        pe_span: int = 1,
        init_scale: float = 1.0,
        scale_weight_before = True,
        enable_fg_guidance: bool = True,
        fg_gate_init: float = 1.0,
    ):
        super().__init__()

        self.pe_span = max(1, int(pe_span))
        self.pe = SinusoidalPE2D(
            pe_channels=pe_channels,
            pe_span=self.pe_span,
        )

        self.pe_adapt = nn.Conv2d(
            pe_channels,
            in_channels,
            kernel_size=1,
            bias=False,
        )

        # Position injection strength.
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # Per-scale learnable weighting.
        self.scale_weight = nn.Parameter(torch.tensor(float(init_scale)))
        self.scale_weight_before = scale_weight_before

        self.bn = nn.BatchNorm2d(in_channels)
        self.dwconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.act = nn.GELU()

        # Foreground-guided gate.
        self.enable_fg_guidance = bool(enable_fg_guidance)
        if self.enable_fg_guidance:
            # TODO：分层控制pos_fg_gate_init=[0.3, 0.6, 1.0]
            self.fg_gate_strength = nn.Parameter(torch.tensor(float(fg_gate_init)))
        else:
            self.register_parameter("fg_gate_strength", None)

    def _resize_mask(self, fg_mask: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        """
        Resize foreground mask to current feature resolution.

        Args:
            fg_mask:
                [B, 1, H, W] or [B, H, W]
            target_hw:
                target feature map size, (H, W)

        Return:
            Binary mask with shape [B, 1, target_H, target_W].
        """
        if fg_mask is None:
            return None

        if fg_mask.dim() == 3:
            fg_mask = fg_mask.unsqueeze(1)

        fg_mask = fg_mask.float()

        fg_mask = F.interpolate(
            fg_mask,
            size=target_hw,
            mode="nearest",
        )

        fg_mask = (fg_mask > 0.5).float()
        return fg_mask

    def forward(self, x: torch.Tensor, fg_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:
                Feature map, [B, C, H, W]
            fg_mask:
                Foreground mask, [B, 1, H, W] or [B, H, W]

        Return:
            Position-enhanced feature map, [B, C, H, W]
        """
        b, c, h, w = x.shape

        # Scale-aware positional encoding.
        pe = self.pe(h, w, x.device).repeat(b, 1, 1, 1)
        pe = self.pe_adapt(pe)

        # Position refinement branch.
        res = x + self.alpha * pe
        res = self.bn(res)
        res = self.dwconv(res)
        res = self.act(res)

        # Scale-aware weighting.
        if self.scale_weight_before:
            scale_weight = torch.clamp(self.scale_weight, min=0.0)
            res = scale_weight * res

        # Foreground-guided gate.
        if self.enable_fg_guidance and (fg_mask is not None):
            fg_mask_ds = self._resize_mask(fg_mask, target_hw=(h, w))
            gate_strength = torch.clamp(self.fg_gate_strength, min=0.0)

            # 前景区域增强：1 + gamma * mask
            # 背景保持 1，前景区域乘上更大的系数
            gate = 1.0 + gate_strength * fg_mask_ds
            res = res * gate

        # Scale-aware weighting.
        if not self.scale_weight_before:
            scale_weight = torch.clamp(self.scale_weight, min=0.0)
            res = scale_weight * res

        # Residual add-back.
        out = x + res

        return out