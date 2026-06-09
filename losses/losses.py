"""Backward-compatible wrapper for the refactored stage-1 SSL losses."""

from .stage1_ssl_losses import compute_stage1_total_loss, compute_total_loss

__all__ = ["compute_stage1_total_loss", "compute_total_loss"]
