"""Backward-compatible wrapper.

The project now uses :mod:`datasets.stage1_ssl_dataset` as the primary module.
This file is kept so that existing scripts continue to work unchanged.
"""

from .stage1_ssl_dataset import Stage1ContrastiveDataset, UnlabeledLeafContrastiveDataset

__all__ = ["Stage1ContrastiveDataset", "UnlabeledLeafContrastiveDataset"]
