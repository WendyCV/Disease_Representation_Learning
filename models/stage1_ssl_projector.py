import math
import torch.nn as nn


def align_pow2(c1):
    return 2 ** math.ceil(math.log2(c1))


class GlobalProjector(nn.Module):
    def __init__(self, in_features, out_features=256, dropout_r=0.1):
        super().__init__()
        hidden_dim = align_pow2(in_features)

        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(in_features, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_r),

            nn.Linear(hidden_dim, out_features, bias=False),
            nn.BatchNorm1d(out_features),
        )

    def forward(self, x):
        return self.net(x)


class LocalProjector(nn.Module):
    def __init__(self, in_channels, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_dim, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.net(x)


class MultiScaleProjector(nn.Module):
    def __init__(self, in_channels, proj_dim=256, local_dim=128, dropout_r=0.1):
        super().__init__()
        self.global_heads = nn.ModuleList([
            GlobalProjector(c, proj_dim, dropout_r=dropout_r) for c in in_channels
        ])
        self.local_heads = nn.ModuleList([
            LocalProjector(c, local_dim) for c in in_channels
        ])

    def forward(self, features):
        global_embs = []
        local_embs = []

        for feat, g_head, l_head in zip(features, self.global_heads, self.local_heads):
            global_embs.append(g_head(feat))
            local_embs.append(l_head(feat))

        return global_embs, local_embs