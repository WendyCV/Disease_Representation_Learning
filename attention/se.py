import torch
import torch.nn as nn
import torch.nn.functional as F

class SE(nn.Module):
    def __init__(self, c1, reduction=4):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze
        self.fc = nn.Sequential(                    # Excitation
            nn.Linear(c1, c1 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c1 // reduction, c1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # [B, C, 1, 1] -> [B, C]
        y = self.fc(y).view(b, c, 1, 1) # [B, C] -> [B, C, 1, 1]
        return x * y.expand_as(x) # [B, C, H, W] × [B, C, 1, 1] → [B, C, H, W]
