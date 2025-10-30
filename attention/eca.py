import torch
import torch.nn as nn
import torch.nn.functional as F

class ECA(nn.Module):
    def __init__(self, c1, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x) # [B, C, 1, 1]
        y = y.squeeze(-1).permute(0, 2, 1) # [B, C, 1] -> [B, 1, C]
        y = self.cv1(y) # [B, 1, C]
        y = self.sigmoid(y).permute(0, 2, 1).unsqueeze(-1) # [B, 1, C] -> [B, C, 1, 1]
        return x * y.expand_as(x) # [B, C, H, W] × [B, C, 1, 1] → [B, C, H, W]
