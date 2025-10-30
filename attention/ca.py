import torch
import torch.nn as nn
import torch.nn.functional as F

class CA(nn.Module):
    def __init__(self, c1, reduction=4):
        super(CA, self).__init__()
        self.in_channels = c1
        self.reduction = reduction
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 保留H维度
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 保留W维度

        mip = max(8, c1 // reduction)  # 降维通道数

        self.cv1 = nn.Conv2d(c1, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU()

        self.conv_h = nn.Conv2d(mip, c1, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, c1, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 方向注意力 - h 和 w 方向分别注意
        x_h = self.pool_h(x)  # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B, C, 1, W] -> [B, C, W, 1]

        # 合并两个方向
        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H+W, 1]

        y = self.cv1(y)     # [B, C//r, H+W, 1]
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)  # 分离出 h 和 w 注意力部分
        x_w = x_w.permute(0, 1, 3, 2)             # [B, C//r, 1, W]

        a_h = self.sigmoid(self.conv_h(x_h))      # [B, C, H, 1]
        a_w = self.sigmoid(self.conv_w(x_w))      # [B, C, 1, W]

        out = identity * a_h * a_w
        return out
