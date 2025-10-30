import torch
import torch.nn as nn
import torch.fft

class FFT(nn.Module):
    def __init__(self, c1):
        super(FFT, self).__init__()
        # 学习频域权重，shape是 (channels, 1, 1)，用于调节每个通道的频域响应
        self.freq_weight = nn.Parameter(torch.ones(c1, 1, 1))

    def forward(self, x):
        # 对每个通道的特征图做FFT，得到频域表示 (复数张量)
        x_fft = torch.fft.fft2(x, norm='ortho')  # [B, C, H, W], complex tensor

        # 将频谱乘以学习的权重，广播至每个频率点
        # 这里用freq_weight作为实数权重乘频谱的幅值部分，也可以设计更复杂的机制
        x_fft = x_fft * self.freq_weight.unsqueeze(0)  # 广播到B

        # 逆FFT还原
        x_ifft = torch.fft.ifft2(x_fft, norm='ortho').real  # 只取实部

        # 残差连接，融合频域调整后的特征
        out = x + x_ifft
        return out
