import torch
import torch.nn as nn
import math

class CenterNetv1(nn.Module):
    def __init__(self, backbone, num_classes=1000):
        self.inplanes = 64
        super(CenterNetv1, self).__init__()
        # backbone特征提取
        self.backbone = backbone
        backbone_out_channels = backbone.out_channels

        # 反卷积upsample
        self.upsample = nn.Sequential(
            self._make_deconv_layer(backbone_out_channels, backbone_out_channels//2),  # /16
            self._make_deconv_layer(backbone_out_channels//2, backbone_out_channels//4),  # /8
            self._make_deconv_layer(backbone_out_channels//4, backbone_out_channels//8),  # /4
        )
        upsample_out_channels = backbone_out_channels//8

        # header检测
        self.hm = self._make_header_layer(upsample_out_channels, num_classes)  # 热力图，检查object
        self.wh = self._make_header_layer(upsample_out_channels, 2)  # 宽高(w, h)，回归bbox宽高
        self.reg = self._make_header_layer(upsample_out_channels, 2)  # 偏移(x, y)，回归bbox相较于热力图网格偏移量

        # 初始化权重
        self._init_weights([self.upsample, self.hm, self.wh, self.reg])
        
    # 初始化权重
    def _init_weights(self, layers):
        for m in [layer.modules() for layer in layers]:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # 检测头
    def _make_header_layer(self, in_ch, out_ch):
        header = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, 1)
        )
        return header

    # 反卷积upsample
    def _make_deconv_layer(self, in_ch, out_ch):
        deconv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # 以上可引入DCNv2网络取代
            nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        return deconv

    # CenterNet网络
    def forward(self, x):
        # backbone
        x = self.backbone(x)
        # upsample
        x = self.upsample(x)
        # header
        return self.hm(x), self.wh(x), self.reg(x)

def certernetv1(backbone, **kwargs) -> nn.Module:
    model = CenterNetv1(backbone, **kwargs)
    return model
