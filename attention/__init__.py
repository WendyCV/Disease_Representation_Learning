all = [
    "C2f",
    "SPPF",
    "C2fWithAttention"
    # 适合检测任务
    "CBAM", "CA",
    # 适合分类任务
    "SE", "ECA",
    # 最新注意力机制
    "FFT"
]

import torch
import torch.nn as nn
from typing import Optional
from ultralytics.nn.modules.conv import CBAM
from ultralytics.nn.modules.block import C2f, SPPF
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.classify.train import ClassificationTrainer

from .se import SE
from .eca import ECA
from .ca import CA
from .fft import FFT

class C2fWithAttention(nn.Module):
    def __init__(self, c2f:C2f, attn_module: Optional[nn.Module] = None):
        super().__init__()
        self.c2f = c2f
        self.attn = attn_module if attn_module is not None else nn.Identity()

    def forward(self, x):
        x = self.c2f(x)
        return self.apply_attn(x)

    def forward_split(self, x):
        if hasattr(self.c2f, 'forward_split'):
            x = self.c2f.forward_split(x)
        else:
            x = self.c2f(x)
        return self.apply_attn(x)
    
    # 注意力机制
    def apply_attn(self, x):
        x = self.attn(x)
        return x
    
    # 以防万一attn用到
    @property
    def in_channels(self): return self.c2f.cv1.conv.in_channels

    @property
    def out_channels(self): return self.c2f.cv2.conv.out_channels

    # 代理C2f的属性(缺什么属性补充什么属性)
    @property
    def c(self): return self.c2f.c

    @property
    def cv1(self): return self.c2f.cv1

    @property
    def cv2(self): return self.c2f.cv2

    @property
    def m(self): return self.c2f.m

    @property
    def f(self): return self.c2f.f

    @property
    def i(self): return self.c2f.i

    @property
    def type(self): return self.c2f.type

class SelfClassificationTrainer(ClassificationTrainer):
    """
    调整训练时的model结构
    """
    def get_model(self, cfg=None, weights=None, verbose=True):
        _model = super().get_model(cfg, None, verbose)
        if hasattr(weights, ATTR_NAME):
            # 先修改模型
            _model = config_model(_model)
            # 再加载预训练模型
            if weights: _model.load(weights)
        else:
            # 先加载预训练模型
            if weights: _model.load(weights)
            # 再修改模型
            _model = config_model(_model)
        # 之后返回
        return _model
    
class SelfDetectionTrainer(DetectionTrainer):
    """
    调整训练时的model结构
    """
    def get_model(self, cfg=None, weights=None, verbose=True):
        _model = super().get_model(cfg, None, verbose)
        if hasattr(weights, ATTR_NAME):
            # 先修改模型
            _model = config_model(_model)
            # 再加载预训练模型
            if weights: _model.load(weights)
        else:
            # 先加载预训练模型
            if weights: _model.load(weights)
            # 再修改模型
            _model = config_model(_model)
        # 之后返回
        return _model

    # def preprocess_batch(self, batch):
    #     imgs = batch['img']
    #     print(f"[DEBUG] img range: min={imgs.min().item():.4f}, max={imgs.max().item():.4f}")
    #     return super().preprocess_batch(batch)
    
ATTR_NAME = "_has_config_c2f_"

def config_model(_model):
    # 配置过直接返回
    if hasattr(_model, ATTR_NAME): return _model
    # 将所有的C2f都增加attention模块
    for child_idx, child in enumerate(_model.model):
        # 过滤非C2f模块
        if not isinstance(child, C2f): continue
        # 调整C2f模块
        new_child = C2fWithAttention(child)
        _model.model[child_idx] = new_child
    # 设定属性
    setattr(_model, ATTR_NAME, True)
    # 返回模型
    return _model

def adapt_label_names(_model, num_classes=4, class_names=None):
    # 更新类别名称映射
    if class_names is not None:
        _model.names = class_names
    else:
        _model.names = {i: f'class_{i}' for i in range(num_classes)}
    # 返回模型
    return _model