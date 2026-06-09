import torch
import torch.nn as nn
from pathlib import Path

from .yolo_model import YOLO


def set_yolo_detect_nc(model, nc: int, verbose: bool = True):
    """
    先加载预训练 YOLO，再将 detect head 的类别数改为指定 nc。
    保持最小侵入，不改现有整体逻辑。
    """
    if nc is None:
        return model

    # 整体模型属性
    if hasattr(model, "nc"):
        model.nc = nc

    if hasattr(model, "yaml") and isinstance(model.yaml, dict):
        model.yaml["nc"] = nc

    # 最后一层通常是 Detect head
    if hasattr(model, "model") and len(model.model) > 0:
        detect_module = model.model[-1]
        if hasattr(detect_module, "nc"):
            detect_module.nc = nc

    if verbose:
        print(f"[Stage1] Detect head nc reset to {nc} after pretrained loading.")

    return model


class MultiScaleYOLOv8Backbone(nn.Module):
    def __init__(self, model_path="yolov8n.pt", layer_indices=(4, 6, 8), sppf_indice=9, nc=None, verbose=True):
        super().__init__()

        self.model_path = Path(str(model_path)).resolve()
        self.backbone_name = self.model_path.stem
        self.verbose = verbose
        
        yolo = YOLO(model_path)

        # 保持现有逻辑：先加载预训练，再改 nc
        yolo.model = set_yolo_detect_nc(yolo.model, nc=nc, verbose=verbose)

        # 只保留真正的 torch 模块
        self.model = yolo.model.model

        self.layer_indices = sorted(list(layer_indices))
        self.sppf_indice = sppf_indice
        self.out_dims = None
        self.nc = nc

        if verbose:
            self._print_backbone_info()

    def _print_backbone_info(self):
        """Print the actually loaded YOLO backbone information for experiment logs."""
        num_layers = len(self.model) if hasattr(self.model, "__len__") else -1
        max_layer = max(self.layer_indices) if self.layer_indices else None
        print("=" * 80)
        print("[Stage1] YOLO backbone loaded")
        print(f"[Stage1]  model_path    : {self.model_path}")
        print(f"[Stage1]  backbone_name : {self.backbone_name}")
        print(f"[Stage1]  layer_indices : {self.layer_indices}")
        print(f"[Stage1]  max_layer     : {max_layer}")
        print(f"[Stage1]  num_modules   : {num_layers}")
        print(f"[Stage1]  nc            : {self.nc}")
        print("=" * 80)

    def forward(self, x):
        sppf_feat = None
        features = []
        outputs = []

        for idx, mod in enumerate(self.model):
            if hasattr(mod, "f") and mod.f != -1:
                if isinstance(mod.f, int):
                    x_in = outputs[mod.f]
                else:
                    x_in = [x if j == -1 else outputs[j] for j in mod.f]
            else:
                x_in = x

            x = mod(x_in)
            outputs.append(x)

            if idx in self.layer_indices:
                feat = x[-1] if isinstance(x, (list, tuple)) else x
                features.append(feat)

            if idx == self.sppf_indice:
                feat = x[-1] if isinstance(x, (list, tuple)) else x
                sppf_feat = feat

            if idx >= max(self.layer_indices) and idx > self.sppf_indice:
                break

        return features, sppf_feat

    @torch.no_grad()
    def infer_out_dims(self, image_size=640, device="cpu"):
        # 只让内部 torch 模块进入 eval
        self.model.eval()

        dummy = torch.randn(1, 3, image_size, image_size, device=device)
        feats, _ = self.forward(dummy)
        self.out_dims = [f.shape[1] for f in feats]
        return self.out_dims