import argparse
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageOps
from pathlib import Path
from yolov8_utils import make_abs_path, release_memory
from yolov8_clr_train import SimCLRv1YOLOv8, SimCLRv2YOLOv8
from yolov8_model_tools import load_model, get_backbone, get_attention

class BackboneVisualizer:
    def __init__(self, model_before: nn.Module, model_after: nn.Module, device='cuda'):
        self.model_before = model_before.to(device).eval()
        self.model_after = model_after.to(device).eval()
        self.device = device
        self.feature_maps = {'before': None, 'after': None}

    def _register_hooks(self, layer_indexes):
        """Register forward hooks to target layer index"""
        if isinstance(layer_indexes, int):
            layer_indexes = [layer_indexes]
        
        self.feature_maps = {layer_index : {'before': None, 'after': None} for layer_index in layer_indexes}

        def hook_fn_before(module, input, output, layer_index):
            self.feature_maps[layer_index ]['before'] = output.detach()

        def hook_fn_after(module, input, output, layer_index):
            self.feature_maps[layer_index ]['after'] = output.detach()

        # 注册 hooks
        hooks_before = {}
        hooks_after = {}
        # 注册 hook
        for layer_index  in layer_indexes:
            layer_before = self.model_before.backbone[layer_index]
            layer_after = self.model_after.backbone[layer_index]
            hooks_before[layer_index] = layer_before.register_forward_hook(lambda m, i, o, layer_index=layer_index: hook_fn_before(m, i, o, layer_index))
            hooks_after[layer_index] = layer_after.register_forward_hook(lambda m, i, o, layer_index=layer_index: hook_fn_after(m, i, o, layer_index))
        # 返回
        return hooks_before, hooks_after
    
    @staticmethod
    def _open_upright(image_or_path):
        """读取并按 EXIF 旋正为肉眼正方向，然后去除方向影响（不再二次旋转）"""
        if isinstance(image_or_path, (str, Path)):
            img = Image.open(image_or_path)
        else:
            img = image_or_path
        # 按 EXIF 旋正像素（只做这一次），这样数组就是“肉眼正方向”
        img = ImageOps.exif_transpose(img).convert("RGB")
        # 之后 img 的像素就是正方向，不再依赖 EXIF
        return img
    
    @staticmethod
    def _pad_to_square(img, target_size=640, fill=(114, 114, 114)):
        """将非方形图像pad到640×640"""
        w, h = img.size
        dw = target_size - w
        dh = target_size - h
        pad_l, pad_t = dw // 2, dh // 2
        pad_r, pad_b = dw - pad_l, dh - pad_t
        return ImageOps.expand(img, border=(pad_l, pad_t, pad_r, pad_b), fill=fill)
    
    @staticmethod
    def _resize_keep_ratio(image, target_long=640):
        """按比例缩放图像，使长边=target_long，短边同比例缩放"""
        w, h = image.size
        scale = target_long / max(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        return image.resize((new_w, new_h), Image.BICUBIC), (new_w, new_h)

    def _preprocess_image(self, image):
        if isinstance(image, (str, Path)): 
            image = self._open_upright(image)
        # === resize (长边 640，保持比例) ===
        resized_img, (new_w, new_h) = self._resize_keep_ratio(image, target_long=640)
        resized_img = self._pad_to_square(resized_img)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_tensor = transform(resized_img).unsqueeze(0).to(self.device)
        img_np = np.array(resized_img)  # for visualization
        return img_tensor, img_np, (new_w, new_h)

    def _process_fmap(self, fmap):
        fmap = fmap[0].mean(dim=0).cpu().numpy()
        fmap = np.maximum(fmap, 0)
        fmap /= (fmap.max() + 1e-8)
        return cv2.resize(fmap, (640, 640))

    def _apply_heatmap(self, heatmap, image_np):
        # 确保 heatmap 是二维 float32，范围 [0, 1]
        heatmap = np.uint8(255 * heatmap)
        heat = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heat = heat.astype(np.float32) / 255.0
        # 如果 image_np 是 uint8，先归一化为 float32
        if image_np.dtype != np.float32:
            image_np = image_np.astype(np.float32)
        if image_np.max() > 1.0:
            image_np = image_np / 255.0
        # 确保尺寸一致
        if heat.shape[:2] != image_np.shape[:2]:
            heat = cv2.resize(heat, (image_np.shape[1], image_np.shape[0]))
        # 确保通道数一致（3通道），否则转为彩色
        if image_np.ndim == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 1:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        # 叠加热力图并返回
        overlay = cv2.addWeighted(image_np, 0.6, heat, 0.4, 0)
        return overlay
    
    def get_heatmaps(self, image, layer_indexes):
        # 预处理图像
        img_tensor, img_np, (new_w, new_h) = self._preprocess_image(image)
        # 注册 hook
        hooks_before, hooks_after = self._register_hooks(layer_indexes)
        # 前向传播，触发 hook
        with torch.no_grad():
            _ = self.model_before(img_tensor)
            _ = self.model_after(img_tensor)
        # 获取特征图
        if isinstance(layer_indexes, int):
            layer_indexes = [layer_indexes]
        # 循环
        overlays_before = {}
        overlays_after = {}
        for layer_index in layer_indexes:
            fmap_before = self.feature_maps[layer_index]['before']
            fmap_after = self.feature_maps[layer_index]['after']
            # 生成热力图
            heatmap_before = self._process_fmap(fmap_before)
            heatmap_after = self._process_fmap(fmap_after)
            # 应用热力图叠加效果
            overlay_before = self._apply_heatmap(heatmap_before, img_np)
            overlay_after = self._apply_heatmap(heatmap_after, img_np)
            # 每一层热力图
            overlays_before[layer_index] = overlay_before
            overlays_after[layer_index] = overlay_after
        # 清理 hooks
        for layer_index in layer_indexes:
            hooks_before[layer_index].remove()
            hooks_after[layer_index].remove()
        # 返回结果
        return overlays_before, overlays_after, img_np, (new_w, new_h)

    def show_heatmaps(self, overlays_before, overlays_after, titles, factor=1):
        """显示叠加的热力图图像"""
        assert len(overlays_before) == len(overlays_after), "前后两者必须相同"
        num_images = len(overlays_before)
        fig, axs = plt.subplots(num_images, 2, figsize=(5 * factor, 3 * num_images))
        if num_images == 1:
            axs = [axs]  # 保持统一格式
        for i in range(num_images):
            # 前
            axs[i][0].imshow(overlays_before[i][..., ::-1])
            if i == 0:
                axs[i][0].set_title(f"{titles[0]}")
            axs[i][0].axis("off")
            # 后
            axs[i][1].imshow(overlays_after[i][..., ::-1])
            if i == 0:
                axs[i][1].set_title(f"{titles[1]}")
            axs[i][1].axis("off")
        plt.tight_layout()
        plt.show()

    def save_heatmaps(self, overlays_before, overlays_after, save_dir, prefix):
        """保存叠加的热力图图像"""
        assert len(overlays_before) == len(overlays_after), "前后两者必须相同"
        # 拼接所有图片：按行拼接 before / after
        rows = []
        for before, after in zip(overlays_before, overlays_after):
            row = np.concatenate([self.to_uint8(before), self.to_uint8(after)], axis=1)
            rows.append(row)
        # 所有行拼接成一张图
        full_image = np.concatenate(rows, axis=0)  
        save_path = Path(save_dir) / f"{prefix}_all.jpg"
        cv2.imwrite(str(save_path), full_image)

    @staticmethod
    def to_uint8(img):
        # 将浮点图像转换为 uint8 格式保存
        return (img * 255).clip(0, 255).astype(np.uint8)
    
    def save_each(self, overlays_after_heatmaps, layer_indexes, save_dir, image_path, tag, new_w, new_h):
        """保存叠加的热力图图像"""
        img_name = Path(image_path).stem
        for layer_index in layer_indexes:
            save_path = Path(save_dir) / f"{img_name}_{layer_index}_{tag}.jpg"
            img_heatmap = self.to_uint8(overlays_after_heatmaps[layer_index])
            # -----------------------------
            # 裁剪到 new_w, new_h
            # -----------------------------
            h, w = img_heatmap.shape[:2]
            pad_left = (w - new_w) // 2
            pad_top = (h - new_h) // 2
            pad_right = pad_left + new_w
            pad_bottom = pad_top + new_h
            img_cropped = img_heatmap[pad_top:pad_bottom, pad_left:pad_right]
            # 保存
            cv2.imwrite(str(save_path), img_cropped)

    def visualize(self, image_paths, layer_indexes=4, titles=('Before SimCLRv2', 'After SimCLRv2'), save_dir="heatmaps", prefix="sample", save_only=False, save_each=True):
        # 统一为列表
        assert isinstance(image_paths, (str, Path, list, tuple)), "只支持str/Path/list/tuple格式"
        if not isinstance(image_paths, (list, tuple)): image_paths = [image_paths]
        assert isinstance(layer_indexes, (int, tuple, list)), "只支持int/list/tuple格式"
        if isinstance(layer_indexes, int): layer_indexes = [layer_indexes]
        overlays_before = []
        overlays_after = []
        for image_path in image_paths:
            overlays_before_heatmaps, overlays_after_heatmaps, _, (new_w, new_h) = self.get_heatmaps(image_path, layer_indexes)
            # 将列表中的图像按水平方向拼接成一张大图
            overlay_before = np.hstack([overlays_before_heatmaps[layer_index] for layer_index in layer_indexes])
            overlay_after = np.hstack([overlays_after_heatmaps[layer_index] for layer_index in layer_indexes])
            if save_each:
                self.save_each(overlays_before_heatmaps, layer_indexes, save_dir, image_path, "before", new_w, new_h)
                self.save_each(overlays_after_heatmaps, layer_indexes, save_dir, image_path, "after", new_w, new_h)
            # 拼接前后的热力图
            overlays_before.append(overlay_before)
            overlays_after.append(overlay_after)
        # 保存
        self.save_heatmaps(overlays_before, overlays_after, save_dir, prefix)
        # 展示
        if not save_only: self.show_heatmaps(overlays_before, overlays_after, titles, factor=len(layer_indexes))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="backbone可视化")
    parser.add_argument("--task", type=str, default="detect", choices=["classify", "detect"], help="训练模型类别")
    parser.add_argument("--skip_sppf", action='store_true', help="是否跳过SPPF层不训练")
    parser.add_argument("--yolo_version", type=str, default="v8", choices=["v8", "v9", "v10", "v11"], help="yolo版本选择")
    parser.add_argument("--config", type=str, default="yolov8m.yaml", help="YAML配置")
    parser.add_argument("--pretrain", type=str, default="yolov8m.pt", help="预训练权重")
    parser.add_argument("--proj_dims", type=int, default=256, help="投影头特征维度")
    parser.add_argument("--clr_version", type=str, default="v2", choices=["v1", "v2"], help="clr版本选择")
    parser.add_argument("--dir_suffix", type=str, default="", help="预训练权重目录后缀")
    parser.add_argument("--use_best_pretrain", action='store_true', help="是否使用训练后的pretrain")
    parser.add_argument("--save_only", action='store_true', help="只保存图片，不plt展示")
    args = parser.parse_args()
    if args.task == "classify":
        dir_suffix = f"_{args.dir_suffix}" if args.dir_suffix and args.dir_suffix != "" else ""
        visualizer_cfg = {
            "task": "classify",
            "model_path": make_abs_path("yolo_models/yolov8m-cls.yaml"),
            "pretrain": make_abs_path("yolo_pretrains/classify/yolov8m-cls.pt"),
            "best_pretrain": make_abs_path(f"runs/classify/train{dir_suffix}/weights/best.pt"),
            "clr_pretrain": make_abs_path(f"runs/classify/clr_train{dir_suffix}/weights/best_clr.pt")
        }
    elif args.task == "detect":
        dir_suffix = f"_{args.dir_suffix}" if args.dir_suffix and args.dir_suffix != "" else ""
        visualizer_cfg = {
            "task": "detect",
            "model_path": make_abs_path(f"yolo_models/{args.config}"),
            "pretrain": make_abs_path(f"yolo_pretrains/detect/{args.pretrain}"),
            "best_pretrain": make_abs_path(f"runs/detect/train{dir_suffix}/weights/best.pt"),
            "clr_pretrain": make_abs_path(f"runs/detect/clr_train{dir_suffix}/weights/best_clr.pt")
        }
    else:
        raise RuntimeError(f"不支持模型类别{args.task}")
    assert args.clr_version in ["v1", "v2"], f"不支持版本{args.clr_version}"
    clr_module_cls = SimCLRv2YOLOv8 if args.clr_version == "v2" else SimCLRv1YOLOv8
    kwargs = {
        "skip_sppf": args.skip_sppf,
        "yolo_version": args.yolo_version,
    }
    # 开始可视化 before
    use_best_pretrain = (args.use_best_pretrain and osp.exists(visualizer_cfg["best_pretrain"]))
    pretrain_path = visualizer_cfg["best_pretrain"] if use_best_pretrain else visualizer_cfg["pretrain"]
    _YOLOv8_model_before_ = load_model(
        model_path=visualizer_cfg["model_path"], task=visualizer_cfg["task"],
        pretrain_path=None if use_best_pretrain else pretrain_path, modify_model=True
    )
    if use_best_pretrain: _YOLOv8_model_before_.load(pretrain_path)
    _YOLOv8_backbone_before_, _YOLOv8_out_channels_before_, _YOLOv8_layer_indices_before_ = get_backbone(_YOLOv8_model_before_, task=visualizer_cfg["task"], **kwargs)
    print(f"[BEFORE]: dims={_YOLOv8_out_channels_before_}, layers={_YOLOv8_layer_indices_before_}")
    model_before = clr_module_cls(_YOLOv8_backbone_before_, _YOLOv8_out_channels_before_, _YOLOv8_layer_indices_before_, augmentation=None, projector_dim=args.proj_dims)
    from yolov8_model_tools import MultiScaleFeatureAttention
    model_before.attn_module = MultiScaleFeatureAttention(nn.ModuleList([
        get_attention(in_channels, visualizer_cfg["task"])
        for in_channels in _YOLOv8_out_channels_before_
    ]))
    # 开始可视化 after
    _YOLOv8_model_after_ = load_model(
        model_path=visualizer_cfg["model_path"], task=visualizer_cfg["task"],
        pretrain_path=None, modify_model=True
    )
    _YOLOv8_backbone_after_, _YOLOv8_out_channels_after_, _YOLOv8_layer_indices_after_ = get_backbone(_YOLOv8_model_after_, task=visualizer_cfg["task"], **kwargs)
    print(f"[AFTER]: dims={_YOLOv8_out_channels_after_}, layers={_YOLOv8_layer_indices_after_}")
    model_after = clr_module_cls(_YOLOv8_backbone_after_, _YOLOv8_out_channels_after_, _YOLOv8_layer_indices_after_, augmentation=None, projector_dim=args.proj_dims)
    model_after.attn_module = MultiScaleFeatureAttention(nn.ModuleList([
        get_attention(in_channels, visualizer_cfg["task"])
        for in_channels in _YOLOv8_out_channels_after_
    ]))
    if osp.exists(visualizer_cfg["clr_pretrain"]):
        checkpoint = torch.load(visualizer_cfg["clr_pretrain"])
        model_after.load_state_dict(checkpoint["model_state_dict"])
    # 创建可视化器
    model_before.eval()
    model_after.eval()
    if use_best_pretrain:
        visualizer = BackboneVisualizer(model_after, model_before)
    else:
        visualizer = BackboneVisualizer(model_before, model_after)
    # 调用可视化
    test_root = Path(make_abs_path("runs")).joinpath("test")
    image_exts = [".jpg", ".jpeg", ".png", ".bmp"]  # 支持的图片扩展名
    images_path = [p for p in test_root.iterdir() if p.suffix.lower() in image_exts] # 遍历 test_root 下所有图片文件
    # 查找最后一层 C2f / C2fWithAttention 层，若无则默认最后一层
    # SPPF/SPPELAN层不适合做热力图展示
    target_layer_indexes = model_after.layer_indices
    target_layer_indexes[-1] = next(
        (i for i, l in reversed(list(enumerate(_YOLOv8_backbone_after_)))
        if l.__class__.__name__ in ('C2f', 'C2fWithAttention', 'RepNCSPELAN4', 'PSA', 'C2fCIB', 'C2PSA', 'C3k2', 'A2C2f', 'A2C2fWithAttention')),
        len(_YOLOv8_backbone_after_) - 1
    )
    # 开始获取热力图
    output_dir = Path(make_abs_path("runs")).joinpath("heatmap")
    if args.dir_suffix and args.dir_suffix != "":
        output_dir = output_dir.joinpath(args.dir_suffix)
    output_dir.mkdir(parents=True, exist_ok=True)
    titles = ('After_SimCLRv2', 'After_YOLOv8') if use_best_pretrain else ('Before_SimCLRv2', 'After_SimCLRv2')
    prefix = f"{args.task.upper()}_YOLOv8_4Test" if use_best_pretrain else f"{args.task.upper()}_SimCLRv2_4Test"
    visualizer.visualize(
        images_path, layer_indexes=target_layer_indexes,
        titles=titles, prefix=prefix,
        save_dir=output_dir, save_only=args.save_only
    )
    # 清理缓存
    release_memory()
    