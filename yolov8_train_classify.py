import argparse
import os.path as osp
from pathlib import Path
import torch
import torch.nn as nn
torch.backends.cudnn.enabled=False
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
import os
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'

from yolov8_utils import make_abs_path, release_memory
from ultralytics import YOLO
from attention import C2f, C2fWithAttention
from attention import SelfClassificationTrainer, config_model, adapt_label_names
from attention import SE, ECA

def load_model(model_path, task="classify", pretrain_path=None, modify_model=False):
    _model = YOLO(model=model_path, task=task, verbose=True)
    # Transferred 228/230 items from pretrained weights
    if pretrain_path and osp.exists(pretrain_path):
        _model.load(weights=pretrain_path)
    # 是否需要config_model, 调整模型，更新参数权重
    # C2f -> C2fWithAttention
    # model.names = sorted(["algal_leaf_spot", "no_disease", "leaf_blight", "leaf_spot"])
    # ***注意这个顺序与detect模型不一样***
    if modify_model:
        _model.model = adapt_label_names(
            _model=config_model(_model.model), 
            class_names=sorted(["algal_leaf_spot", "no_disease", "leaf_blight", "leaf_spot"])
        )
    # 返回模型
    return _model

def get_backbone(_model):
    # 获取backbone
    BACKBONE_INDEX = 9
    backbone = _model.model.model[:BACKBONE_INDEX]
    layer_indices = [8]
    layers_dims = []
    for layer_indice in layer_indices:
        layer = backbone[layer_indice]
        if not (isinstance(layer, C2f) or isinstance(layer, C2fWithAttention)):
            raise RuntimeError("模型加载错误，classify模型不正确，请检查")
        layers_dims.append(
            layer.out_channels if isinstance(layer, C2fWithAttention) else layer.cv2.conv.out_channels
        )
    return backbone, layers_dims, layer_indices

def get_attention(in_channels):
    # 获取attention
    attn_module = nn.Sequential(
        ECA(c1=in_channels),
    )
    return attn_module

if __name__ == '__main__':
    """
    指标评估：https://cloud.tencent.com/developer/article/1624811
    """
    parser = argparse.ArgumentParser(description="训练YOLOv8分类模型")
    parser.add_argument("--task", type=str, default="classify", choices=["classify"], help="训练任务类型")
    parser.add_argument("--proj_dims", type=int, default=256, help="投影头特征维度")
    parser.add_argument("--config", type=str, default="yolov8m-cls.yaml", help="YAML配置")
    parser.add_argument("--pretrain", type=str, default="yolov8m-cls.pt", help="预训练权重")
    parser.add_argument("--clr_pretrain", type=str, default="best_pretrain.pt", help="clr预训练模型")
    parser.add_argument("--dir_suffix", type=str, default="", help="预训练权重目录后缀")
    parser.add_argument("--data_path", type=str, default="data.yaml", help="训练数据配置")
    parser.add_argument("--epochs", type=int, default=300, help="训练epochs次数")
    parser.add_argument("--patience", type=int, default=25, help="early stopping策略")
    parser.add_argument("--image_size", type=int, default=640, help="图片尺寸")
    parser.add_argument("--batch_size", type=int, default=32, help="训练batch大小")
    parser.add_argument("--visualizer", action='store_true', help="是否输出backbone可视化结果")
    parser.add_argument("--predict", action='store_true', help="是否输出predict&val结果")
    args = parser.parse_args()
    # 获取参数开始训练
    kwargs = {
        "device": "cuda",
        # 训练迭代参数
        "epochs": args.epochs,
        # "patience": args.patience,
        "imgsz": args.image_size, 
        "batch":args.batch_size,
        "workers": min(os.cpu_count() // 2, 8),
        # 超参数设置
        "lr0": 0.005, "lrf": 0.01, "momentum": 0.937, "weight_decay": 0.0005,
        "cos_lr": True, "warmup_epochs": 3,
        # 数据增强设置
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "flipud": 0, "fliplr": 0.5, "scale": 0.5, "translate": 0.1,
        "mixup": 0.2, "copy_paste": 0.2, # 针对病斑
        "label_smoothing": 0.0, "cache": True, 
        # 训练稳定性
        "close_mosaic": int(0.1*args.epochs),
        # 其他参数
        "save": True,
        "exist_ok": True,
        "cache": "disk"
    }
    dir_suffix = f"_{args.dir_suffix}" if args.dir_suffix and args.dir_suffix != "" else ""
    model_path = Path(make_abs_path("models")).joinpath(args.config)
    pretrain_path = Path(make_abs_path("pretrains")).joinpath(args.task).joinpath(args.pretrain)
    clr_pretrain_path = Path(make_abs_path("runs")).joinpath(args.task).joinpath(f"clr_train{dir_suffix}").joinpath("weights").joinpath(args.clr_pretrain)
    _model = load_model(
        model_path=model_path, task=args.task, 
        pretrain_path=None if osp.exists(clr_pretrain_path) else pretrain_path,
        modify_model=True if osp.exists(clr_pretrain_path) else False
    )
    if osp.exists(clr_pretrain_path): _model.load(clr_pretrain_path)
    print(f"模型加载完毕，已加载{clr_pretrain_path if osp.exists(clr_pretrain_path) else pretrain_path}")
    # 进行训练
    data_path = Path(make_abs_path("datasets")).joinpath(args.task).joinpath(args.data_path)
    data_root_dir = data_path.parent
    result = _model.train(
        task=args.task,
        data=data_root_dir,
        trainer=SelfClassificationTrainer,
        **kwargs
    )
    # 清理缓存
    release_memory()
    if args.visualizer:
        # 计算backbone可视化
        import sys
        import subprocess
        subprocess.run([
            sys.executable,  # 当前解释器路径
            "yolov8_backbone_visualizer.py",
            "--task", args.task,
            "--proj_dims", str(args.proj_dims),
            "--dir_suffix", args.dir_suffix,
            "--save_only", "--use_best_pretrain",
        ], shell=True, check=True)
    if args.predict:
        # 计算predict & val结果
        import sys
        import subprocess
        subprocess.run([
            sys.executable,  # 当前解释器路径
            "yolov8_predict_classify.py",
            "--dir_suffix", args.dir_suffix,
        ], shell=True, check=True)
    