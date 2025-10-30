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
from attention import C2f, C2fWithAttention, SPPF
from attention import SelfDetectionTrainer, config_model, adapt_label_names
from attention import CBAM, CA

def load_model(model_path, task="detect", pretrain_path=None, modify_model=False):
    _model = YOLO(model=model_path, task=task, verbose=True)
    # Transferred 469/475 items from pretrained weights
    if pretrain_path is not None and osp.exists(pretrain_path):
        _model.load(weights=pretrain_path)
    # 调整模型，更新参数权重
    # C2f -> C2fWithAttention
    # model.names = ["algal_leaf_spot", "no_disease", "leaf_blight", "leaf_spot"]
    # ***注意这个顺序与classify模型不一样***
    if modify_model:
        _model.model = adapt_label_names(
            _model=config_model(_model.model), 
            class_names=["algal_leaf_spot", "no_disease", "leaf_blight", "leaf_spot"]
        )
    # 返回模型
    return _model

def get_backbone(_model, skip_sppf=False):
    # 获取backbone
    BACKBONE_INDEX = 9 if skip_sppf else 10
    backbone = _model.model.model[:BACKBONE_INDEX]
    layer_indices = [4, 6, BACKBONE_INDEX-1]
    layers_dims = []
    for layer_indice in layer_indices:
        layer = backbone[layer_indice]
        if not (isinstance(layer, SPPF) or isinstance(layer, C2f) or isinstance(layer, C2fWithAttention)):
            raise RuntimeError("模型加载错误，detect模型不正确，请检查")
        layers_dims.append(
            layer.out_channels if isinstance(layer, C2fWithAttention) else layer.cv2.conv.out_channels
        )
    return backbone, layers_dims, layer_indices

def get_attention(in_channels):
    # 获取attention
    attn_module = nn.Sequential(
        CBAM(c1=in_channels),
    )
    return attn_module

if __name__ == '__main__':
    """
    指标评估：https://cloud.tencent.com/developer/article/1624811
    """
    parser = argparse.ArgumentParser(description="训练YOLOv8分类模型")
    parser.add_argument("--task", type=str, default="detect", choices=["detect"], help="训练任务类型")
    parser.add_argument("--proj_dims", type=int, default=256, help="投影头特征维度")
    parser.add_argument("--clr_version", type=str, default="v2", choices=["v1", "v2"], help="clr版本选择")
    parser.add_argument("--config", type=str, default="yolov8m.yaml", help="YAML配置")
    parser.add_argument("--pretrain", type=str, default="yolov8m.pt", help="预训练权重")
    parser.add_argument("--clr_pretrain", type=str, default="best_pretrain.pt", help="clr预训练模型")
    parser.add_argument("--dir_suffix", type=str, default="", help="预训练权重目录后缀")
    parser.add_argument("--data_path", type=str, default="detect_durian_leaf.yaml", help="训练数据配置")
    parser.add_argument("--epochs", type=int, default=120, help="训练epochs次数")
    parser.add_argument("--patience", type=int, default=20, help="early stopping策略")
    parser.add_argument("--image_size", type=int, default=640, help="图片尺寸")
    parser.add_argument("--batch_size", type=int, default=64, help="训练batch大小")
    parser.add_argument("--visualizer", action='store_true', help="是否输出backbone可视化结果")
    parser.add_argument("--predict", action='store_true', help="是否输出predict&val结果")
    args = parser.parse_args()
    print("[RUN-args]:", args)
    # 获取参数开始训练
    kwargs = { 
        "device": "cuda",
        "epochs": args.epochs,
        # "patience": args.patience,
        "imgsz": args.image_size, 
        "batch": args.batch_size,
        "workers": min(os.cpu_count() // 2, 8),
        "save": True, "exist_ok": True, "cache": "disk",
        "freeze": 0, "seed": 42,
        # 训练参数
        "cls": 0.70, "box": 1.45, "dfl": 1.25,
        "mosaic": 0.10, "copy_paste": 0.05, "translate": 0.07, "scale": 0.60,
        "close_mosaic": int(0.70 * args.epochs),
        "iou": 0.55,
    }
    # 开始训练
    dir_suffix = f"_{args.dir_suffix}" if args.dir_suffix and args.dir_suffix != "" else ""
    model_path = Path(make_abs_path("models")).joinpath(args.config)
    pretrain_path = Path(make_abs_path("pretrains")).joinpath(args.task).joinpath(args.pretrain)
    clr_pretrain_path = Path(make_abs_path("runs")).joinpath(args.task).joinpath(f"clr_train{dir_suffix}").joinpath("weights").joinpath(args.clr_pretrain)
    _model = load_model(
        model_path=model_path, task=args.task,
        pretrain_path=None if osp.exists(clr_pretrain_path) else pretrain_path,
        modify_model=True if osp.exists(clr_pretrain_path) else False
    )
    if osp.exists(clr_pretrain_path): 
        _model.load(clr_pretrain_path)
    print(f"模型加载完毕，已加载{clr_pretrain_path if osp.exists(clr_pretrain_path) else pretrain_path}")
    # 设为 channels_last；通常能省 5–10% 显存
    # _model = _model.to(memory_format=torch.channels_last)
    # 指定训练输出目录
    if args.dir_suffix and args.dir_suffix != "":
        kwargs.update({
            "name": f"train_{args.dir_suffix}",
            # "image_weights": True,
            "pretrained": False,
        })
    data_path = Path(make_abs_path("datasets")).joinpath(args.task).joinpath(args.data_path)
    # 开始训练
    result = _model.train(
        task=args.task,
        data=data_path,
        trainer=SelfDetectionTrainer,
        **kwargs
    )
    # if args.dir_suffix and args.dir_suffix != "":
    #     old_dir = Path(result.save_dir)
    #     new_dir = old_dir.with_name(old_dir.name + f"_{args.dir_suffix}")
    #     os.rename(old_dir, new_dir)
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
            "--clr_version", args.clr_version,
            "--dir_suffix", args.dir_suffix,
            "--save_only", "--use_best_pretrain",
        ], shell=True, check=True)
    if args.predict:
        # 计算predict & val结果
        import sys
        import subprocess
        subprocess.run([
            sys.executable,  # 当前解释器路径
            "yolov8_predict_detect.py",
            "--dir_suffix", args.dir_suffix,
        ], shell=True, check=True)
