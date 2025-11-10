import torch
# torch.backends.cudnn.enabled=False
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
import os
import sys
import subprocess
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
import yaml
from pathlib import Path

from yolov8_utils import make_abs_path, to_cmd_args
from yolov8_clr_train import SimCLRv1YOLOv8, args_parser

# 适配模型
def pre_train(clr_model:SimCLRv1YOLOv8):
    print(f"调整后模型结构:\n{clr_model}")
    return clr_model

if __name__ == '__main__':
    """
    python yolov8_clr_train_detect.py --train_downstream --visualizer --epoch 120
# ✅ (1 * max_lr)批次16: 
    --lr_factor 1.0 --batch_size 16 --lr 1e-3 --proj_dims 128 --bank_momentum 0.95 --bank_sample_size 256 --temperature 0.20 --dropout_r 0.3
# ✅ (1 * max_lr)批次32: 
    --lr_factor 1.0 --batch_size 32 --lr 2e-3 --proj_dims 128 --bank_momentum 0.97 --bank_sample_size 256 --temperature 0.175 --dropout_r 0.25
# ✅ (1 * max_lr)批次64: 
    --lr_factor 1.0 --batch_size 64 --lr 2.2e-3 --proj_dims 256 --bank_momentum 0.99 --bank_sample_size 384 --temperature 0.17 --dropout_r 0.2
# (1 * max_lr)批次64:--bank_sample_size 320/448
    2a:--patience 20 --lr_factor 1.0 --batch_size 64 --lr 2.2e-3 --proj_dims 256 --bank_momentum 0.99 --bank_sample_size 320 --temperature 0.25 --margin 0.1 --dropout_r 0.2 --trip_mode hard
    """
    parser = args_parser()
    parser.add_argument("--data_path", type=str, default="detect_durian_leaf.yaml", help="图片目录")
    parser.add_argument("--train_downstream", action='store_true', help="是否继续训练下游任务")
    parser.add_argument("--predict", action='store_true', help="是否输出predict&val结果")
    parser.add_argument("--skip_if_exist", action='store_true', help="如果训练过则跳过clr训练")
    args = parser.parse_args()
    # detect模型参数配置
    args.task = "detect"
    yaml_path = Path(make_abs_path("datasets")).joinpath(args.task).joinpath(args.data_path)
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    args.image_dir = yaml_path.joinpath(data["train"]).resolve()
    # args.config = make_abs_path("models/yolov8m.yaml")
    # args.pretrain = make_abs_path("pretrains/detect/yolov8m.pt")
    args.output_dir = make_abs_path("runs/detect/clr_train")
    # 参数batch_size=16，image_size=640
    # clr_model_train(args, pre_train=pre_train)
    # 结果目录
    if args.dir_suffix and args.dir_suffix != "":
        output_dir = Path(args.output_dir)
        new_dir = output_dir.with_name(output_dir.name + f"_{args.dir_suffix}")
    skip_training = args.skip_if_exist and os.path.exists(new_dir) and len(os.listdir(new_dir)) > 0
    # 跳过clr训练
    if not skip_training:
        cmd_args = to_cmd_args(args, exclude=["data_path", "train_downstream", "predict", "skip_if_exist"])
        result = subprocess.run([
            sys.executable,  # 当前解释器路径
            "yolov8_clr_train.py", 
        ] + cmd_args, shell=True, check=True)
    # 继续训练yolov8的下游detect任务
    if (skip_training or result.returncode == 0) and args.train_downstream:
        cmd_args = [
            "--task", args.task,
            "--config", Path(args.config).name,
            "--pretrain", Path(args.pretrain).name,
            "--yolo_version", args.yolo_version,
            "--proj_dims", str(args.proj_dims),
            "--clr_version", args.clr_version,
            "--spp_mode", args.spp_mode,
            "--clr_pretrain", "best_pretrain.pt",
            "--dir_suffix", args.dir_suffix,
            "--data_path", args.data_path,
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--image_size", str(args.image_size),
            "--batch_size", str(args.batch_size),
        ]
        if args.visualizer: cmd_args.append("--visualizer")
        if args.predict: cmd_args.append("--predict")
        if args.skip_sppf: cmd_args.append("--skip_sppf")
        subprocess.run([
            sys.executable,  # 当前解释器路径
            "yolov8_train_detect.py",
        ] + cmd_args, shell=True, check=True)
    # 训练完成
