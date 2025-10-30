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
    parser = args_parser()
    parser.add_argument("--data_path", type=str, default="data.yaml", help="图片目录")
    parser.add_argument("--train_downstream", action='store_true', help="是否继续训练下游任务")
    parser.add_argument("--predict", action='store_true', help="是否输出predict&val结果")
    args = parser.parse_args()
    # classify模型参数配置
    args.task = "classify"
    yaml_path = Path(make_abs_path("datasets")).joinpath(args.task).joinpath(args.data_path)
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    args.image_dir = yaml_path.parent.joinpath(data["train"]).resolve()
    args.config = make_abs_path("models/yolov8m-cls.yaml")
    args.pretrain = make_abs_path("pretrains/classify/yolov8m-cls.pt")
    args.output_dir = make_abs_path("runs/classify/clr_train")
    # 参数batch_size=16，image_size=640
    # clr_model_train(args, pre_train=pre_train)
    cmd_args = to_cmd_args(args, exclude=["data_path", "train_downstream", "predict"])
    result = subprocess.run([
        sys.executable,  # 当前解释器路径
        "yolov8_clr_train.py", 
    ] + cmd_args, shell=True, check=True)
    # 继续训练yolov8的下游classify任务
    if result.returncode == 0 and args.train_downstream:
        cmd_args = ["--task", args.task,
            "--proj_dims", str(args.proj_dims),
            "--config", Path(args.config).name,
            "--pretrain", Path(args.pretrain).name,
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
        subprocess.run([
            sys.executable,  # 当前解释器路径
            "yolov8_train_classify.py",
        ] + cmd_args, shell=True, check=True)
    # 训练完成
