import argparse
import os.path as osp
import torch
torch.backends.cudnn.enabled=False
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
import os
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
import yaml
from pathlib import Path

from yolov8_utils import make_abs_path
from yolov8_model_tools import load_model, get_backbone

if __name__ == '__main__':
    """
    指标评估：https://cloud.tencent.com/developer/article/1624811
    """
    parser = argparse.ArgumentParser(description="训练YOLOv8分类模型")
    parser.add_argument("--task", type=str, default="classify", choices=["classify"], help="训练任务类型")
    parser.add_argument("--config", type=str, default="yolov8m-cls.yaml", help="YAML配置")
    parser.add_argument("--pretrain", type=str, default="best.pt", help="预训练权重")
    parser.add_argument("--dir_suffix", type=str, default="", help="预训练权重目录后缀")
    parser.add_argument("--data_path", type=str, default="data.yaml", help="预测数据配置")
    parser.add_argument("--val_only", action='store_true', help="只做val测试")
    parser.add_argument("--conf", type=float, default=0.5, help="检测置信度设定")
    args = parser.parse_args()
    # 获取参数开始训练
    model_path = Path(make_abs_path("yolo_models")).joinpath(args.config)
    _model = load_model(model_path=model_path, task=args.task, modify_model=True)
    train_output_dir = f"train_{args.dir_suffix}" if args.dir_suffix and args.dir_suffix != "" else "train"
    runs_weights_path = Path(make_abs_path("runs")).joinpath(args.task).joinpath(train_output_dir).joinpath("weights").joinpath(args.pretrain)
    if osp.exists(runs_weights_path): 
        _model.load(runs_weights_path)
    print(f"模型加载完毕，已加载{runs_weights_path if osp.exists(runs_weights_path) else None}")
    predict_output_dir = f"predict_{args.dir_suffix}" if args.dir_suffix and args.dir_suffix != "" else "predict"
    save_root = Path(make_abs_path("runs")).joinpath(args.task).joinpath(predict_output_dir)
    if not args.val_only:
        kwargs = {
            "save": True,
            "save_json": True,
            "save_frames": True,
            "save_txt": True,
            "save_conf": True,
            "show_labels": True,
            "show_conf": True,
            "conf": args.conf,    # 置信度阈值
        }
        # 进行推理预测
        train_dir = Path(make_abs_path("datasets")).joinpath(args.task).joinpath("train")
        test_dirs = [
            Path(make_abs_path("datasets")).joinpath(args.task).joinpath("test")
        ]
        class_names = sorted(os.listdir(train_dir))
        for class_name in class_names:
            cls_save_dir = osp.join(save_root, class_name)
            # 多个测试目录
            for test_dir in test_dirs:
                source_dir = osp.join(test_dir, class_name)
                result = _model.predict(source=source_dir, save_dir=cls_save_dir, **kwargs)
    # //todo: 检查数据精度
    yaml_path = Path(make_abs_path("datasets")).joinpath(args.task).joinpath(args.data_path)
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    # 替换 val 为 test
    data["train"] = str(yaml_path.parent.joinpath(data["train"]).resolve())
    data["val"] = str(yaml_path.parent.joinpath(data["test"]).resolve())
    data["test"] = str(yaml_path.parent.joinpath(data["test"]).resolve())
    # 保存修改后的 YAML 回原路径（覆盖原文件）
    save_root.mkdir(parents=True, exist_ok=True)
    yaml_path = save_root.parent.joinpath("data.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    # 继续运行
    _model.val(data=yaml_path, imgsz=640)
    # 运行完之后删除临时yaml文件
    os.remove(yaml_path)
    