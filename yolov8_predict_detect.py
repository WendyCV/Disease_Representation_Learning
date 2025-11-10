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
    parser.add_argument("--task", type=str, default="detect", choices=["detect"], help="训练任务类型")
    parser.add_argument("--config", type=str, default="yolov8m.yaml", help="YAML配置")
    parser.add_argument("--pretrain", type=str, default="best.pt", help="预训练权重")
    parser.add_argument("--dir_suffix", type=str, default="", help="预训练权重目录后缀")
    parser.add_argument("--data_path", type=str, default="detect_durian_leaf.yaml", help="预测数据配置")
    parser.add_argument("--predict_only", action='store_true', help="只做predict测试")
    parser.add_argument("--val_only", action='store_true', help="只做val测试")
    parser.add_argument("--conf", type=float, default=0.28, help="检测置信度设定")
    parser.add_argument("--iou", type=float, default=0.62, help="检测IoU设定")
    args = parser.parse_args()
    print("[RUN-args]:", args)
    # 获取参数开始训练
    model_path = Path(make_abs_path("models")).joinpath(args.config)
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
            "iou": args.iou,      # IoU 阈值
            # "half": True,
            # "augment": True,
        }
        # 进行推理预测
        test_dir = Path(make_abs_path("datasets")).joinpath(args.task).joinpath("test").joinpath("images")
        # test_dir = Path(fr"D:\Durian_YOLO\runs\test")
        result = _model.predict(source=test_dir, save_dir=save_root, **kwargs)[0]
        if args.dir_suffix and args.dir_suffix != "":
            old_dir = Path(result.save_dir)
            new_dir = old_dir.with_name(old_dir.name + f"_{args.dir_suffix}")
            os.rename(old_dir, new_dir)
    # //todo: 检查数据精度
    if not args.predict_only:
        yaml_path = Path(make_abs_path("datasets")).joinpath(args.task).joinpath(args.data_path)
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        # 替换 val 为 test
        data["train"] = str(yaml_path.joinpath(data["train"]).resolve())
        data["val"] = str(yaml_path.joinpath(data["test"]).resolve())
        data["test"] = str(yaml_path.joinpath(data["test"]).resolve())
        # 保存修改后的 YAML 回原路径（覆盖原文件）
        save_root.mkdir(parents=True, exist_ok=True)
        yaml_path = save_root.parent.joinpath("data.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        # 继续运行
        kwargs = {
            "save": True,
            "save_json": True,
            "conf": args.conf,    # 置信度阈值
            "iou": args.iou,      # IoU 阈值
            # "augment": True,
        }
        result = _model.val(data=yaml_path, imgsz=640, **kwargs)
        if args.dir_suffix and args.dir_suffix != "":
            old_dir = Path(result.save_dir)
            new_dir = old_dir.with_name(old_dir.name + f"_{args.dir_suffix}")
            os.rename(old_dir, new_dir)
        # 运行完之后删除临时yaml文件
        os.remove(yaml_path)
        if len(os.listdir(save_root)) == 0: os.rmdir(save_root)