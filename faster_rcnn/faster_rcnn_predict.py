import os
import os.path as osp
from faster_rcnn_tools import model_by, predict_by
from voc_dataset import VOC_CLASSES, IMG_SIZE, SZIE
from utils import logger, logger_file
import torch
torch.backends.cudnn.enabled=False
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

def makd_abs_path(fn):
    return osp.abspath(osp.join(osp.dirname(__file__), fn))

import glob

def load_trues_dict(labels_dir):
    label_files = glob.glob(osp.join(labels_dir, "*.txt"))
    trues_dict = {}
    for label_path in label_files:
        if not osp.exists(label_path):
            continue
        boxes, labels = [], []
        with open(label_path, "r", encoding="utf8", errors="ignore") as fd:
            for line in fd.readlines():
                parts = line.split(" ", 5)
                if len(parts) != 5:
                    continue
                labels.append(int(parts[0])+1)
                boxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
        name_key = osp.splitext(osp.basename(label_path))[0]
        trues_dict[name_key] = {"boxes": boxes, "labels": labels}
    return trues_dict

if __name__ == '__main__':
    run_dir = r"faster_rcnn\predict"
    if not osp.exists(run_dir): os.makedirs(run_dir)
    logger_file(osp.join(run_dir, "faster_rcnn_predict_log.log"))
    _model_ = model_by(required_size=SZIE, num_classes=len(VOC_CLASSES), model_path=makd_abs_path(r"faster_rcnn\train\weights\best.pth"))
    logger(_model_)
    trues_dict = load_trues_dict(labels_dir=makd_abs_path(r"datasets\labels\val"))
    predict_by(_model_, source_dir=makd_abs_path(r"datasets\images\val"), trues_dict=trues_dict, required_size=IMG_SIZE, threshold=0.5)
    