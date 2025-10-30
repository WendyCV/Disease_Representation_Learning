import sys
import logging
from logging import Logger
_log_logger_: Logger = logging.getLogger('DURIAN_LEAF')
_log_logger_.setLevel(logging.DEBUG)
# 创建Formatter，设定日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 添加一个handler来输出到控制台
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
# 将Handler添加到Logger
_log_logger_.addHandler(console_handler)

def logger_file(log_path):
    # 创建FileHandler，输出到日志文件
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    # 将Handler添加到Logger
    _log_logger_.addHandler(file_handler)

def logger(msg: str):
    _log_logger_.info(msg)


import torch
from torch import Tensor
import torchvision

def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()

def filter_overlapping(prediction, overlapping_threshold=0.8):
    boxes = prediction['boxes']
    area = torchvision.ops.box_area(boxes)

    lt = torch.max(boxes[:, None, :2], boxes[:, :2])  # [N,M,2]
    rb = torch.min(boxes[:, None, 2:], boxes[:, 2:])  # [N,M,2]

    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    overlap = inter/area
    overlap.fill_diagonal_(0)
    high_overlap_set = set()
    for idx_i in range(0, len(boxes)-1):
        for idx_j in range(idx_i+1, len(boxes)):
            overlap_ratio = max(overlap[idx_i, idx_j], overlap[idx_j, idx_i])
            if overlap_ratio > overlapping_threshold:
                high_overlap_set.add(idx_j)
    low_overlap_index = [idx for idx in range(len(boxes)) if idx not in high_overlap_set]
    if len(low_overlap_index) < len(boxes):
        prediction['boxes'] = prediction['boxes'][low_overlap_index]
        prediction['scores'] = prediction['scores'][low_overlap_index]
        prediction['labels'] = prediction['labels'][low_overlap_index]
    return prediction

def apply_nms(prediction, iou_threshold=0.5, score_threshold=0.5):
    """
    通过iou重叠区域去除多余候选框
    :param prediction:
    :param iou_threshold:
    :param score_threshold:
    :return:
    """
    keep = torchvision.ops.nms(prediction['boxes'], prediction['scores'], iou_threshold)
    prediction['boxes'] = prediction['boxes'][keep]
    prediction['scores'] = prediction['scores'][keep]
    prediction['labels'] = prediction['labels'][keep]
    # Remove boxes with scores below the score threshold
    high_score_idx = torch.where(prediction['scores'] >= score_threshold)
    prediction['boxes'] = prediction['boxes'][high_score_idx]
    prediction['scores'] = prediction['scores'][high_score_idx]
    prediction['labels'] = prediction['labels'][high_score_idx]
    #
    prediction = filter_overlapping(prediction)
    return prediction
