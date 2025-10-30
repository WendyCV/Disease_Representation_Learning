import os
import os.path as osp
import sys
import tqdm
import torch
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models import resnet50, ResNet50_Weights, resnet34, ResNet34_Weights
from torchvision.models.detection.faster_rcnn import FasterRCNN, TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from voc_dataset import VOC_CLASSES, VOC2007Dataset, collate_fn, IMG_SIZE, SZIE
from utils import logger

def makd_abs_path(fn):
    return osp.abspath(osp.join(osp.dirname(__file__), fn))

def model_by(required_size=SZIE, num_classes=len(VOC_CLASSES), model_path:str=None, backbone_weights=None):
    # 提取特征
    # resnet34
    if backbone_weights is None: backbone_weights = ResNet34_Weights.DEFAULT
    backbone = resnet34(weights=backbone_weights)
    # resnet50
    # if backbone_weights is None: backbone_weights = ResNet50_Weights.DEFAULT
    # backbone = resnet50(weights=backbone_weights)
    # 训练网络部分
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1", "bn1"]
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]): parameter.requires_grad_(False)
    # FPN部分
    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    in_channels_stage = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage,         # layer1 out_channel=256
        in_channels_stage * 2,     # layer2 out_channel=512
        in_channels_stage * 4,     # layer3 out_channel=1024
        in_channels_stage * 8,     # layer4 out_channel=2048
    ]
    out_channels = 512
    backbone = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
    # 候选框
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = [(0.5, 1.0, 2.0),] * len(anchor_sizes)
    rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios)
    rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)
    # 自回归：cls、bbox
    featmap_names = [v for k, v in return_layers.items()]
    box_roi_pool = MultiScaleRoIAlign(featmap_names=featmap_names, output_size=7, sampling_ratio=2)
    resolution = box_roi_pool.output_size[0]
    box_head = TwoMLPHead(out_channels * resolution**2, 1024)
    box_predictor = FastRCNNPredictor(1024, num_classes)
    # faster rcnn模型
    neck_bridge = NeckBridge(backbone)
    model = FasterRCNN(backbone=neck_bridge, min_size=required_size, max_size=2*required_size,
                       rpn_anchor_generator=rpn_anchor_generator, rpn_head=rpn_head,
                       box_roi_pool=box_roi_pool, box_head=box_head, box_predictor=box_predictor)
    # 加载训练pth
    if model_path is not None and osp.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    # 返回模型
    return model

import torch.nn as nn
from attention import CBAM

class NeckBridge(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.out_channels = backbone.out_channels
        self.att_seq = nn.Sequential(
            CBAM(c1=backbone.out_channels, kernel_size=7)
        )

    def forward(self, x):
        x = self.backbone(x)
        for k, v in x.items():
            x[k] = self.att_seq(v)
        return x

import numpy as np
import torch.optim as optim
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader, RandomSampler
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from voc_dataset import F_train_transforms, F_val_transforms

def train_by(model, data_path:str, num_epochs=100, last_epoch=-1, **kwargs):
    # 等待50个epoch没有更新，则退出训练
    early_stop_patience = 50
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True)
    epoch_lr_s = CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs, eta_min=0.00005)
    # epoch_lr_s = MultiStepLR(optimizer=optimizer, milestones=[80, 150], gamma=0.1)
    weight_dir = makd_abs_path(r"faster_rcnn\train\weights") if "out_dir" not in kwargs else kwargs["out_dir"]
    if not osp.exists(weight_dir): os.makedirs(weight_dir)
    # 数据集配置
    dataset_train = VOC2007Dataset(root=data_path, image_set='train', transforms=F_train_transforms, run_dir=makd_abs_path(r"faster_rcnn\train"))
    dataloader_train = DataLoader(dataset_train, batch_size=8, num_workers=4, shuffle=True, collate_fn=collate_fn)
    dataset_val = VOC2007Dataset(root=data_path, image_set='val', transforms=F_val_transforms)
    # sampler = RandomSampler(dataset_val, num_samples=16)
    dataloader_val = DataLoader(dataset_val, batch_size=1, num_workers=4, collate_fn=collate_fn)
    # 开始训练epoch
    def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor=0.001):
        def f(x):
            if x >= warmup_iters:
                return 1
            alpha = float(x) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha
        return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
    model.cuda()
    best_score = 0.0
    best_mAP = {"mAP":0.0, "mAP50": 0.0, "mAP75": 0.0, "mAR": 0.0, "loss": 0.0, "epoch": 0}
    for epoch in range(max(0, last_epoch), num_epochs):
        # 训练epoch
        if epoch <= 0: warmup_lr_s = warmup_lr_scheduler(optimizer, min(1000, len(dataloader_train)))
        else: warmup_lr_s = None
        last_loss, avg_loss = train_a_epoch(model, dataloader_train, optimizer, epoch, num_epochs, warmup_lr_s)
        # 验证epoch
        mAP, mAP50, mAP75, mAR = val_a_epoch(model, dataloader_val, epoch, num_epochs)
        val_score = calc_best_score(mAP, mAR, mAP50, mAP75)
        # 保持val最优结果
        if val_score > best_score:
            torch.save(model.state_dict(), osp.join(weight_dir, r'best.pth'))
            best_mAP["mAP"] = mAP
            best_mAP["mAP50"] = mAP50
            best_mAP["mAP75"] = mAP75
            best_mAP["mAR"] = mAR
            best_mAP["loss"] = last_loss
            best_mAP["epoch"] = epoch
            best_score = val_score
        torch.save(model.state_dict(), osp.join(weight_dir, r'last.pth'))
        # 更新lr
        epoch_lr_s.step()
        # 输出结果
        logger(f'Epoch [{epoch + 1}/{num_epochs}], lr: {epoch_lr_s.get_last_lr()[0]:.5f}, mAP: {best_mAP["mAP"]:.4f}, mAP@0.50: {best_mAP["mAP50"]:.4f}, mAP@0.75: {best_mAP["mAP75"]:.4f}, mAR: {best_mAP["mAR"]:.4f}, loss: {best_mAP["loss"]:.4f}')
        # EarlyStopping: 如果50次都没有更新
        if epoch > best_mAP["epoch"] + early_stop_patience:
            break
    # 清空缓存
    torch.cuda.empty_cache()
    # 保存结果
    torch.save(model.state_dict(), osp.join(weight_dir, fr'epoch{num_epochs}.pth'))

def calc_best_score(mp, mr, mp50, mp75):
    return 0.25*mp + 0.25*mr + 0.35*mp50 + 0.15*mp75

def log_gpu_memory() -> float:
    return (torch.cuda.memory_allocated() + torch.cuda.memory_reserved()) / 1024 ** 3

def train_a_epoch(model, dataloader, optimizer, epoch, num_epochs, warmup_lr_s=None):
    model.train()
    batch_loss = []
    gpu_usage = []
    for images, targets in tqdm.tqdm(dataloader, file=sys.stdout):
        optimizer.zero_grad()
        images = [image.to("cuda") for image in images]
        targets = [{k: v.to("cuda") for k, v in t.items()} for t in targets]
        # 前向传播
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        # 反向传播
        losses.backward()
        optimizer.step()
        if warmup_lr_s: warmup_lr_s.step()
        # 记录epoch的loss值
        batch_loss.append(loss_value)
        # 记录gpu使用
        gpu_usage.append(log_gpu_memory())
    # 清空缓存
    torch.cuda.empty_cache()
    # 计算指标
    last_loss = batch_loss[-1]
    avg_loss = np.mean(batch_loss)
    avg_gpu_usage = np.max(gpu_usage)
    # 输出结果
    logger(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {last_loss:.4f}, Avg loss: {avg_loss:.4f}, GPU: {avg_gpu_usage:.2f}GB')
    return last_loss, avg_loss

def val_a_epoch(model, dataloader, epoch, num_epochs):
    model.eval()
    target = []
    preds = []
    for images, targets in tqdm.tqdm(dataloader, file=sys.stdout):
        images = [image.to("cuda") for image in images]
        targets = [{k: v.to("cuda") for k, v in t.items()} for t in targets]
        # 前向传播
        with torch.no_grad():
            outputs = model(images, targets)
        # 获取结果
        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)
    # 清空缓存
    torch.cuda.empty_cache()
    # 计算指标
    metric = MeanAveragePrecision()
    metric.warn_on_many_detections = False
    metric.update(preds, target)
    metric_summary = metric.compute()
    mAP = metric_summary["map"] if metric_summary["map"] >= 0 else 0.0
    mAP50 = metric_summary["map_50"] if metric_summary["map_50"] >= 0 else 0.0
    mAP75 = metric_summary["map_75"] if metric_summary["map_75"] >= 0 else 0.0
    mAR = np.mean([
        metric_summary["mar_1"] if metric_summary["mar_1"] >= 0 else 0.0, 
        metric_summary["mar_10"] if metric_summary["mar_10"] >= 0 else 0.0, 
        metric_summary["mar_100"] if metric_summary["mar_100"] >= 0 else 0.0
    ])
    # 输出结果
    logger(f'Epoch [{epoch + 1}/{num_epochs}], mAP: {mAP:.4f}, mAP@0.50: {mAP50:.4f}, mAP@0.75: {mAP75:.4f}, mAR: {mAR:.4f}')
    return mAP, mAP50, mAP75, mAR

import glob
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F

def predict_by(model, source_dir, trues_dict=None, **kwargs):
    model.eval()
    images = glob.glob(osp.join(source_dir, "*.jpg"))
    predict_dir = makd_abs_path(r"faster_rcnn\predict") if "out_dir" not in kwargs else kwargs["out_dir"]
    if not osp.exists(predict_dir): os.makedirs(predict_dir)
    size = kwargs["required_size"] if "required_size" in kwargs else None
    threshold = kwargs["threshold"] if "threshold" in kwargs else 0.0
    trues = []
    preds = []
    model.cuda()
    for img_path in tqdm.tqdm(images, file=sys.stdout):
        from voc_dataset import random_resize
        # 读取图片
        img = Image.open(img_path).convert("RGB")
        if size is None:
            img_size = img.copy() 
        else:
            resize = max(size[0], size[1])
            img_size = F.resize(img, size=[resize], max_size=resize*2)
        img_tensor = F.to_tensor(img_size)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        img_tensor = img_tensor.to("cuda")
        # 前向传播
        with torch.no_grad():
            outputs = model(img_tensor)
        # 获取结果
        boxes = outputs[0]['boxes'].detach().cpu().numpy()
        scores = outputs[0]['scores'].detach().cpu().numpy()
        labels = outputs[0]['labels'].detach().cpu().numpy()
        # 缩放还原
        if img.size != img_size.size:
            boxes[:, [0, 2]] *= img.width/img_size.width
            boxes[:, [1, 3]] *= img.height/img_size.height
        preds_dict = {'boxes': torch.tensor(boxes, dtype=torch.float32), 'scores': torch.tensor(scores, dtype=torch.float32), 'labels': torch.tensor(labels, dtype=torch.int64)}
        preds.append(preds_dict)
        # 得分过滤
        select_indicators = scores >= threshold
        boxes = boxes[select_indicators]
        scores = scores[select_indicators]
        labels = labels[select_indicators]
        # 保存结果
        preds_dict = {'boxes': boxes, 'scores': scores, 'labels': labels}
        base_name = osp.basename(img_path)
        save_predict_result(image=img, preds_dict=preds_dict, save_path=osp.join(predict_dir, base_name), color="red")
        # 如果存在trues_dict计算指标
        name_key = osp.splitext(base_name)[0]
        if trues_dict and name_key in trues_dict:
            true_boxes = trues_dict[name_key]["boxes"]
            true_boxes = [[img.width*box[0], img.height*box[1], 
                           img.width*box[2], img.height*box[3]] for box in true_boxes]
            true_boxes = [[box[0]-box[2]/2, box[1]-box[3]/2, box[0]+box[2]/2, box[1]+box[3]/2] for box in true_boxes]
            true_labels = trues_dict[name_key]["labels"]
            true_scores = [1.0 for _ in true_labels]
            true_dict = {'boxes': true_boxes, 'scores': true_scores, 'labels': true_labels}
            save_predict_result(image=img, preds_dict=true_dict, save_path=osp.join(predict_dir, base_name), color="white")
            true_dict = {'boxes': torch.tensor(true_boxes, dtype=torch.float32), 'scores': torch.tensor(true_scores, dtype=torch.float32), 'labels': torch.tensor(true_labels, dtype=torch.int64)}
            trues.append(true_dict)
    # 清空缓存
    torch.cuda.empty_cache()
    # 计算指标
    if len(preds) == len(trues) and len(preds) > 0:
        metric = MeanAveragePrecision()
        metric.warn_on_many_detections = False
        metric.update(preds, trues)
        metric_summary = metric.compute()
        mAP50 = metric_summary["map_50"]
        mAP = metric_summary["map"]
        logger(f'mAP: {mAP:.4f}, mAP@0.50: {mAP50:.4f}')

def save_predict_result(image, preds_dict, save_path, color="red"):
    draw = ImageDraw.Draw(image)
    line_width = 12
    font_size = 48.0
    font = ImageFont.truetype("arial.ttf", font_size)
    boxes = preds_dict["boxes"]
    scores = preds_dict["scores"]
    labels = preds_dict["labels"]
    for box, score, label in zip(boxes, scores, labels):
        draw.rectangle(box, outline=color, width=line_width)
        draw.text((box[0], 0.0 if box[1]<=font_size else box[1]-font_size), text=f"{score:.2f}:{label}", font=font, fill=color)
    image.save(save_path)
