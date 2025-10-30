import os
import os.path as osp
import sys
import tqdm
import torch
from torchvision.models import resnet50, ResNet50_Weights, resnet34, ResNet34_Weights
from CenterNetV1.losses import FocalLoss, RegL1Loss
from voc_dataset import VOC_CLASSES, CenterNetDataset, collate_fn, IMG_SIZE, SZIE
from utils import logger

def makd_abs_path(fn):
    return osp.abspath(osp.join(osp.dirname(__file__), fn))

def model_by(required_size=SZIE, num_classes=len(VOC_CLASSES), model_path:str=None, backbone_weights=None):
    # 提取特征
    # resnet34
    if backbone_weights is None: backbone_weights = ResNet34_Weights.DEFAULT
    model_backbone = resnet34(weights=backbone_weights)
    backbone_out_channels = model_backbone.inplanes
    # resnet50
    # if backbone_weights is None: backbone_weights = ResNet50_Weights.DEFAULT
    # model_backbone = resnet50(weights=backbone_weights)
    # 删除avgpool层和fc层
    backbone = nn.Sequential(
        model_backbone.conv1, model_backbone.bn1, model_backbone.relu, model_backbone.maxpool, 
        model_backbone.layer1, model_backbone.layer2, model_backbone.layer3, model_backbone.layer4
    )
    # 注意力机制
    neck_bridge = NeckBridge(backbone, backbone_out_channels)
    # 构建网络
    from CenterNetV1.models import certernetv1
    model = certernetv1(neck_bridge, num_classes=num_classes)
    # 加载训练pth
    if model_path is not None and osp.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    # 返回模型
    return model

import torch.nn as nn
from attention import CBAM

class NeckBridge(nn.Module):
    def __init__(self, backbone, out_channels):
        super().__init__()
        self.backbone = backbone
        self.out_channels = out_channels
        self.att_seq = nn.Sequential(
            CBAM(c1=out_channels, kernel_size=7)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.att_seq(x)
        return x

import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader, RandomSampler
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from voc_dataset import F_train_transforms, F_val_transforms

def train_by(model, data_path:str, num_epochs=100, last_epoch=-1, **kwargs):
    # 等待50个epoch没有更新，则退出训练
    early_stop_patience = 50
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)
    epoch_lr_s = CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs, eta_min=0.00005)
    weight_dir = makd_abs_path(r"centernet\train\weights") if "out_dir" not in kwargs else kwargs["out_dir"]
    if not osp.exists(weight_dir): os.makedirs(weight_dir)
    # 数据集配置
    train_dataset = CenterNetDataset(root=data_path, image_set='train', transforms=F_train_transforms, run_dir=makd_abs_path(r"centernet\train"))
    dataloader_train = DataLoader(dataset=train_dataset, batch_size=16, num_workers=4, shuffle=True)
    val_dataset = CenterNetDataset(root=data_path, image_set='val', transforms=F_val_transforms)
    # sampler = RandomSampler(dataset_val, num_samples=16)
    dataloader_val = DataLoader(dataset=val_dataset, batch_size=1, num_workers=4)
    # 损失函数
    cerition_hm = FocalLoss()
    cerition_wh = RegL1Loss()
    cerition_reg = RegL1Loss()
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
        last_loss, avg_loss = train_a_epoch(model, dataloader_train, optimizer, cerition_hm, cerition_wh, cerition_reg, epoch, num_epochs, warmup_lr_s)
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

def train_a_epoch(model, dataloader, optimizer, cerition_hm, cerition_wh, cerition_reg, epoch, num_epochs, warmup_lr_s=None):
    model.train()
    batch_loss = []
    gpu_usage = []
    for batch_item in tqdm.tqdm(dataloader, file=sys.stdout):
        batch_item = [x.to("cuda") for x in batch_item]
        img, hm, wh, reg, ind, reg_mask, _ = batch_item
        optimizer.zero_grad()
        # 前向传播
        out_hm, out_wh, out_reg = model(img)
        hm_loss = cerition_hm(out_hm, hm)
        wh_loss = cerition_wh(out_wh, wh, reg_mask, ind)
        reg_loss = cerition_reg(out_reg, reg, reg_mask, ind)
        loss = hm_loss + 0.1 * wh_loss + reg_loss
        # 反向传播
        loss.backward()
        optimizer.step()
        if warmup_lr_s: warmup_lr_s.step()
        # 记录epoch的loss值
        batch_loss.append(loss.item())
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

import CenterNetV1.utils as c_utils

def val_a_epoch(model, dataloader, epoch, num_epochs):
    model.eval()
    targets = []
    preds = []
    for batch_item in tqdm.tqdm(dataloader, file=sys.stdout):
        batch_item = [x.to("cuda") for x in batch_item]
        max_bbox_size = 0
        img, hm, wh, reg, ind, reg_mask, cls_bbox_batch = batch_item
        for i in range(len(cls_bbox_batch)):
            true_dict = dict()
            bbox_list= []
            cls_list = []
            cls_bbox_list = cls_bbox_batch[i].detach().cpu().numpy()
            for cls_bbox in cls_bbox_list:
                cls = int(cls_bbox[0])
                bbox = cls_bbox[-4:]
                if bbox[2]>bbox[0] and bbox[3]>bbox[1]:
                    bbox_list.append(bbox)
                    cls_list.append(cls)
            if len(bbox_list) > max_bbox_size:
                max_bbox_size = len(bbox_list)
            true_dict['boxes'] = torch.tensor(np.array(bbox_list), dtype=torch.float32)
            true_dict['labels'] = torch.tensor(cls_list, dtype=torch.int64)
            targets.append(true_dict)
        # 前向传播
        with torch.no_grad():
            out_hm, out_wh, out_reg = model(img)
        bbox, cls, scores = c_utils.heatmap_bbox(out_hm, out_wh, out_reg, k=100)
        # 同一维度和类型，便于cat
        cls = cls.unsqueeze(-1).float()
        scores = scores.unsqueeze(-1).float()
        #只测试一张图片batch=1，去掉该维度
        bbox_cls_score = torch.cat([bbox, cls, scores], dim=-1).squeeze()
        #使用soft_nms过滤掉不同类别在同一个关键点位置的情况
        bbox_cls_score = c_utils.soft_nms(bbox_cls_score, score_threshold=0.5, top_k=100)
        bbox_cls_score = bbox_cls_score.detach().cpu().numpy()
        bbox_list= []
        cls_list = []
        score_list = []
        for bcs in bbox_cls_score:
            bbox, cls, score = bcs[:4], int(bcs[4]), bcs[-1]
            bbox_list.append(bbox)
            cls_list.append(cls)
            score_list.append(score)
        # 添加一个元素
        preds_dict = dict()
        preds_dict['boxes'] = torch.tensor(np.array(bbox_list), dtype=torch.float32)
        preds_dict['labels'] = torch.tensor(cls_list, dtype=torch.int64)
        preds_dict['scores'] = torch.tensor(score_list, dtype=torch.float32)
        preds.append(preds_dict)
    # 清空缓存
    torch.cuda.empty_cache()
    # 计算指标
    metric = MeanAveragePrecision()
    metric.warn_on_many_detections = False
    metric.update(preds, targets)
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
    predict_dir = makd_abs_path(r"centernet\predict") if "out_dir" not in kwargs else kwargs["out_dir"]
    if not osp.exists(predict_dir): os.makedirs(predict_dir)
    size = kwargs["required_size"] if "required_size" in kwargs else None
    threshold = kwargs["threshold"] if "threshold" in kwargs else 0.0
    trues = []
    preds = []
    model.cuda()
    for img_path in tqdm.tqdm(images, file=sys.stdout):
        # 读取图片
        img = Image.open(img_path).convert("RGB")
        real_w, real_h = img.size
        if size is None:
            img_size = img.copy() 
            w_ratio = real_w * 4
            h_ratio = real_h * 4
        else:
            resize = max(size[0], size[1])
            img_size = F.resize(img, size=[resize], max_size=resize*2)
            size_w, size_h = img_size.size
            w_ratio = real_w * 4 / size_w
            h_ratio = real_h * 4 / size_h
        img_tensor = F.to_tensor(img_size)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        img_tensor = img_tensor.to("cuda")
        # 前向传播
        with torch.no_grad():
            out_hm, out_wh, out_reg = model(img_tensor)
        bbox, cls, scores = c_utils.heatmap_bbox(out_hm, out_wh, out_reg, k=100)
        # 同一维度和类型，便于cat
        cls = cls.unsqueeze(-1).float()
        scores = scores.unsqueeze(-1).float()
        #只测试一张图片batch=1，去掉该维度
        bbox_cls_score = torch.cat([bbox, cls, scores], dim=-1).squeeze()
        #使用soft_nms过滤掉不同类别在同一个关键点位置的情况
        bbox_cls_score = c_utils.soft_nms(bbox_cls_score, score_threshold=0.5, top_k=100)
        bbox_cls_score = bbox_cls_score.detach().cpu().numpy()
        bbox_list= []
        cls_list = []
        score_list = []
        for bcs in bbox_cls_score:
            bbox, cls, score = bcs[:4], int(bcs[4]), bcs[-1]
            if score > threshold:
                bbox_list.append([bbox[0] * w_ratio, bbox[1] * h_ratio, bbox[2] * w_ratio, bbox[3] * h_ratio])
                cls_list.append(cls)
                score_list.append(score)
        # 保存结果
        preds_dict = {'boxes': bbox_list, 'scores': score_list, 'labels': cls_list}
        base_name = osp.basename(img_path)
        save_predict_result(image=img, preds_dict=preds_dict, save_path=osp.join(predict_dir, base_name), color="red")
    # 清空缓存
    torch.cuda.empty_cache()

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