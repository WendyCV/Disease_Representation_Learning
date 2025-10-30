import os
import os.path as osp
import torch
from torchvision.datasets import VOCDetection
from torchvision.transforms import functional as F

VOC_CLASSES = ('__background__', 'unhealthy')
SZIE = 640
IMG_SIZE = (SZIE, SZIE)

class VOC2007Dataset(VOCDetection):
    def __init__(self, root, image_set = "train", transforms=None, size=SZIE, run_dir=None):
        super().__init__(root, "2007", image_set)
        self._transforms = transforms
        self._size = size
        if run_dir:
            self._run_dir = osp.join(run_dir, "ds")
            if not osp.exists(self._run_dir): os.makedirs(self._run_dir)
        else:
            self._run_dir = None
    
    def __getitem__(self, index):
        img, target = super(VOC2007Dataset, self).__getitem__(index)
        # 转换target标注为tensor格式
        boxes, labels = [], []
        for obj in target['annotation']['object']:
            if obj['name'] in VOC_CLASSES:
                bbox = obj['bndbox']
                xmin = int(bbox['xmin'])
                ymin = int(bbox['ymin'])
                xmax = int(bbox['xmax'])
                ymax = int(bbox['ymax'])
                # 插入box才能插入label
                if xmin >= 0 and ymin >= 0 and xmax > xmin and ymax > ymin:
                    boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
                    labels.append(VOC_CLASSES.index(obj['name']))
        # 转换targets为tensor格式
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels}
        # 转换image图片为tensor格式
        if self._transforms is not None:
            img, target = self._transforms(img, target, self._size)
            if self._run_dir:
                save_path = osp.join(self._run_dir, osp.basename(self.images[index]))
                self.save_transforms(img, target, save_path)
        # 返回tensor格式img和标注
        return F.to_tensor(img), target

    def save_transforms(self, img, target, save_path):
        from PIL import ImageDraw, ImageFont
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)
        font_size = 24.0
        font = ImageFont.truetype("arial.ttf", font_size)
        boxes = target['boxes'].tolist()
        labels = target['labels'].tolist()
        for box, label in zip(boxes, labels):
            draw.rectangle(box, outline='white', width=3)
            draw.text((box[0], 0.0 if box[1]<=font_size else box[1]-font_size), f'gt-{label}', font=font, fill='white')
        draw_img.save(save_path)

from torchvision import transforms
import CenterNetV1.utils as c_utils

class CenterNetDataset(VOC2007Dataset):
    def __init__(self, root, image_set = "train", transforms=None, size=SZIE, run_dir=None):
        super().__init__(root, image_set, transforms, size, run_dir)
        self._down_ratio = 4
        self._num_classes = 1
        self._max_objs = 100

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        real_h, real_w = img.shape[-2:]
        # 必须统一宽高，缩放至_size
        img = transforms.Resize((self._size, self._size))(img)
        # 开始计算centernet训练数据
        heatmap_size = self._size // self._down_ratio
        # heatmap
        hm = np.zeros((self._num_classes, heatmap_size, heatmap_size), dtype=np.float32)
        # width and hight
        wh = np.zeros((self._max_objs, 2), dtype=np.float32)
        # regression
        reg = np.zeros((self._max_objs, 2), dtype=np.float32)
        # index in 1D heatmap
        ind = np.zeros((self._max_objs), dtype=np.int64)
        # 1=there is a target in the list 0=there is not
        reg_mask = np.zeros((self._max_objs), dtype=np.uint8)
        # passthrought the original data
        by_pass = np.zeros((self._max_objs, 5), dtype=np.float32)
        # 遍历计算数据
        w_ratio = self._size / real_w / self._down_ratio
        h_ratio = self._size / real_h / self._down_ratio
        for idx, (bbox, cls) in enumerate(zip(target['boxes'].tolist(), target['labels'].tolist())):
            cls = 0
            # original bbox size -> heatmap bbox size
            bbox = bbox[0] * w_ratio, bbox[1] * h_ratio, bbox[2] * w_ratio, bbox[3] * h_ratio
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            # center point(x,y)
            center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            center_int = center.astype(np.int32)
            reg[idx] = center - center_int
            wh[idx] = 1. * width, 1. * height
            reg_mask[idx] = 1
            ind[idx] = center_int[1] * heatmap_size + center_int[0]
            radius = c_utils.gaussian_radius((height, width))
            # 半径保证为整数
            radius = max(0, int(radius))
            c_utils.draw_umich_gaussian(hm[cls], center_int, radius)
            # 透传
            by_pass[idx] = cls, bbox[0], bbox[1], bbox[2], bbox[3]
        # 返回数据结果
        return (img, hm, wh, reg, ind, reg_mask, by_pass)
    
def collate_fn(batch):
    return tuple(zip(*batch))

import random
import numpy as np
from PIL import Image

def random_resize(img, target, p=1.0, size=IMG_SIZE, max_size=2*SZIE):
    """
    随机缩放图片
    :param img:
    :param target:
    :param p:
    :return:
    """
    width, height = img.width, img.height
    if torch.rand(1) < p:
        img = F.resize(img, size=size, max_size=max_size)
        target['boxes'][:, [0, 2]] *= (img.size[0] / width)
        target['boxes'][:, [1, 3]] *= (img.size[1] / height)

    return img, target

def random_horizontal_flip(img, target, p=0.5):
    """
    随机水平翻转
    :param img:
    :param target:
    :param p:
    :return:
    """
    if torch.rand(1) < p:
        img = F.hflip(img)
        target['boxes'][:, [0, 2]] = img.size[0] - target['boxes'][:, [2, 0]]

    return img, target

def random_vertical_flip(img, target, p=0.5):
    """
    随机垂直翻转
    :param img:
    :param target:
    :param p:
    :return:
    """
    if torch.rand(1) < p:
        img = F.vflip(img)
        target['boxes'][:, [1, 3]] = img.size[1] - target['boxes'][:, [3, 1]]
    
    return img, target

def random_flip(img, target, p=0.5):
    random_flip = random.choice([random_horizontal_flip, random_vertical_flip])
    img, target = random_flip(img, target, p=p)
    
    return img, target

def color_jitter(img, target, p=1.0, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    """
    颜色随机调整
    :param img:
    :param target:
    :param brightness:
    :param contrast:
    :param saturation:
    :param hue:
    :return:
    """
    if torch.rand(1) < p:
        img = F.adjust_brightness(img, 1 + brightness * (torch.rand(1).item() * 2 - 1))
        img = F.adjust_contrast(img, 1 + contrast * (torch.rand(1).item() * 2 - 1))
        img = F.adjust_saturation(img, 1 + saturation * (torch.rand(1).item() * 2 - 1))
        img = F.adjust_hue(img, hue * (torch.rand(1).item() * 2 - 1))

    return img, target

def _random_erase_in_box_(img, box, p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)):
    """
    标注框随机擦除
    :param img:
    :param box:
    :param p:
    :param scale:
    :param ratio:
    :return:
    """
    if random.uniform(0, 1) > p:
        return img

    xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    box_w, box_h = xmax - xmin, ymax - ymin

    while True:
        erase_area = random.uniform(scale[0], scale[1]) * box_w * box_h
        aspect_ratio = random.uniform(ratio[0], ratio[1])

        erase_h = int(np.sqrt(erase_area * aspect_ratio))
        erase_w = int(np.sqrt(erase_area / aspect_ratio))

        if erase_h < box_h and erase_w < box_w:
            break

    x1 = random.randint(ymin, ymax - erase_h)
    y1 = random.randint(xmin, xmax - erase_w)

    x2 = x1 + erase_h
    y2 = y1 + erase_w

    img[x1:x2, y1:y2, :] = 0  # 擦除区域设置为黑色

    return img

def random_erase_boxes(img, target, p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)):
    """
    对图片中所有标注框进行随机擦除
    :param img:
    :param target:
    :param p:
    :param scale:
    :param ratio:
    :return:
    """
    img = np.array(img)

    for box in target['boxes']:
        img = _random_erase_in_box_(img, box.numpy(), p=p, scale=scale, ratio=ratio)
    
    return Image.fromarray(img), target

def random_gaussian_noise(img, target, p=0.2, mean=0.1, sigma=0.08):
    """
    对图片进行随机高斯噪声
    :param img:
    :param target:
    :param p:
    :param mean:
    :param sigma:
    :return:
    """
    img = np.array(img)

    if torch.rand(1) < p:
        img = img/255
        noise = np.random.normal(mean, sigma, img.shape)
        img = img + noise
        img = np.clip(img, 0, 1)
        img = np.uint8(img*255)

    return Image.fromarray(img), target

def random_gaussian_blur(img, target, p=0.3, kernel_size=7, sigma=3):
    """
    对图片进行随机高斯模糊
    :param img:
    :param target:
    :param p:
    :param kernel_size:
    :param sigma:
    :return:
    """
    if torch.rand(1) < p:
        img = F.gaussian_blur(img, kernel_size, sigma)

    return img, target

def random_motion_blur(img, target, p=0.3, degree=15, angle=45):
    """
    对图片进行随机运动模糊
    :param img:
    :param target:
    :param p:
    :param degree:
    :param angle:
    :return:
    """
    img = np.array(img)

    if torch.rand(1) < p:
        import cv2
        M = cv2.getRotationMatrix2D((degree/2, degree/2), angle, 1)
        K = np.diag(np.ones(degree))
        K = cv2.warpAffine(K, M, (degree, degree))
        K = K/degree
        img = cv2.filter2D(img, -1, K)
        cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        img = np.array(img, dtype=np.uint8)

    return Image.fromarray(img), target

def random_median_blur(img, target, p=0.3, ksize=5):
    """
    对图片进行随机运动模糊
    :param img:
    :param target:
    :param p:
    :param ksize:
    :return:
    """
    img = np.array(img)

    if torch.rand(1) < p:
        import cv2
        img = cv2.medianBlur(img, ksize)

    return Image.fromarray(img), target

def random_blur(img, target, p=0.3):
    """
    对图片进行随机模糊
    :param img:
    :param target:
    :param p:
    :return:
    """
    random_blur = random.choice([random_motion_blur, random_median_blur, random_gaussian_blur])
    img, target = random_blur(img, target, p=p)
    
    return img, target

def random_grayscale(img, target, p=0.2, output_channels=3):
    """
    对图片进行灰度变换
    :param img:
    :param target:
    :param p:
    :param output_channels:
    :return:
    """
    if torch.rand(1) < p:
        img = F.to_grayscale(img, num_output_channels=output_channels)

    return img, target

def random_padding(img, target, p=0.2, padding=(320, 320, 320, 320)):
    """
    对图片进行随机padding
    :param img:
    :param target:
    :param p:
    :param padding: left, top, right, bottom
    :return:
    """
    if torch.rand(1) < p:
        left = 2 * (random.randint(padding[0]/10, padding[0]) // 2)
        top = 2 * (random.randint(padding[1]/10, padding[1]) // 2)
        right = 2 * (random.randint(padding[2]/10, padding[2]) // 2)
        bottom = 2 * (random.randint(padding[3]/10, padding[3]) // 2)
        img = F.pad(img, padding=[left, top, right, bottom])
        # 调整left和top的offset
        target['boxes'][:, [0, 2]] += left
        target['boxes'][:, [1, 3]] += top 

    return img, target

def random_cutoff(img, target, p=0.2, size=IMG_SIZE):
    """
    对图片进行随机cutoff
    :param img:
    :param target:
    :param p:
    :param size: cutoff后图片尺寸
    :return:
    """
    def area(left, top, right, bottom):
        return (right-left)*(bottom-top)

    if torch.rand(1) < p:
        width, height = img.width, img.height
        w_delta = width - size[0]
        h_delta = height - size[1]
        if w_delta <= 0 or h_delta <= 0:
            return img, target
        x_start = random.randint(0, w_delta)
        y_start = random.randint(0, h_delta)
        x_end = x_start + size[0]
        y_end = y_start + size[1]
        boxes, labels = [], []
        for box, label in zip(target['boxes'], target['labels']):
            box, label = box.numpy(), label.item()
            area_box = area(box[0], box[1], box[2], box[3])
            left, top = max(x_start, box[0]), max(y_start, box[1])
            right, bottom = min(x_end, box[2]), min(y_end, box[3])
            area_iou = area(left, top, right, bottom)
            if right > left and bottom > top and area_iou > 196 and (area_iou / area_box) > 0.25:
                boxes.append([left-x_start, top-y_start, right-x_start, bottom-y_start])
                labels.append(label)
        if len(boxes) <= 0:
            return img, target
        target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.tensor(labels, dtype=torch.int64)
        img = img.crop(box=(x_start, y_start, x_end, y_end))

    return img, target

def F_train_transforms(img, target, size:int=SZIE):
    """
    对训练图片进行随机augment
    :param img:
    :param target:
    :param p:
    :param size: cutoff后图片尺寸
    :return:
    """
    img, target = random_resize(img, target, size=size, max_size=2*size, p=1.0)
    img, target = random_flip(img, target, p=0.1)
    img, target = random_gaussian_noise(img, target, p=0.01)
    img, target = random_blur(img, target, p=0.01)
    img, target = color_jitter(img, target, p=0.01)
    img, target = random_erase_boxes(img, target, p=0.1)
    img, target = random_grayscale(img, target, p=0.01)
    img, target = random_padding(img, target, p=0.1)
    img, target = random_cutoff(img, target, p=0.01, size=(size, size))
    # 如果变换过程，taget的box会发生变化，需要删除该box
    
    return img, target

def F_val_transforms(img, target, size=SZIE):
    """
    对验证图片进行随机resize
    :param img:
    :param target:
    :param p:
    :param size: cutoff后图片尺寸
    :return:
    """
    img, target = random_resize(img, target, size=size, max_size=2*size, p=1.0)
    return img, target
