import torch.nn as nn
from yolov8_utils import make_abs_path
from yolov8_train_classify import load_model as load_classify_model, get_backbone as get_classify_backbone, get_attention as get_classify_attention
from yolov8_train_detect import load_model as load_detect_model, get_backbone as get_detect_backbone, get_attention as get_detect_attention

def load_model(model_path, task, pretrain_path=None, modify_model=False):
    if task == "classify":
        return load_classify_model(model_path=model_path, task=task, pretrain_path=pretrain_path, modify_model=modify_model)
    elif task == "detect":
        return load_detect_model(model_path=model_path, task=task, pretrain_path=pretrain_path, modify_model=modify_model)
    else:
        raise RuntimeError("不支持task类型，请检查参数")

def get_backbone(model, task, **kwargs):
    if task == "classify":
        return get_classify_backbone(model)
    elif task == "detect":
        return get_detect_backbone(model, **kwargs)
    else:
        raise RuntimeError("不支持task类型，请检查参数")
    
def get_attention(in_channels, task):
    if task == "classify":
        return get_classify_attention(in_channels)
    elif task == "detect":
        return get_detect_attention(in_channels)
    else:
        raise RuntimeError("不支持task类型，请检查参数")

def find_best_k(lab):
    from sklearn.cluster import KMeans
    from kneed import KneeLocator
    """计算下降减缓的折点"""
    inertia_list = []
    k_list = range(2, 11)
    lab_flat = lab.reshape(-1, 3)
    for k in k_list:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(lab_flat)
        inertia_list.append(kmeans.inertia_)
    # 精确查找肘部
    ks = list(range(1, len(inertia_list) + 1))
    kneedle = KneeLocator(ks, inertia_list, curve="convex", direction="decreasing")
    optimal_k = kneedle.knee
    return optimal_k, k_list, inertia_list

class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, backbone, layer_indices=[4, 6, 9], dropout_r=1e-8):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.backbone = backbone
        self.layer_indices = sorted(layer_indices)
        # 定义dropout层
        for idx in layer_indices: setattr(self, self._dropout_name_(idx), nn.Dropout(p=dropout_r))
    
    def _dropout_name_(self, idx): return f"_layer_{idx}"
    
    def forward(self, x):
        feature_maps = []
        for idx, mod in enumerate(self.backbone):
            x = mod(x)
            if idx in self.layer_indices:
                # 经过dropout层
                if hasattr(self, self._dropout_name_(idx)): x = getattr(self, self._dropout_name_(idx))(x)
                feature_maps.append(x)
        return feature_maps
    
class MultiScaleFeatureAttention(nn.Module):
    def __init__(self, attentions):
        super(MultiScaleFeatureAttention, self).__init__()
        self.attentions = attentions

    def forward(self, features):
        attented_features = []
        for feat, attention in zip(features, self.attentions):
            attented = attention(feat)
            attented_features.append(attented)
        return attented_features
    
class MultiScaleFeatureProjector(nn.Module):
    def __init__(self, projector_builder):
        super(MultiScaleFeatureProjector, self).__init__()
        self.projectors = projector_builder()
        self._initialize_weights_()

    def _initialize_weights_(self):
        for projector in self.projectors:
            for m in projector:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None: nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                    if m.bias is not None: nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                # 只处理nn.Linear
            # 处理下一个投影头
        # 初始化权重结束
    
    def forward(self, features):
        projected_features = []
        for feat, projector in zip(features, self.projectors):
            projected = projector(feat)
            projected_features.append(projected)
        return projected_features

if __name__ == '__main__':
    pass
