import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 为了更稳定复现；若你追求极致速度可调整
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False