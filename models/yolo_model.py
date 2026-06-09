import os
import sys

# =========================================================
# Path injection
# =========================================================
# 说明：
# 1. ULTRALYTICS_ROOT：本地 ultralytics 仓库根目录，请按你的实际路径修改
# =========================================================
ULTRALYTICS_ROOT = r"E:\ultralytics"

if os.path.isdir(ULTRALYTICS_ROOT) and ULTRALYTICS_ROOT not in sys.path:
    sys.path.insert(0, ULTRALYTICS_ROOT)

from ultralytics import YOLO   # type: ignore

import warnings
warnings.filterwarnings(action="ignore", category=UserWarning)

__all__ = [
    "YOLO",
]