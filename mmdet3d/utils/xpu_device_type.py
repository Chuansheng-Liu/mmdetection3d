# Copyright (c) OpenMMLab. All rights reserved.
import torch

def is_xpu_available():
    try:
        return hasattr(torch, 'xpu') and torch.xpu.is_available()
    except Exception:
        return False

IS_XPU_AVAILABLE = is_xpu_available()
