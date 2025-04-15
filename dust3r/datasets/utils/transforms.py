# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUST3R default transforms
# --------------------------------------------------------
import math
import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision.transforms as tvf
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

# define the standard image transforms
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# ColorJitter = tvf.Compose([tvf.ColorJitter(0.5, 0.5, 0.5, 0.1), ImgNorm])
# PhotometricTransforms = tvf.Compose(
#     [
#         tvf.ColorJitter(0.5, 0.5, 0.5, 0.1),
#         tvf.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
#         ImgNorm
#     ]
# )

ImgNormAlb = A.Compose(
    [
        # A.LongestMaxSize(max_size=512, interpolation=1, p=1.0),
        # A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, fill=0, p=1.0),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
)