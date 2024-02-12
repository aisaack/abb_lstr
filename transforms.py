import random

import torch
import cv2
import numpy as np
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize
)
from torchvision.transforms.v2 import Compose, Normalize, RandomCrop, Resize
# import torchvision.transforms.v2.functional as F

    

class CustomRandomCrop:
    def __init__(self, size=[256, 224, 192, 168]):
        self.size = size

    def __call__(self, image):
        sz = random.choice(self.size)
        return RandomCrop(size=sz)(image)


def preprocess(cfg, target):
    if target == 'rgb_kinetics_resnet50':
        return ApplyTransformToKey(
            key='video',
            transform = Compose(
                    [
                        Normalize(cfg.DATA.MEAN, cfg.DATA.STD),
                        # Resize(size=size)
                        # ToTensor(),
                    ]
                )
        )
    elif target == 'flow_kinetics_bninception':
        return ApplyTransformToKey(
            key='video',
            transform = Compose(
                    [
                        Normalize(
                            [sum(cfg.DATA.MEAN) / len(cfg.DATA.MEAN)], [sum(cfg.DATA.STD) / len(cfg.DATA.MEAN)]),
                        # Resize(size=size)
                        # ToTensor(),
                    ]
                )
        )
    
if __name__ == '__main__':
    from configure import load_cfg

    sz = [192, 6, 256, 340]
    v = torch.randn(sz)
    cfg = load_cfg()
    sampler = preprocess(cfg, (256, 256), 'flow_kinetics_bninception')
    # print(type(sampler))
    res = sampler(dict(video=v))
    res = res.get('video')
    print(res.size())