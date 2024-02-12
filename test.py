import time
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim

from configure import load_cfg
from model_builder import get_model
from dataset_builder import get_dataset
from criterions import get_criterion
from scheduler_builder import get_scheduler
from utils import (
    resize_image, get_fusion_features, get_device, compute_features)


        

cfg = load_cfg()
num_epochs = 5
dataset = get_dataset(cfg)
models = get_model(cfg, pretrained=False)
device = get_device(cfg)
batch_size = 32

