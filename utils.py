import os
import numpy as np
from bisect import bisect_right

import torch
import torch.optim as optim
import torch.nn.functional as F

# from dataset_builder import DataSet
from transforms import GET_Transform
from model_builder import Lstr, Resnet, Flownet


def get_device(cfg):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'We have {torch.cuda.device_count()} GPU(s) available')
        return device
    device = torch.device('cpu')
    print('No GPU available. Using CPU instead')
    return device


# def get_dataloader(cfg, phase):
#     '''
#     initialize dataset object and call dataloader
#     '''
#     cfgs = cfg.DATA_LOADER
#     dataset = get_dataset(cfg, phase)
#     return DataLoader(
#         dataset,
#         batch_size = cfgs.BATCH_SIZE,
#         num_workers = cfgs.NUM_WORKERS,
#         pin_memory = cfgs.PIN_MEMORY
#         )

def resize_image(image, div_size=128):
    height, width = image.size(-2), image.size(-1)
    if height % div_size != 0 or width % div_size != 0:
        input_size = (
            int(div_size * np.ceil(height / div_size)), 
            int(div_size * np.ceil(width / div_size))
        )
    else:
        input_size = (height, width)
    return F.interpolate(image, size=input_size, mode='bilinear', align_corners=False)


def compute_features(rgb, flow, target, model, work_memory_length, train_extractor=False):
  
    if train_extractor is False:
        with torch.no_grad():
            rgb_feature = model['resnet'](rgb)
            flow_feature = model['flownet'](flow)
            flow_feature = flow_feature.squeeze(-1).squeeze(-1)
    else:
        model['resnet'].train(train_extractor)
        model['flownet'].train(train_extractor)
        rgb_feature = model['resnet'](rgb)
        flow_feature = model['flownet'](flow)
        flow_feature = flow_feature.squeeze(-1).squeeze(-1)

    helper = []
    seed = torch.randint(work_memory_length, size=(1,))
    for work_start, work_end in zip(
        range(seed, target.size(0), work_memory_length),
        range(seed + work_memory_length, target.size(0), work_memory_length)
    ):
        helper.append([work_start, work_end, target[work_start:work_end]])
    
    return rgb_feature, flow_feature, helper


def segment_sampler(start, end, num_samples):
    indices = torch.linspace(start, end, num_samples).type(torch.int32)
    indices, _ = torch.sort(indices)
    return indices

def uniform_sampler(start, end, num_samples, sample_rate):
    indices = torch.arange(start, end + 1)[::sample_rate]
    padding = num_samples - indices.shape[0]
    if padding > 0:
        indices = torch.cat((torch.zeros(padding), indices))
    indices = indices.type(torch.int32)
    indices, _ = torch.sort(indices)
    return indices

def get_fusion_features(cfg, rgb_feature:list, flow_feature:list, helpers:list, training=True):
    # print(rgb_feature.size(), flow_feature.size())
        
    fusion_rgbs, fusion_flows, fusion_targets = [], [], []
    memory_key_padding_masks = []
    for data in helpers:
        # print(i)
        work_start, work_end, target = data
        # print(work_start, work_end)
        fusion_target = target[::cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE]
        work_indices = torch.arange(work_start, work_end).clip(0)
        work_indices = work_indices[::cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE]
        # print('work indices', work_indices)
        work_rgb = rgb_feature[work_indices]
        work_flow = flow_feature[work_indices]
        # print('work: ', work_rgb.size(), work_flow.size())
        if cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE > 0:
            long_start, long_end = max(0, work_start - cfg.MODEL.LSTR.LONG_MEMORY_LENGTH), work_start - 1
            if training is True:
                long_indices = segment_sampler(long_start, long_end, cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES)
            else:
                long_indices = uniform_sampler(
                        long_start, long_end, cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES, cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE).clip(0)
            long_rgb = rgb_feature[long_indices]
            long_flow = flow_feature[long_indices]
            # print('long: ', long_rgb.size(), long_flow.size())
            memory_key_padding_mask = torch.zeros(long_indices.size(0))     # [512] sized torch tensor filled with 0
            last_zero = bisect_right(long_indices.numpy(), 0) - 1
            # print('last_zero: ', last_zero)
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')
        
        else:
            long_rgb = None
            long_flow = None
            memory_key_padding_mask = None

        if long_rgb is not None and long_flow is not None:
            # print('Long feature exists')
            # print(work_rgb.size(), long_rgb.size(), work_flow.size(), long_flow.size())
            fusion_rgb = torch.cat([long_rgb, work_rgb], dim=0)
            fusion_flow = torch.cat([long_flow, work_flow], dim=0)
        else:
            # print('No long feature')
            fusion_rgb = work_rgb
            fusion_flow = work_flow
        
        fusion_rgbs.append(fusion_rgb)
        fusion_flows.append(fusion_flow)
        fusion_targets.append(fusion_target)
        
        if memory_key_padding_mask is not None:
            memory_key_padding_masks.append(memory_key_padding_mask.type(torch.float32))
        # else:
        #     memory_key_padding_masks.append(0)
    if len(memory_key_padding_masks) > 0:
        return fusion_rgbs, fusion_flows, memory_key_padding_masks, fusion_targets
    return fusion_rgbs, fusion_flows, fusion_targets


def infer_compute_features(rgb, flow, target, model, work_memory_length):
    
    with torch.no_grad():
        rgb_feature = model['resnet'](rgb)
        flow_feature = model['flownet'](flow)
        flow_feature = flow_feature.squeeze(-1).squeeze(-1)

    helper = []
    for work_start, work_end in zip(
        range(0, target.size(0) + 1),
        range(work_memory_length, target.size(0) + 1)
    ):
        helper.append([work_start, work_end, target[work_start:work_end], target.size(0)])
    
    return rgb_feature, flow_feature, helper


def infer_get_fusion_features(cfg, rgb_feature:list, flow_feature:list, helpers:list):

    fusion_rgbs, fusion_flows, fusion_targets = [], [], []
    memory_key_padding_masks = []

    for data in helpers:
        work_start, work_end, target, num_frames = data

        fusion_target = target[::cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE]

        work_indices = torch.arange(work_start, work_end).clip(0)
        work_indices = work_indices[::cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE]
        work_rgb = rgb_feature[work_indices]
        work_flow = flow_feature[work_indices]

        if cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES > 0:
            long_start, long_end = max(0, work_start - cfg.MODEL.LSTR.LONG_MEMORY_LENGTH), work_start - 1
            long_indices = uniform_sampler(long_start, long_end, cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES, cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE).clip(0)
            long_rgb = rgb_feature[long_indices]
            long_flow = flow_feature[long_indices]

            memory_key_padding_mask = torch.zeros(long_indices.size(0))
            last_zero =  bisect_right(long_indices.numpy(), 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')
        
        else:
            long_rgb = None
            long_flow = None
            memory_key_padding_mask = None

        if long_rgb is not None and long_flow is not None:
            fusion_rgb = torch.cat([long_rgb, work_rgb], dim=0)
            fusion_flow = torch.cat([long_flow, work_flow], dim=0)
        else:
            fusion_rgb = work_rgb
            fusion_flow = work_flow

        fusion_rgbs.append(fusion_rgb)
        fusion_flows.append(fusion_flow)
        fusion_targets.append(fusion_target)

        if memory_key_padding_mask is not None:
            memory_key_padding_masks.append(memory_key_padding_mask.type(torch.float32))

    if len(memory_key_padding_masks) > 0:
        return [fusion_rgbs, fusion_flows, memory_key_padding_masks, fusion_targets, work_indices, num_frames]
    return [fusion_rgbs, fusion_flows, fusion_targets, work_indices, num_frames]



if __name__ == '__main__':
    import torch
    from dataset_builder import get_dataset
    from configure import load_cfg

    cfg = load_cfg()

    dataset = get_dataset(cfg, 'train')
    rgb, flow, target = dataset[4]
    helper = []
    work_memory_length=32
    seed = torch.randint(work_memory_length, size=(1,))
    for work_start, work_end in zip(
        range(seed, target.size(0), work_memory_length),
        range(seed + work_memory_length, target.size(0), work_memory_length)
    ):
        helper.append([work_start, work_end, target[work_start:work_end]])


    print(len(helper))

    
    