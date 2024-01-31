import torch
import cv2
import numpy as np
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize
)
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.transforms.functional as F

class SelectFirstAndLastFrames:
    '''
    Returns the first and last frames of each group of sampling_rate frames.
    Ex: sample_rate = 6, num_frames = 192 -> 0,5 6,11 12,17 18,23 ... 186,191
    '''
    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def __call__(self, video):
        indices = self.get_first_last_frames_indices(video.shape[0], self.sampling_rate)
        return torch.index_select(video, 0, indices)
    
    def get_first_last_frames_indices(self, num_frames, sampling_rate):
        indices = []
        for i in range(0, num_frames, sampling_rate):
            indices.append(i)
            indices.append(i+sampling_rate-1)
        return torch.clamp(torch.tensor(indices, dtype=torch.long), 0, num_frames-1)
    



# class UnifromTemporalUpsample:
#     def __init__(self, num_frames):
#         self.num_frames = num_frames
    
#     def __call__(self, video:list):
#         total_frames = len(video)
#         indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
#         return [video[i] for i in indices]


# class ShortSideScale:
#     def __init__(self, short_siede_length):
#         self.short_side = short_siede_length

#     def __call__(self, image):
#         _, _, h, w = image.size()
#         aspect_ratio = w / h

#         if h < w:
#             new_h = self.short_side
#             new_w = int(new_h * aspect_ratio)
#         else:
#             new_w = self.short_side
#             new_h = int(new_w / aspect_ratio)

#         resized_image = F.resize(image, (new_h, new_w))
#         return resized_image


class GET_Transform:
    def __init__(self, cfg):
        self.cfg = cfg
        # self.flow = self.get_flow_transform()
        self.process = self.get_rgb_transform()
        # print(self.cfg.DATA.MEAN)


    def get_rgb_transform(self):
        return ApplyTransformToKey(
            key='video',
            transform = Compose(
                    [
                        Normalize(self.cfg.DATA.MEAN, self.cfg.DATA.STD),
                        # ToTensor(),
                    ]
                )
        )
    
if __name__ == '__main__':
    sz = [192, 3, 50, 50]
    v = torch.randn(sz)
    sampler = SelectFirstAndLastFrames(6)
    res = sampler(v)
    print(res.size())