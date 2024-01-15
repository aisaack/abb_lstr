import os
import cv2
import numpy as np
from scipy import io    # for extract video annotation

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
# from pytorchvideo.data.encoded_video import EncodedVideo

from transforms import GET_Transform


class DataSet(Dataset):
    def __init__(self, cfg, phase='train', transform=None):
        self.cfg = cfg
        self.class_names = self.cfg.DATA.CLASS_NAMES
        self.video_path = os.path.join(self.cfg.DATA.BASE_PATH, self.cfg.DATA.VIDEO_PATH)
        if phase == 'train':
            self.file_list = self.cfg.DATA.TRAIN_SESSION_SET
        elif phase == 'test':
            self.file_list = self.cfg.DATA.TEST_SESSION_SET
        # print(type(self.file_list), len(self.file_list))
        print(f'# {self.__len__()} of {phase} dataset are read.')

        def get_meta_file(meta_path):
            return io.loadmat(meta_path).get('validation_videos')[0]
        
        meta_path = os.path.join(self.cfg.DATA.BASE_PATH, self.cfg.DATA.VAL_META)

        self.meta_file = get_meta_file(meta_path)
        self.anno_path = os.path.join(self.cfg.DATA.BASE_PATH, self.cfg.DATA.ANNOTATION)

        self.sampling_rate = self.cfg.DATA.SAMPLING_RATE    # 6

        self.transform = transform

    def __getitem__(self, idx):
        video_file = self.file_list[idx]
        path2file = os.path.join(self.video_path, video_file + '.mp4')
        video_class = self.get_vid_cls(video_file)
        
        starts, ends = self.get_action_duration(video_class, video_file)
        # print(video_file, video_class, starts, ends)
    
        cap = cv2.VideoCapture(path2file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # total_duration = self.get_video_duration(cap, total_frames)

        video_cap = list()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_cap.append(frame)
        cap.release()

        ditch = total_frames % self.sampling_rate
        if ditch != 0:
            video_cap = video_cap[:-ditch]
        rgbs = self.get_rgb_sample(video_cap, int(total_frames / self.sampling_rate))
        flows = self.get_flow_sample(video_cap)
        targets = self.get_target_label(rgbs, starts, ends, video_class)
         
        rgbs = dict(video=rgbs)
        rgbs = self.transform.rgb(rgbs)
        rgbs = rgbs['video']

        flows = dict(video=flows)
        flows = self.transform.rgb(flows)
        flows = flows['video']
        flows = torch.cat([flows[::2, :, :, :], flows[1::2, :, :, :]], dim=1)
        # print('rgb size: ', rgbs.size())
        # print('flow size: ', flows.size())
        # print('target size: ', targets.size())

        return rgbs, flows, targets
        

    
    def __len__(self):
        return len(self.file_list)
    
    def get_video_duration(self, cap, total_frames):
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        cap.release()
        return duration
    
    def uniform_temporal_subsample(self, video, num_frames):
        total_frames = len(video)
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        return [video[i] for i in indices]
    

    def get_rgb_sample(self, video_cap, num_rgb_sample):
        rgb_sam = self.uniform_temporal_subsample(video_cap, num_rgb_sample)

        for i, frame in enumerate(rgb_sam):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_sam[i] = frame

        rgbs = np.stack(rgb_sam, axis=0)
        rgbs = torch.from_numpy(rgbs).type(torch.float32)
        rgbs = rgbs.permute(0, 3, 1, 2)

        return rgbs
    
    def get_flow_sample(self, video_cap):
        flow_sam = self.get_first_last_frames(video_cap)

        for i, frame in enumerate(flow_sam):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            flow_sam[i] = frame
        
        flows = np.stack(flow_sam, axis=0)
        flows = torch.from_numpy(flows).type(torch.float32)
        flows = flows.permute(0, 3, 1, 2)

        return flows

    
    def get_first_last_frames(self, video):
        num_frames = len(video)
        def get_first_last_frames_indices(num_frames):
            indices = []
            for i in range(0, num_frames, self.sampling_rate):
                indices.append(i)
                indices.append(i+self.sampling_rate-1)
            return np.clip(np.array(indices, dtype=np.int_), 0, num_frames-1)
        indices = get_first_last_frames_indices(num_frames)        
        return [video[i] for i in indices]
    
    def get_vid_cls(self, vid_name):
        for meta in self.meta_file:
            if meta[0] == vid_name:
                vid_cls = meta[2][0]
                return vid_cls
            
    def get_action_duration(self, vid_cls, vid_name):
        '''
        Returns [int], [int]
            timestemps of starting and end of action
        '''
        anno_path = os.path.join(self.anno_path, vid_cls + '_val.txt')
        with open(anno_path, 'r') as f:
            anno = f.readlines()

        starts = []
        ends = []
        for ann in anno:
            anno_list = ann.split(' ')
            if vid_name == anno_list[0]:
                start = float(anno_list[2])
                end = float(anno_list[3].rstrip())
                starts.append(start)
                ends.append(end)
        
        return starts, ends
    
    def get_target_label(self, video_frames, starts, ends, video_class):
        target_label = torch.zeros((len(video_frames), len(self.class_names)))
        class_id = self.class_names.index(video_class)
        for start, end in zip(starts, ends):
            start = start * (self.sampling_rate-1)
            end = end * (self.sampling_rate-1)
            target_label[int(start):int(end)+1, class_id] = 1
        return target_label
    
    @staticmethod
    def get_video_duration(video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        return duration

    @staticmethod
    def uniform_temporal_subsample(video, num_frames):
        total_frames = len(video)
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        return [video[i] for i in indices]
    
    

def get_dataset(cfg, phase):
    '''Return pytorch dataset object'''
    transform = GET_Transform(cfg)
    return DataSet(cfg, phase, transform)



if __name__ == '__main__':
    from configure import load_cfg
    from utils import compute_features, resize_image
    from model_builder import get_model

    cfg = load_cfg()
    models = get_model(cfg, pretrained='extractor')
    datasets = get_dataset(cfg, 'train')
    rgb, flow, target = datasets[100]
    print(rgb.size(), flow.size(), target.size())