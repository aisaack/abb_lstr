import os
import cv2
import numpy as np
from scipy import io    # for extract video annotation
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
# from pytorchvideo.data.encoded_video import EncodedVideo

from transforms import GET_Transform
from configure import load_cfg
from model_builder import get_model
from model_builder import Resnet, Flownet
from utils import resize_image, get_device


class ExtractDataset(Dataset):
    def __init__(self, cfg, phase='train', transform=None, mode=None):
        self.cfg = cfg
        self.mode = mode
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
        if self.mode == 'target':
            return targets, video_file
         
        rgbs = dict(video=rgbs)
        rgbs = self.transform.rgb(rgbs)
        rgbs = rgbs['video']

        if self.mode == 'rgb_feature':
            return rgbs, video_file

        flows = dict(video=flows)
        flows = self.transform.rgb(flows)
        flows = flows['video']
        flows = torch.cat([flows[::2, :, :, :], flows[1::2, :, :, :]], dim=1)
        # print('rgb size: ', rgbs.size())
        # print('flow size: ', flows.size())
        # print('target size: ', targets.size())
        if self.mode == 'flow_feature':
            return flows, video_file

        if self.mode == None:

            return rgbs, flows, targets, video_file
        

    
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


def chunk_frames(frames):
    num_chunk = frames.size(0) // 840
    if num_chunk == 0:
        return frames
    return frames.chunk(num_chunk, dim=0)

def main(extract_target='rgb_feature', phase='train'):
    """
    extraction_target: one of {rgb_feature, flow_feature, target_feature}
    """

    print()
    print(f'Start extracting {extract_target}.')
    print()
    cfg = load_cfg()
    video_path = './dataset/thumos14/validation'
    target_folder = video_path.replace('validation', extract_target)

    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)
        print(f'Create folder at {target_folder}')

    device = get_device(cfg)

    if extract_target == 'rgb_feature':
        model = Resnet(cfg)
        model.load_ckpt()
        # model = DDP(model, device_ids=device_id, output_device=device_id)
    elif extract_target == 'flow_feature':
        model = Flownet(cfg)
        model.load_ckpt()
        # model = DDP(model, device_ids=deivce_id, output_device=device_id)
    elif extract_target == 'target':
        model = None
    else:
        raise ValueError(f'{extract_target} is not defined. Choose one of [rgb_feature, flow_feature, target]')

    if model is not None:
        model = torch.nn.DataParallel(model)
        model = model.to(device)

    dataset = ExtractDataset(cfg, phase, GET_Transform(cfg), mode=extract_target)
    pbar = tqdm(dataset)
    for idx, data in enumerate(pbar):
        
        if not extract_target == 'target':
            frames, name = data
            if frames.size(0) > 840:
                frame_chunks = chunk_frames(frames)      # returns list of chunks of frame.
                frame_chunks = list(frame_chunks)
                for i, frame in enumerate(frame_chunks):
                    frame = frame.to(device)
                    if extract_target == 'flow_feature':
                        frame = resize_image(frame).contiguous()
                    with torch.no_grad():
                        feature = model(frame)
                        if extract_target == 'flow_feature':
                            feature = feature.squeeze(-1).squeeze(-1)
                    frame_chunks[i] = feature
                    del feature, frame
                frame_chunks = torch.cat(frame_chunks, dim=0).detach().cpu().numpy()
                np.save(os.path.join(target_folder, name + '.npy'), frame_chunks)
            else:
                frames = frames.to(device)
                if extract_target == 'flow_feature':
                    frames = resize_image(frames).contiguous()
                with torch.no_grad():
                    feature = model(frames)
                    if extract_target == 'flow_feature':
                        feature = feature.squeeze(-1).squeeze(-1)
                feature = feature.detach().cpu().numpy()
                np.save(os.path.join(target_folder, name + '.npy'), feature)
            torch.cuda.empty_cache()
        
        elif extract_target == 'target':
            target, name = data
            np.save(os.path.join(target_folder, name + '.npy'), target)

if __name__ == '__main__':
    phase = 'test'
    target = 'rgb_feature' # one of {rgb_feature, flow_feature, target}
    main(extract_target=target, phase=phase)

      

