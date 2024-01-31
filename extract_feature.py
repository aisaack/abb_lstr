import os
import cv2
import numpy as np
from scipy import io    # for extract video annotation
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

# from pytorchvideo.data.encoded_video import EncodedVideo

from transforms import GET_Transform
from configure import load_cfg
from model_builder import Resnet, Flownet
from utils import resize_image, get_device


class ExtractDataset(Dataset):
    def __init__(self, cfg, phase='train', target=None, transform=None):
        self.cfg = cfg
        self.tar = target
        self.class_names = self.cfg.DATA.CLASS_NAMES

        def get_meta_file(meta_path, phase):
            key = 'validation' if phase == 'train' else 'test'
            return io.loadmat(meta_path).get(f'{key}_videos')[0]
        
        if phase == 'train':
            self.video_path = os.path.join(self.cfg.DATA.BASE_PATH, self.cfg.DATA.TRAIN_PATH)
            self.meta_file = os.path.join(self.cfg.DATA.BASE_PATH, self.cfg.DATA.VAL_META)
            self.anno_path = os.path.join(self.cfg.DATA.BASE_PATH, self.cfg.DATA.VAL_ANNOTATION)
            self.file_list = self.cfg.DATA.TRAIN_SESSION_SET
            meta_path = os.path.join(self.cfg.DATA.BASE_PATH, self.cfg.DATA.VAL_META)
            

        elif phase == 'test':
            self.video_path = os.path.join(self.cfg.DATA.BASE_PATH, self.cfg.DATA.TEST_PATH)
            self.meta_file = os.path.join(self.cfg.DATA.BASE_PATH, self.cfg.DATA.TEST_META)
            self.anno_path = os.path.join(self.cfg.DATA.BASE_PATH, self.cfg.DATA.TEST_ANNOTATION)
            self.file_list = self.cfg.DATA.TEST_SESSION_SET
            self.file_list.remove('video_test_0000270')     # video_test_0000270 gives me Haircut class.
            meta_path = os.path.join(self.cfg.DATA.BASE_PATH, self.cfg.DATA.TEST_META)
        
        else:
            raise ValueError(f'phase has to be one of [train, test] but {phase} is given')

        # print(type(self.file_list), len(self.file_list))
        print(f'# {self.__len__()} of {phase} dataset are read.')

        self.meta_file = get_meta_file(meta_path, phase)
        self.sampling_rate = self.cfg.DATA.SAMPLING_RATE    # 6
        self.transform = transform
        self.phase = phase

    def __getitem__(self, idx):
        video_file = self.file_list[idx]
        path2file = os.path.join(self.video_path, video_file + '.mp4')
        video_class = self.get_vid_cls(video_file)

        start_end_indices = self.get_action_duration(video_class, video_file)
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
        rgbs = self.get_rgb_sample(video_cap, int(len(video_cap) / self.sampling_rate))
        targets = self.get_target_label(rgbs, start_end_indices, video_class)

        if self.tar == 'target':
            return targets, video_file
        
        rgbs = dict(video=rgbs)
        rgbs = self.transform.process(rgbs)
        rgbs = rgbs['video']

        if self.tar == 'rgb_feature':
            return rgbs, video_file

        flows = self.get_flow_sample(video_cap)
        flows = self.transform.process(flows)
        flows = flows['video']
        flows = torch.cat([flows[::2, :, :, :], flows[1::2, :, :, :]], dim=1)
        # print('rgb size: ', rgbs.size())
        # print('flow size: ', flows.size())
        # print('target size: ', targets.size())
        if self.tar == 'flow_feature':
            return flows, video_file

        if self.tar == None:
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
            # print(type(frame), frame.dtype)
            frame = frame / 255.
            rgb_sam[i] = frame

        rgbs = np.stack(rgb_sam, axis=0)
        rgbs = torch.from_numpy(rgbs).type(torch.float32).permute(0, 3, 1, 2)
        return rgbs
    
    def get_flow_sample(self, video_cap):
        flow_sam = self.get_first_last_frames(video_cap)

        for i, frame in enumerate(flow_sam):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            flow_sam[i] = frame
        
        flows = np.stack(flow_sam, axis=0)
        flows = torch.from_numpy(flows).type(torch.float32).permute(0, 3, 1, 2)
        flows = dict(video=flows)
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
        Returns [[int, int], [int, int], ...]
            timestemps of starting and end of action
        '''
        key = 'val' if self.phase == 'train' else 'test'
        anno_path = os.path.join(self.anno_path, vid_cls + f'_{key}.txt')
        with open(anno_path, 'r') as f:
            anno = f.readlines()

        start_end_idx = []
        for ann in anno:
            anno_list = ann.split(' ')
            if vid_name == anno_list[0]:
                start = float(anno_list[2])
                end = float(anno_list[3].rstrip())
                start_end_idx.append([start, end])
        return start_end_idx
    
    def get_target_label(self, video_frames, start_end_indices, video_class):
        # print("target indices: ", start_end_indices)
        target_label = torch.zeros((len(video_frames), len(self.class_names)))
        # print('target shape: ', target_label.size())
        class_id = self.class_names.index(video_class)
        for indices in start_end_indices:
            start, end = indices
            end_sample = int(end * self.sampling_rate - 1)
            if int(end * 30) % 6 != 0:
                end_sample += 1
            target_label[int(start * self.sampling_rate):end_sample, class_id] = 1
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

def process_flow_feature(model, frame):
    frame = resize_image(frame).contiguous()
    with torch.no_grad():
        feature = model(frame)
        feature = feature.squeeze(-1).squeeze(-1)
    return feature

def process_rgb_feature(model, frame):
    with torch.no_grad():
        feature = model(frame)
    return feature

def main(extract_target='rgb_feature', phase='train'):
    """
    extraction_target: one of {rgb_feature, flow_feature, target_feature}
    """

    print()
    print(f'Start extracting {phase} dataset {extract_target}.')
    print()
    cfg = load_cfg()
    key = 'validation' if phase == 'train' else 'test'
    video_path = f'./dataset/thumos14/{key}'
    target_folder = video_path.replace(key, extract_target)
    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)
        print(f'Create folder at {target_folder}')

    device = get_device(cfg)
    # device = 'cpu'

    if extract_target == 'rgb_feature':
        model = Resnet(cfg)
        # model = DDP(model, device_ids=device_id, output_device=device_id)
    elif extract_target == 'flow_feature':
        model = Flownet(cfg)
        # model = DDP(model, device_ids=deivce_id, output_device=device_id)
    elif extract_target == 'target':
        model = None
    else:
        raise ValueError(f'{extract_target} is not defined. Choose one of [rgb_feature, flow_feature, target]')
    
    if model is not None:
        model.load_ckpt()
        model.eval()
        model = model.to(device)
        model = torch.nn.DataParallel(model)
    dataset = ExtractDataset(cfg, phase, extract_target, GET_Transform(cfg))
    pbar = tqdm(dataset)
    for idx, data in enumerate(pbar):
        
        if data is None:
            continue
        if not extract_target == 'target':
            frames, name = data
            if frames.size(0) > 840:
                frame_chunks = chunk_frames(frames)      # returns tuple of chunks of frame.
                frame_chunks = list(frame_chunks)
                for i, frame in enumerate(frame_chunks):
                    frame = frame.to(device)

                    if extract_target == 'flow_feature':
                        feature = process_flow_feature(model, frame)
                    else:
                        feature = process_rgb_feature(model, frame)

                    frame_chunks[i] = feature
                    del feature, frame
                frame_chunks = torch.cat(frame_chunks, dim=0).detach().cpu().numpy()
                np.save(os.path.join(target_folder, name + '.npy'), frame_chunks)


            else:
                frames = frames.to(device)

                if extract_target == 'flow_feature':
                    feature = process_flow_feature(model, frames)
                else: 
                    feature = process_rgb_feature(model, frames)

                feature = feature.detach().cpu().numpy()
                np.save(os.path.join(target_folder, name + '.npy'), feature)
                torch.cuda.empty_cache()
        
        elif extract_target == 'target':
            target, name = data
            np.save(os.path.join(target_folder, name + '.npy'), target)


if __name__ == '__main__':
    test = False
    phase = 'test'          # one of {train, test}
    target = 'flow_feature'       # one of {rgb_feature, flow_feature, target}

    if test is True:
        cfg = load_cfg()
        dataset = ExtractDataset(cfg, phase, target, GET_Transform(cfg))
        feature, name = dataset[2]
        print(feature.size(), name)

    else:
        main(extract_target=target, phase=phase)

      

