import os
import cv2
import numpy as np
from scipy import io    # for extract video annotation
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# from pytorchvideo.data.encoded_video import EncodedVideo

from transforms import preprocess
from configure import load_cfg
from model_builder import Resnet, Flownet
from utils import get_device


def resize_image(img, multiplier):
    h, w = img.shape[2:]
    new_h = int(multiplier * h) + 1
    new_w = int(multiplier * w) + 1
    if new_h % 64 != 0:
        remain = new_h % 64
        new_h -= remain
    
    if new_w  % 64 != 0:
        reamin = new_w % 64
        new_w -= reamin

    return F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)


class ExtractDataset(Dataset):
    def __init__(self, cfg, phase='train', target=None):
        self.cfg = cfg
        self.tar = target
        self.class_names = self.cfg.DATA.CLASS_NAMES
        self.fps = self.cfg.DATA.NUM_FRAMES
        self.rgb_chunk_size = cfg.DATA.RGB_CHUNK_SIZE
        self.flo_chunk_size = cfg.DATA.FLOW_CHUNK_SIZE

        def get_meta_file(meta_path, phase):
            key = 'validation' if phase == 'train' else 'test'
            return io.loadmat(meta_path).get(f'{key}_videos')[0]
        
        if phase == 'train':
            self.video_path = os.path.join(self.cfg.DATA.BASE_PATH, self.cfg.DATA.TRAIN_PATH)
            self.meta_file = os.path.join(self.cfg.DATA.BASE_PATH, self.cfg.DATA.VALIDATION_META)
            self.anno_path = os.path.join(self.cfg.DATA.BASE_PATH, self.cfg.DATA.VALIDATION_ANNOTATION)
            self.file_list = self.cfg.DATA.TRAIN_SESSION_SET
            meta_path = os.path.join(self.cfg.DATA.BASE_PATH, self.cfg.DATA.VALIDATION_META)
            

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
        self.phase = phase

    def __getitem__(self, idx):
        video_file = self.file_list[idx]
        path2file = os.path.join(self.video_path, video_file + '.mp4')
         
        cap = cv2.VideoCapture(path2file)
        video_cap = self.extract_frames(cap)
        video_cap = self.uniform_temporal_subsample(video_cap)

        if self.tar == 'rgb_kinetics_resnet50':
            rgbs = self.get_rgb_sample(video_cap)
            rgbs = dict(video=rgbs)
            rgbs = preprocess(self.cfg, self.tar)(rgbs)
            rgbs = rgbs.get('video')
            cap.release()
            return rgbs, video_file
        
        if self.tar == 'flow_kinetics_bninception':
            flows = self.get_flow_sample(video_cap)
            flows = dict(video=flows)
            flows = preprocess(self.cfg, self.tar)(flows)
            flows = flows.get('video')
            cap.release()
            return flows, video_file
        
        if self.tar == 'target':
            rgbs = self.get_rgb_sample(video_cap)
            video_class = self.get_vid_cls(video_file)
            start_end_indices = self.get_action_duration(video_class, video_file)
            # rgbs = self.get_rgb_sample(video_cap, int(total_frames / self.sampling_rate))
            targets = self.get_target_label(rgbs, start_end_indices, video_class)
            cap.release()
            return targets, video_file

        if self.tar == None:
            return rgbs, flows, targets, video_file
            
    def __len__(self):
        return len(self.file_list)
    
    def extract_frames(self, caputure):
        video_cap = list()
        while True:
            ret, frame = caputure.read()
            if not ret:
                break
            video_cap.append(frame)
        return video_cap
    

    def uniform_temporal_subsample(self, video):
        """
        Sample 30 fps video into 24 fps video.
        """
        total_frames = len(video)
        indices = np.linspace(0, total_frames - 1, round(total_frames / 30) * self.fps).astype(np.compat.long)
        indices = np.clip(indices, 0, total_frames - 1)
        return [video[i] for i in indices]
    

    def get_rgb_sample(self, sampled_frame):
        indices = np.arange(2, len(sampled_frame), self.rgb_chunk_size)
        sampled_frame = [sampled_frame[idx] for idx in indices]
        for i, frame in enumerate(sampled_frame):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sampled_frame[i] = frame

        rgbs = np.stack(sampled_frame, axis=0)
        rgbs = torch.as_tensor(rgbs, dtype=torch.uint8)
        rgbs = torch.permute(rgbs, (0, 3, 1, 2))
        return rgbs / 255.
    
    def get_flow_sample(self, sampled_frame):
        total_frames = len(sampled_frame)
        ignore_idx = np.arange(5, total_frames-1, self.flo_chunk_size)

        for i, frame in enumerate(sampled_frame):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sampled_frame[i] = frame
        
        flows = np.stack(sampled_frame, axis=0)
        flows = np.concatenate([flows[:-1, ...], flows[1:, ...]], axis=-1)
        flows = np.delete(flows, ignore_idx, axis=0)
        flows = torch.as_tensor(flows, dtype=torch.uint8)
        flows = torch.permute(flows, (0, 3, 1, 2))
        return flows / 255.

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
        target_label = torch.zeros((len(video_frames), len(self.class_names)))
        class_id = self.class_names.index(video_class)
        for indices in start_end_indices:
            start, end = indices
            multiplyer = self.fps / self.sampling_rate
            frame_start, frame_end = int(start * multiplyer), int(end * multiplyer)
            target_label[frame_start:frame_end, class_id] = 1
        return target_label

def chunk_frames(frames, chunk_size):
    num_chunk = frames.size(0) // chunk_size
    if num_chunk == 0:
        return frames
    return frames.chunk(num_chunk, dim=0)

def process_flow_feature(model, frame, div_size, device):
    frame = resize_image(frame, div_size)
    if frame.size(0) > 840:
        frame_chunks = chunk_frames(frame, 840)
        frame_chunks = list(frame_chunks)
        for idx, chunk in enumerate(frame_chunks):
            chunk = chunk.to(device)
            with torch.no_grad():
                feature = model(chunk.contiguous())
                del chunk
            frame_chunks[idx] = feature
            
        return frame_chunks
    
    else:
        frame = frame.to(device)
        with torch.no_grad():
            feature = model(frame)
        return feature

def process_rgb_feature(model, frame, div_size, device):
    frame = resize_image(frame, div_size)
    if frame.size(0) > 840:
        frame_chunks = chunk_frames(frame, 840)
        frame_chunks = list(frame_chunks)
        for idx, chunk in enumerate(frame_chunks):
            chunk = chunk.to(device)
            with torch.no_grad():
                feature = model(chunk)
                del chunk
            frame_chunks[idx] = feature
            
        return frame_chunks
    
    else:
        frame = frame.to(device)
        with torch.no_grad():
            feature = model(frame)
        return feature

    

def main(multiplier, extract_target='rgb_kinetics_resnet50', phase='train'):
    """
    extraction_target: one of {rgb_kinetics_resnet50, flow_kinetics_bninception, target}
    """
    h, w = 180, 320
    new_h = int(multiplier * h) + 1
    new_w = int(multiplier * w) + 1
    if new_h % 64 != 0:
        remain = new_h % 64
        new_h -= remain
    
    if new_w  % 64 != 0:
        reamin = new_w % 64
        new_w -= reamin

    cfg = load_cfg()
    print()
    print(f'Start extracting {phase} dataset {extract_target} {new_h, new_w} sized video in {cfg.DATA.NUM_FRAMES} fps.')
    print()
    key = 'validation' if phase == 'train' else 'test'
    video_path = f'./dataset/thumos14/{key}'
    target_folder = video_path.replace(key, extract_target)
    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)
        print(f'Create folder at {target_folder}')

    device = get_device(cfg)
    # device = 'cpu'

    if extract_target == 'rgb_kinetics_resnet50':
        model = Resnet(cfg)
        # model = DDP(model, device_ids=device_id, output_device=device_id)
    elif extract_target == 'flow_kinetics_bninception':
        model = Flownet(cfg)
        # model = DDP(model, device_ids=deivce_id, output_device=device_id)
    elif extract_target == 'target':
        model = None
    else:
        raise ValueError(f'{extract_target} is not defined. Choose one of [rgb_kinetics_resnet50, flow_kinetics_bninception, target]')
    
    if model is not None:
        model.load_ckpt()
        model.eval()
        model = model.to(device)
        model = torch.nn.DataParallel(model)
    dataset = ExtractDataset(cfg, phase, extract_target)
    pbar = tqdm(dataset)
    for idx, data in enumerate(pbar):
        if data is None:
            continue
        if not extract_target == 'target':
            frames, name = data

            if extract_target == 'rgb_kinetics_resnet50':
                feature = process_rgb_feature(model, frames, multiplier, device)
            elif extract_target == 'flow_kinetics_bninception':
                feature = process_flow_feature(model, frames, multiplier, device)

            feature = feature.detach().cpu().numpy()
            np.save(os.path.join(target_folder, name + '.npy'), feature)
            torch.cuda.empty_cache()
        
        elif extract_target == 'target':
            target, name = data
            np.save(os.path.join(target_folder, name + '.npy'), target)


if __name__ == '__main__':
    test = False
    phase = 'train'          # one of {train, test}
    target = 'flow_feature'       # one of {rgb_feature, flow_feature, target}
    muliplier = 1.3
    # mult  resolution  hight/width
    # 1.3 > (192, 384): 0.5
    # 1.4 > (192, 448): 0.42
    # 1.5 > (256, 448): 0.57    OOM happens when extracting flow feature with this.

    if target == 'flow_feature':
        target = 'flow_kinetics_bninception'
    elif target == 'rgb_feature':
        target = 'rgb_kinetics_resnet50'
    elif target == 'target':
        pass
    else:
        raise ValueError('Undefined.')

    if test is True:
        cfg = load_cfg()
        dataset = ExtractDataset(cfg, size, phase, target)
        feature, name = dataset[3]
        print(feature.size(), name)
        # model = Flownet(cfg)
        # out = model(feature.unsqueeze(0))
        # print(out.size(), name)

    else:
        main(extract_target=target, phase=phase, multiplier=muliplier)

      

