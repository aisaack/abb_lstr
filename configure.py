import argparse
import os.path as osp
import argparse
import json


class LSTR:
    def __init__(self):
        self.IS_STREAM = False
        self.NUM_HEADS = 16
        self.DIM_FEEDFORWARD = 1024
        self.DROPOUT = 0.2
        self.ACTIVATION = 'relu'
        self.AGES_MEMORY_SECONDS = 0
        self.AGES_MEMORY_SAMPLE_RATE = 1
        self.LONG_MEMORY_SECONDS = 512
        self.LONG_MEMORY_SAMPLE_RATE = 4
        self.WORK_MEMORY_SECONDS = 8
        self.WORK_MEMORY_SAMPLE_RATE = 1
        self.ENC_MODULE = [[16, 1, True], [32, 2, True]]
        self.DEC_MODULE = [-1, 2, True]
        self.INFERENCE_MODE = 'batch'
        self.CKPT = None #'./checkpoints/lstr_th14_long_512_work_8_box.epoch-25.pth'


class RESNET:
    def __init__(self):
        self.DEPTH = 50
        if self.DEPTH == 50:
            self.LAYERS = [3, 4, 6, 3]
        self.CKPT = './checkpoints/resnet50.pth'


class FAST_FLOWNET:
    def __init__(self):
        self.CKPT = './checkpoints/fastflownet_ft_mix.pth'


class BN_INCEPTION:
    def __init__(self):
        self.IN_CHANNELS = 10
        self.CKPT = './checkpoints/flow_bninception.pth'


class COMBINED_FLOWNET:
    def __init__(self):
        self.FAST_FLOWNET = FAST_FLOWNET()
        self.BN_INCEPTION = BN_INCEPTION()


class FEATURE_HEAD:
    def __init__(self):
        self.LINEAR_ENABLED = True
        self.LINEAR_OUT_FEATURES = 1024


class MODEL:
    def __init__(self):
        self.MODEL_NAME = 'LSTR'
        self.FEATURE_HEAD = FEATURE_HEAD()
        self.LSTR = LSTR()
        self.FLOWNET = COMBINED_FLOWNET()
        self.RESNET = RESNET()
        self.BN_INCEPTION = BN_INCEPTION()
        self.CRITERIONS = [['MCE', {}]]
        self.PRETRAINED = True
        

class INPUT:
    def __init__(self):
        self.MODALITY = 'twostream'
        self.VISUAL_FEATURE = 'rgb_feature'
        self.MOTION_FEATURE = 'flow_feature'
        self.TARGET_PERFRAME = 'target'


class DATA_LOADER:
    def __init__(self):
        self.BATCH_SIZE = 16
        self.NUM_WORKERS = 8
        self.PIN_MEMORY = True


class SCHEDULAR:
    def __init__(self):
        self.SCHEDULER_NAME = 'warmup_cosine'
        self.WARMUP_FACTOR = 0.3
        self.WARMUP_EPOCHS = 10.0
        self.WARMUP_METHOD = 'linear'
        self.GAMMA = 0.1

class SOLVER:
    def __init__(self):
        self.NUM_EPOCHS = 50
        self.OPTIMIZER = 'adam'
        self.BASE_LR = 7e-05
        self.WEIGHT_DECAY = 5e-05
        self.SCHEDULER = SCHEDULAR()
        self.MOMENTUM = None            # for sgd
        self.PHASES = ['train', 'test']
        self.START_EPOCH = 1
        self.RESUME = False


class DATA:
    def __init__(self):
        self.DATA_NAME = 'THUMOS'
        self.DATA_INFO = './data/data_info.json'
        self.BASE_PATH = './dataset/thumos14'
        self.TRAIN_PATH = 'validation'
        self.TEST_PATH = 'test'
        self.VAL_META = self.TRAIN_PATH + '_meta/validation_set.mat'
        self.TEST_META = self.TEST_PATH + '_meta/test_set_meta.mat'
        self.VAL_ANNOTATION = self.TRAIN_PATH + '_anno/'
        self.TEST_ANNOTATION = self.TEST_PATH + '_anno/'
        self.CLASS_NAMES = None
        self.NUM_CLASSES = 22
        # assert len(self.CLASS_NAMES) == self.NUM_CLASSES

        self.IGNORE_INDEX = None
        self.METRICS = None
        self.FPS = None
        self.TRAIN_SESSION_SET = None
        self.TEST_SESSION_SET = None
        self.SAMPLING_RATE = 6
        self.NUM_FRAMES = 30
        self.MEAN = [0.485, 0.456, 0.406] #[123.675, 116.28, 103.53]
        self.STD = [0.229, 0.224, 0.225] #[58.395, 57.12, 57.375]
        

class Config:
    def __init__(self):
        self.NAME = 'lstr_distil'
        self.GPU = '0'
        self.DATA = DATA()
        self.INPUT = INPUT()
        self.MODEL = MODEL()
        self.DATA_LOADER = DATA_LOADER()
        self.SOLVER = SOLVER()
        self.OUTPUT_DIR = './checkpoints'
        self.VERBOSE = True
        self.SESSION = ''

def parse_args():
    parser = argparse.ArgumentParser(description='Rekognition Online Action Detection')
    parser.add_argument(
        '--config_file',
        default='',
        type=str,
        help='path to yaml config file',
    )
    parser.add_argument(
        '--gpu',
        default='0',
        type=str,
        help='specify visible devices'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs='*',
        help='modify config options using the command-line',
    )
    return parser.parse_args()

def assert_and_infer_cfg(cfg):
    # Setup the visible devices

    # Infer data info
    with open(cfg.DATA.DATA_INFO, 'r') as f:
        data_info = json.load(f)[cfg.DATA.DATA_NAME]

    cfg.DATA.BASE_PATH = data_info['data_root'] if cfg.DATA.BASE_PATH is None else cfg.DATA.BASE_PATH
    cfg.DATA.CLASS_NAMES = data_info['class_names'] if cfg.DATA.CLASS_NAMES is None else cfg.DATA.CLASS_NAMES
    cfg.DATA.NUM_CLASSES = data_info['num_classes'] if cfg.DATA.NUM_CLASSES is None else cfg.DATA.NUM_CLASSES
    cfg.DATA.IGNORE_INDEX = data_info['ignore_index'] if cfg.DATA.IGNORE_INDEX is None else cfg.DATA.IGNORE_INDEX
    cfg.DATA.METRICS = data_info['metrics'] if cfg.DATA.METRICS is None else cfg.DATA.METRICS
    cfg.DATA.FPS = data_info['fps'] if cfg.DATA.FPS is None else cfg.DATA.FPS
    cfg.DATA.TRAIN_SESSION_SET = data_info['train_session_set'] if cfg.DATA.TRAIN_SESSION_SET is None else cfg.DATA.TRAIN_SESSION_SET
    cfg.DATA.TEST_SESSION_SET = data_info['test_session_set'] if cfg.DATA.TEST_SESSION_SET is None else cfg.DATA.TEST_SESSION_SET

    # Ignore two mis-labeled videos
    if False and cfg.DATA_NAME == 'THUMOS':
        cfg.DATA.TEST_SESSION_SET.remove('video_test_0000270')
        cfg.DATA.TEST_SESSION_SET.remove('video_test_0001496')

    # Input assertions
    assert cfg.INPUT.MODALITY in ['visual', 'motion', 'twostream']

    # Infer memory
    if cfg.MODEL.MODEL_NAME in ['LSTR']:
        cfg.MODEL.LSTR.AGES_MEMORY_LENGTH = cfg.MODEL.LSTR.AGES_MEMORY_SECONDS * cfg.DATA.FPS
        cfg.MODEL.LSTR.LONG_MEMORY_LENGTH = cfg.MODEL.LSTR.LONG_MEMORY_SECONDS * cfg.DATA.FPS
        cfg.MODEL.LSTR.WORK_MEMORY_LENGTH = cfg.MODEL.LSTR.WORK_MEMORY_SECONDS * cfg.DATA.FPS
        cfg.MODEL.LSTR.TOTAL_MEMORY_LENGTH = \
            cfg.MODEL.LSTR.AGES_MEMORY_LENGTH + \
            cfg.MODEL.LSTR.LONG_MEMORY_LENGTH + \
            cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        assert cfg.MODEL.LSTR.AGES_MEMORY_LENGTH % cfg.MODEL.LSTR.AGES_MEMORY_SAMPLE_RATE == 0
        assert cfg.MODEL.LSTR.LONG_MEMORY_LENGTH % cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE == 0
        assert cfg.MODEL.LSTR.WORK_MEMORY_LENGTH % cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE == 0
        cfg.MODEL.LSTR.AGES_MEMORY_NUM_SAMPLES = cfg.MODEL.LSTR.AGES_MEMORY_LENGTH // cfg.MODEL.LSTR.AGES_MEMORY_SAMPLE_RATE
        cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH // cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH // cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        cfg.MODEL.LSTR.TOTAL_MEMORY_NUM_SAMPLES = \
            cfg.MODEL.LSTR.AGES_MEMORY_NUM_SAMPLES + \
            cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES + \
            cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES

        assert cfg.MODEL.LSTR.INFERENCE_MODE in ['batch', 'stream']

    # Infer output dir
    config_name = cfg.NAME # osp.splitext(args.config_file)[0].split('/')[1:]
    cfg.OUTPUT_DIR = osp.join(cfg.OUTPUT_DIR, config_name)
    if cfg.SESSION:
        cfg.OUTPUT_DIR = osp.join(cfg.OUTPUT_DIR, cfg.SESSION)  # What is it?

def load_cfg():
    # args = parse_args()
    cfg = Config()
    # args.config = './configs/lstr_long_512_work_8_kinetics_1x.yaml'
    assert_and_infer_cfg(cfg)
    return cfg

    
if __name__ == '__main__':
    import numpy as np
    import torch

    cfg = load_cfg()
    print('work memory second: ', cfg.MODEL.LSTR.WORK_MEMORY_LENGTH)
    print('verbos: ', cfg.VERBOSE)
    print('output dir: ', cfg.OUTPUT_DIR)
    print('number of memory samples: ',cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES)
    print('long memory length: ', cfg.MODEL.LSTR.LONG_MEMORY_LENGTH)
    print(cfg.INPUT.TARGET_PERFRAME)
    # print(cfg.DATA.TRAIN_SESSION_SET)
    
    