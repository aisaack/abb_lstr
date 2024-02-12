import os
import os.path as osp
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle as pkl

from configure import load_cfg
from model_builder import Lstr
from dataset_builder import LSTRBatchInferenceDataLayer
from utils import get_device
from evaluation import compute_result
from logger import setup_logger



def inference(cfg, weight_dir):
    cfg.MODEL.WEIGHT = weight_dir
    device = get_device(cfg)
    logger = setup_logger(cfg, 'test')
    model = Lstr(cfg)
    state_dict = torch.load(cfg.MODEL.WEIGHT)
    model.load_state_dict(state_dict['model_state_dict'])
    print(f'model is loaded with {cfg.MODEL.WEIGHT}')

    model = model.to(device)
    model.eval()

    dataloader = DataLoader(
        dataset = LSTRBatchInferenceDataLayer(cfg, phase='test'),
        batch_size=cfg.DATA_LOADER.BATCH_SIZE * 16,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY
    )

    pred_scores = {}
    gt_targets = {}

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='BatchInference')
        for idx, data in enumerate(pbar, start=1):
            target = data[-4]

            score = model(*[x.to(device) for x in data[:-4]])
            score = score.softmax(dim=-1).cpu().numpy()

            for bs, (session, query_indices, num_frames) in enumerate(zip(*data[-3:])):
                if session not in pred_scores:
                    pred_scores[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))
                if session not in gt_targets:
                    gt_targets[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))

                if query_indices[0] == 0:
                    pred_scores[session][query_indices] = score[bs]
                    gt_targets[session][query_indices] = target[bs]
                else:
                    pred_scores[session][query_indices] = score[bs][-1]
                    gt_targets[session][query_indices] = target[bs][-1]


    pkl.dump({
        'cfg': cfg,
        'perframe_pred_scores': pred_scores,
        'perframe_gt_targets': gt_targets,
    }, open(osp.splitext(cfg.MODEL.CHECKPOINT)[0] + '.pkl', 'wb'))

    result = compute_result['perframe'](
        cfg,
        np.concatenate(list(gt_targets.values()), axis=0),
        np.concatenate(list(pred_scores.values()), axis=0),
    )
    logger.info('Action detection preframe m{}: {:5f}'.format(cfg.DATA.METRICS, result['mean_AP']))

if __name__ == '__main__':
    weight_dir = './checkpoints/lstr_distil/epoch-6.pth'
    inference(load_cfg(), weight_dir)


