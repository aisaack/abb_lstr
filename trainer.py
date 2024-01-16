import time
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from configure import load_cfg
from model_builder import Lstr
from dataset_builder import LSTRDataLayer
from criterions import get_criterion
from scheduler_builder import get_scheduler
from evaluation import compute_result
from logger import setup_logger
from checkpointer import setup_checkpointer
from utils import get_device



def build_dataloader(cfg, phase):
    data_loader = DataLoader(
        dataset = LSTRDataLayer(cfg, phase),
        batch_size=cfg.DATA_LOADER.BATCH_SIZE,
        shuffle = True if phase == 'train' else False,
        num_workers = cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory = cfg.DATA_LOADER.PIN_MEMORY
    )
    return data_loader



def get_optimizer(cfg, model):
    if cfg.SOLVER.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            [{'params': model.parameters(), 'initial_lr': cfg.SOLVER.BASE_LR}],
            lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY, momentum=cfg.SOLVER.MOMENTUM,
        )
    elif cfg.SOLVER.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            [{'params': model.parameters(), 'initial_lr': cfg.SOLVER.BASE_LR}],
            lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(
            [{'params': model.parameters(), 'initial_lr': cfg.SOLVER.BASE_LR}],
            lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise RuntimeError('Unknown optimizer: {}'.format(cfg.SOLVER.OPTIMIZER))
    return optimizer


def train(cfg):
    logger = setup_logger(cfg, 'train')
    print('logger loaded')

    ckpt_setter = setup_checkpointer(cfg, phase='train')
    print('checkpoint loadded')
    
    data_loaders = {phase: build_dataloader(cfg, phase) for phase in cfg.SOLVER.PHASES}
       
    model = Lstr(cfg)   #get_model(cfg, pretrained='extractor')    # load weight of resnet and flownet
    
    criterion = get_criterion(cfg)

    device = get_device(cfg)        
    # device = 'cpu'
    model = model.to(device)
    model.train(True)

    optimizer = get_optimizer(cfg, model.model)
    print('Optimizer Loaded')

    if cfg.SOLVER.RESUME is True:
        ckpt_setter.load(model, optimizer)     # here load lstr and optimizer checkpoint
        print('Model and optimizer resumed')
                                                  
    scheduler = get_scheduler(cfg, optimizer, len(data_loaders['train']))
    print('Scheduler loaded')

    for epoch in range(cfg.SOLVER.START_EPOCH, cfg.SOLVER.START_EPOCH + cfg.SOLVER.NUM_EPOCHS):
        det_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        det_pred_scores = []
        det_gt_targets = []
        
        start = time.time()
        for phase in cfg.SOLVER.PHASES:
            training = phase == 'train'
            model.train(training)

            with torch.set_grad_enabled(training):

                pbar = tqdm(data_loaders[phase], desc=f'{phase} Epoch: {epoch}')
                for idx, data in enumerate(pbar, start=1):
                    batch_size = data[0].size(0)
                    det_target = data[-1].to(device)

                    det_score = model(*[x.to(device) for x in data[:-1]])
                    det_score = det_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                    det_target = det_target.reshape(-1, cfg.DATA.NUM_CLASSES)
                    det_loss = criterion['MCE'](det_score, det_target)
                    det_losses[phase] += det_loss.item() * batch_size

                    pbar.set_postfix({
                        'lr': '{:.7f}'.format(scheduler.get_last_lr()[0]),
                        'det_loss': '{:.5f}'.format(det_loss.item()),
                    })

                    if training:
                        optimizer.zero_grad()
                        det_loss.backward()
                        optimizer.step()
                        scheduler.step()
                    else:
                        # Prepare for evaluation
                        det_score = det_score.softmax(dim=1).cpu().tolist()
                        det_target = det_target.cpu().tolist()
                        det_pred_scores.extend(det_score)
                        det_gt_targets.extend(det_target)
        
        end = time.time()

        log = []
        log.append('Epoch {:2}'.format(epoch))
        log.append('train det_loss: {:.5f}'.format(
            det_losses['train'] / len(data_loaders['train'].dataset),
        ))
        if 'test' in cfg.SOLVER.PHASES:
            # Compute result
            det_result = compute_result['perframe'](
                cfg,
                det_gt_targets,
                det_pred_scores,
            )
            log.append('test det_loss: {:.5f} det_mAP: {:.5f}'.format(
                det_losses['test'] / len(data_loaders['test'].dataset),
                det_result['mean_AP'],
            ))
        log.append('running time: {:.2f} sec'.format(
            end - start,
        ))
        logger.info(' | '.join(log))

        # Save checkpoint for model and optimizer
        ckpt_setter.save(epoch, model.model, optimizer)

        # Shuffle dataset for next epoch
        data_loaders['train'].dataset.shuffle()

if __name__ == '__main__':
    cfg = load_cfg()
    train(cfg)

