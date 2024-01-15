import time
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim

from configure import load_cfg
from model_builder import get_model
from dataset_builder_old import get_dataset
from criterions import get_criterion
from scheduler_builder import get_scheduler
from evaluation import compute_result
from logger import setup_logger
from checkpointer import setup_checkpointer
from utils import (
    resize_image, get_fusion_features, get_device, compute_features)




class BatchDispatcher:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.buffer = None

    def __call__(self, fusion_features):
        
        stream = [[] for _ in fusion_features]

        if self.buffer is None:
            self.buffer = [[] for _ in fusion_features]

        for idx, feat in enumerate(fusion_features):
            self.buffer[idx].extend(feat)
        
        del fusion_features
        curr_batch = len(self.buffer[0])
        
        if curr_batch == self.batch_size:
            stream = self.buffer
            self.buffer = None
            return stream

        if curr_batch < self.batch_size:
            print(f'Filled batch is {curr_batch}. {self.batch_size - curr_batch} to fill.')
            return None

        if curr_batch > self.batch_size:
            diff = curr_batch - self.batch_size
            cut = np.random.randint(diff)
            print(f'Fetched batch size is {curr_batch}. Cut off {diff}, and take from {cut} to {cut+self.batch_size}')
            for idx, feat in enumerate(self.buffer):
                stream[idx] = feat[cut:cut + self.batch_size]
                # feat = feat[diff:]
            # print(len(self.buffer[0]), len(stream[0]))
            self.buffer = None
            return stream
        

def empty_mem(*args):
    for arg in args:
        del arg


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
    batch_size = 16
    logger = setup_logger(cfg, 'train')
    print('logger loaded')

    ckpt_setter = setup_checkpointer(cfg, phase='train')
    print('checkpoint loadded')
    
    datasets = {phase: get_dataset(cfg, phase) for phase in cfg.SOLVER.PHASES}
    # memory = Memory(long_term_size=cfg.MODEL.LSTR.LONG_MEMORY_SECONDS,
    #                 short_term_size=cfg.MODEL.LSTR.WORK_MEMORY_SECONDS)
    
    models = get_model(cfg, pretrained='extractor')    # load weight of resnet and flownet
    
    criterion = get_criterion(cfg)

    device = get_device(cfg)        
    # device = 'cpu'

    for name, model in models.items():
        model.to(device)
        # torch.compile(model)
        print(f'{name} on {device}')

    models['lstr'].train(True)

    optimizer = get_optimizer(cfg, models.get('lstr'))
    print('Optimizer Loaded')

    if cfg.SOLVER.RESUME is True:
        ckpt_setter.load(models['lstr'], optimizer)     # here load lstr and optimizer checkpoint
        print('Model and optimizer resumed')
                                                  
    scheduler = get_scheduler(cfg, optimizer, datasets['train'].__len__())
    print('Scheduler loaded')



    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(models['lstr'])
    for epoch in range(cfg.SOLVER.START_EPOCH, cfg.SOLVER.START_EPOCH + cfg.SOLVER.NUM_EPOCHS):
        det_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        det_pred_scores = []
        det_gt_targets = []
        
        dispatcher = BatchDispatcher(batch_size)
        start = time.time()
        for phase in cfg.SOLVER.PHASES:
            training = phase == 'train'
            models['lstr'].train(training)

            pbar = tqdm(datasets[phase], desc=f'{phase}  Epoch: {epoch}')
            for idx, (rgb, flow, target) in enumerate(pbar, start=1):
                
                rgb = rgb.to(device)                        # [frame, 3, 180, 320]
                flow = flow.to(device)                      # [frame, 6, 180, 320]
                target = target.to(device)                  # [frame, num_class]
                flow = resize_image(flow).contiguous()      # [frame, 6, 256, 384]

                rgb_feature, flow_feature, helpers = compute_features(
                    rgb, flow, target, models, cfg.MODEL.LSTR.WORK_MEMORY_LENGTH, train_extractor=False)
                empty_mem(*[rgb, flow, target])
                
                fusion_features = get_fusion_features(cfg, rgb_feature, flow_feature, helpers)    # [[fusion_rgbs * len(helper)], [fusion_flows * len(helper)], [masks * len(helper], [targets * len(helper]]
                empty_mem(*[rgb_feature, flow_feature, helpers])

                batch_stream = dispatcher(fusion_features)
                if batch_stream is None:
                    continue
                else:
                    batch_stream = [torch.stack(stream, dim=0) for stream in batch_stream]

                    with torch.set_grad_enabled(training):
                        det_score = models['lstr'](*[stream.to(device) for stream in batch_stream[:-1]])
                        det_score = det_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                        det_target = batch_stream[-1].reshape(-1, cfg.DATA.NUM_CLASSES)
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
                            det_score = det_score.softmax(dim=-1).cpu().tolist()
                            det_target = det_target.cpu().tolist()
                            det_pred_scores.extend(det_score)
                            det_gt_targets.extend(det_target)
                torch.cuda.empty.cache()
                # for test
                # if idx > 0:
                #     break

        end = time.time()

        log = []
        log.append(f'Epoch {epoch:2}')
        train_det_loss = det_losses['train'] / len(datasets['train'])
        log.append(f'train det_loss: {train_det_loss:.5f}')

        if 'test' in cfg.SOLVER.PHASES:
            det_result = compute_result['perframe'](cfg, det_gt_targets, det_pred_scores)
            test_det_loss = det_losses['test'] / len(datasets['test'])
            mAP = det_result['mean_AP']
            log.append(f'test det_loss: {test_det_loss:.5f}   det mAP: {mAP:.5f}')
        
        log.appned(f'Running time: {end-start:.2f} sec')
        logger.info(' | '.join(log))

        ckpt_setter.save(epoch, models['lstr'], optimizer)


if __name__ == '__main__':
    cfg = load_cfg()
    train(cfg)
        


        
            

            



    # return
    # for epoch in range(num_epochs):
    #     for idx, feature in enumerate(fusion_features):
    #         if len(feature) == 4:
    #             fusion_rgb, fusion_flow, memory_key_padding_mask, fusion_target = feature
    #         elif len(feature) == 3:
    #             fusion_rgb, fusion_flow, fusion_target = feature
            




    #         # print(rgb_feature.size())
    #         # print(flow_feature.size())

    #         rgb_feature_padded = pad_sequence(rgb_feature, batch_first=True, padding_value=0)
    #         flow_feature_padded = pad_sequence(flow_feature, batch_first=True, padding_value=0)
    #         target_padded = pad_sequence(target, batch_first=True, padding_value=0)


    # # #         #TODO
    # # #         # 1. feature fusion

    #         for frame in range(flow_feature_padded.shape[1]):
    #             # flow_frame = flow_feature[:, frame, :]
    #             # rgb_frame = rgb_feature[:, frame, :]
    #             # target_frame = target[:, frame, :]

    #             flow_frame = rgb_feature_padded[:, frame, :]
    #             rgb_frame = flow_feature_padded[:, frame, :]
    #             target_frame = target_padded[:, frame, :]

    #             combined_feature = torch.cat((rgb_frame, flow_frame), dim=1)

    #             memory.update(combined_feature)

    #             if len(memory.long_term_memory) > 0 and len(memory.short_term_memory) > 0:
    #                 long_term_memory_tensor = torch.stack(list(memory.long_term_memory)).to(device)
    #                 short_term_memory_tensor = torch.stack(list(memory.short_term_memory)).to(device)

    #                 outputs = models.get('lstr')(long_term_memory_tensor, short_term_memory_tensor)

    #                 loss = criterion(outputs, target_frame)

    #                 optimizer.zero_grad()
    #                 loss.backward()
    #                 optimizer.step()
                    
    #                 if (frame+1) % 10 == 0:
    #                     print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Frame [{frame+1}/{flow.shape[1]}], Loss: {loss.item():.4f}')
    #             else:
    #                 print('Memory is not yet filled. Continue collecting features.')

    # print('Training complete')


if __name__ == '__main__':
    cfg = load_cfg()
    train(cfg)


