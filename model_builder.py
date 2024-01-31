import torch
import torch.nn as nn
from collections import deque

from configure import load_cfg
from models import CombinedFlowModel, ResNet, Bottleneck, LSTRStream, LSTR


# Long-Term and Short-Term Memory
class Memory:
    def __init__(self, long_term_size, short_term_size):
        self.long_term_memory = deque(maxlen=long_term_size)
        self.short_term_memory = deque(maxlen=short_term_size)

    def update(self, new_feature):
        self.short_term_memory.append(new_feature)
        if len(self.short_term_memory) == self.short_term_memory.maxlen:
            self.long_term_memory.append(self.short_term_memory[0])
            

class Lstr(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.MODEL.LSTR.IS_STREAM is True:
            self.model = LSTRStream(self.cfg)
        else:
            self.model = LSTR(self.cfg)
        print(f'  {self.model.__class__.__name__} initialized')
    
    def load_ckpt(self):
        '''
        initiate each model's pretrained weight
            self.cfg has each module's pre-trained weight path
        '''
        print(self.cfg.MODEL.LSTR.CKPT)
        lstr_state_dict = torch.load(self.cfg.MODEL.LSTR.CKPT)
        self.model.load_state_dict(lstr_state_dict['model_state_dict'])
        self.model.eval()
        print(f'  {self.model.__class__.__name__} resumed')
        

    def forward(self, rgb, flow, mask=None):
        out = self.model(rgb, flow, mask)
        return out
        

class Resnet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = ResNet(block=Bottleneck, layers=self.cfg.MODEL.RESNET.LAYERS)
        print(f'  ResNet{self.cfg.MODEL.RESNET.DEPTH} initialized')

    def forward(self, x):
        out = self.model(x)
        #torch.Size([1681, 2048])
        return out

    def load_ckpt(self, device=None):
        self.model.load_state_dict(torch.load(self.cfg.MODEL.RESNET.CKPT, map_location=device if device is not None else 'cpu'))
        self.model.eval()
        print(f'  ResNet{self.cfg.MODEL.RESNET.DEPTH} resumed')




class Flownet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = CombinedFlowModel(self.cfg.MODEL.FLOWNET)
        print('  CombinedFlowNet initialized')

    def forward(self, x):
        """
        Input shape is UNKNOWN
        """
        out = self.model(x)
        return out

    def load_ckpt(self):
        self.model.load_ckpt()
        self.model.eval()
        print('  CombinedFlowNet resumed')


def get_resume(models, mode):
    if mode == 'detector':
        models['lstr'].load_ckpt()

    elif mode == 'all':
        for k, v in models.items():
            v.load_ckpt()

    elif mode == 'extractor':
        models['resnet'].load_ckpt()
        models['flownet'].load_ckpt()


def get_model(cfg, pretrained=None):
    '''
    cfg         : configureation object
    pretrained  : string. if it's one of {detector, extractor, all, None}, load model weight and turn on eval() mode 
                          else None of model resume checkpoint
                  detector - lstm only
                  extractor - resnet, flownet only
                  all - all of model resumes checkpoint
                  None - None of the models load checkpoints
    '''
    models = dict(
        resnet = Resnet(cfg),
        flownet = Flownet(cfg),
        lstr = Lstr(cfg)
        )
    if pretrained is not None:
        get_resume(models, pretrained)
    return models

if __name__ == '__main__':
    from configure import load_cfg
    cfg= load_cfg()

    rgb = torch.randn(50, 3, 180, 320)
    flow = torch.randn(50, 6, 256, 384)
    models = get_model(cfg)

    rgb_feat = models.get('resnet')(rgb)        # [frame, 2048]
    flow_feat = models.get('flownet')(flow)     # [frame, 1024]
    print(rgb_feat.size(), flow_feat.size())
    
