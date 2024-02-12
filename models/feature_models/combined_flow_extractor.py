#%%
import torch
import torch.nn as nn
from .bn_inception import BNInception, get_bninception
from .flownet import FastFlowNet, get_flownet
import time
from tqdm import trange

__all__ = ['CombinedFlowModel']

class CombinedFlowModel(nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.flow_extractor = FastFlowNet()      # get_flownet(cfg.MODEL.FLOWNET_CKPT)
        self.bn_inception = BNInception(self.cfg.BN_INCEPTION.IN_CHANNELS)        # get_bninception(cfg.MODEL.BNINCEPTION_CKPT)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        x = self.flow_extractor(x)
        print(x.size())
        x = x.view(x.size(0) // 5, self.cfg.BN_INCEPTION.IN_CHANNELS, x.size(2), x.size(3))
        x = self.bn_inception(x)
        x = self.avg_pool(x)
        return x.unsqueeze(-1).unsqueeze(-1)



        x = self.flow_extractor(x) # (num_frames, 6, height, width)
        # copy the flow channels to match the input of BNInception
        x = torch.cat([x, x, x, x, x], dim=1)
        #Should be (x,y,x,y,x,y,x,y,x,y,x,y)
        #x = x.view(-1, 2, x.shape[2], x.shape[3]) # (B, 2, height, width)
        x = self.bn_inception(x)
        x = self.avg_pool(x)
        return x
    
    def load_ckpt(self):
        flownet_dict = torch.load(self.cfg.FAST_FLOWNET.CKPT)
        bn_inception_dict = torch.load(self.cfg.BN_INCEPTION.CKPT)
        self.flow_extractor.load_state_dict(flownet_dict)
        new_weights = {}
        for key in bn_inception_dict.keys():
            if 'fc_action' in key:
                continue
            if bn_inception_dict[key].shape[0] == 1:
                new_weights[key] = bn_inception_dict[key].squeeze(0)
            else:
                new_weights[key] = bn_inception_dict[key]
        self.bn_inception.load_state_dict(new_weights)


#%%
if __name__ == '__main__':
    comb_model = CombinedFlowModel().cuda().eval()
    # input is stacked pair of frames (N-1, 3*2, H, W)
    # N-1 acts as the batch dimension for flow extractor
    input_t = torch.randn(1, 6, 384, 512).cuda()
    num_passes = 5
    print(f"Running {num_passes} passes of forward pass")
    start = time.time()
    with torch.no_grad():
        for x in trange(num_passes):
            output_t = comb_model(input_t) 
    end = time.time()
    print(f'Time elapsed: {end-start:.3f}s for {num_passes} passes, Each forward pass took: {(end-start)/num_passes*1000:.3f}ms')
    out_t = comb_model(input_t)
    print(out_t.shape)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = comb_model.train()
    print('Number of parameters: {:.2f} M'.format(count_parameters(model) / 1e6))
# %%
