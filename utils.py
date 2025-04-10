import torch 
from ops.models import TSN
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))

def load_model(eval=True, device='cuda'):
    net = TSN(1380, 1, 'RGB', base_model='resnet50', consensus_type='avg', img_feature_dim=256,pretrain='imagenet',is_shift=False, non_local=False, dropout=0.5, partial_bn=True)
    ckpt = torch.load(os.path.join(cur_dir,"pretrained_models/TSM_Assembly101_combined_resnet50_shift8_blockres_avg_segment8_e50.pth"), weights_only=False)
    checkpoint = ckpt['state_dict']

    sd = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
    for k,v in list(sd.items()):
        if 'net.' in k:
            sd[k.replace('net.','')] = sd.pop(k)
    replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                    'base_model.classifier.bias': 'new_fc.bias',
                    }
    for k, v in replace_dict.items():
        if k in sd:
            sd[v] = sd.pop(k)

    net.load_state_dict(sd)
    if eval:
        net.eval()
    net = net.to(device)
    return net

