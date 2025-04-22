import os
import torch

from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np


from ops.models import TSN

# net = TSN(1380, 8, 'RGB', base_model='resnet50', consensus_type='avg', img_feature_dim=256,pretrain='imagenet',is_shift=True,shift_div=8, shift_place='blockres',non_local=False, dropout=0.5, partial_bn=True)
net = TSN(1380, 1, 'RGB', base_model='resnet50', consensus_type='avg', img_feature_dim=256,pretrain='imagenet',is_shift=False, non_local=False, dropout=0.5, partial_bn=True)

ckpt = torch.load("pretrained_models/TSM_Assembly101_combined_resnet50_shift8_blockres_avg_segment8_e50.pth", weights_only=False)
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
print(f"Loading pretrained model")
net.eval()
net = net.to('cuda')


transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(224),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]).cuda()

def extract_frame_feature(frame):
    # H, W, C
    with torch.no_grad():
        # print(frame.shape)
        frame = torch.from_numpy(frame)
        frame = transform(frame.cuda().permute(2, 0, 1))  # [3, 224, 224]
        # print(frame.shape)
        feature = net.extract_feature(frame).squeeze(0)  # [2048]
        # print(feature.shape)
    return feature.cpu().numpy()

class Extractor():
    def __init__(self):
        self.features = []

    def extract_videos(self, frame):
        self.features.append(extract_frame_feature(frame))
        return np.stack(self.features)
    
    def clear(self):
        self.features = []
    

    
if __name__ == '__main__':
    extractor = Extractor()
    for i in range(3):
        dummy_input = torch.randn(224, 224, 3)
        output = extractor.extract_videos(dummy_input)
        print(output.shape)