## 不实用，一个很慢的版本，建议使用chunk_extractor.py

from torchvision.transforms import v2
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from PIL import Image

from ops.models import TSN

net = TSN(1380, 1, 'RGB', base_model='resnet50', consensus_type='avg', img_feature_dim=256,pretrain='imagenet',is_shift=False, non_local=False, dropout=0.5, partial_bn=True, semantic=True)

ckpt = torch.load("/home/fitz_joye/TSM-action-recognition/checkpoint/TSM_Assembly101_combined_resnet50_shift8_blockres_avg_segment8_e5_cos/ckpt.best.pth.tar", weights_only=False)
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
        feature = net.extract_feature(frame).squeeze(0)  # [2048]
        # print(feature.shape)
    return feature.cpu().numpy()

class DistillDataset(Dataset):
    def __init__(self, images_root='/home/fitz_joye/TSM-action-recognition/data/images',
                 save_path='/home/fitz_joye/TSM-action-recognition/data/assembly101/semantic_features',
                 transform=transform):
        """
        Args:
            images_root (string): Path to the directory with images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_root = images_root
        self.transform = transform
        self.images_list = self._get_images_list()
        self.save_path = save_path
    
    def _get_images_list(self):
        with open("/home/fitz_joye/assembly101-temporal-action-segmentation/data/statistic_input.pkl", "rb") as f:
            data = pickle.load(f)
        images_list = []
        for video in data.keys():
            for view in data[video].keys():
                start, end = data[video][view]
                images_list.extend(
                    [f"{video}/{view}/{i:010d}.jpg" for i in range(start, end)])
                break
        return images_list

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        key = self.images_list[idx]
        image = Image.open(os.path.join(self.images_root,key)).convert('RGB')            
        return np.array(image), key

save_path = '/home/fitz_joye/TSM-action-recognition/data/assembly101/semantic_features'
dataset = DistillDataset()
# for i in dataset:
#     break
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
for i, (image,key) in enumerate(tqdm(dataloader, ncols=70)):
    image, key = image[0], key[0]
    image = transform((torch.tensor(image).cuda().permute(2,0,1)))
    with torch.no_grad():
        feature = net.extract_feature(image).squeeze(0)  # [2048]
    # print(feature.shape)
    output_path = os.path.join(save_path, key.replace('.jpg', '.npy'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, feature.cpu().numpy())
