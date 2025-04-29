import torch

from torch.utils.data import Dataset, DataLoader
from glob import glob
import numpy as np
from functools import lru_cache
import lmdb
import pickle
from PIL import Image
import torchvision.transforms.v2 as v2
import os
from tqdm import tqdm

transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(224),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

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

class DistillDataset(Dataset):
    def __init__(self, images_root='/home/fitz_joye/TSM-action-recognition/data/images',
                 features_path='/home/fitz_joye/TSM-action-recognition/data/assembly101/TSM_features',
                 transform=transform, chunk_size=128):
        """
        Args:
            images_root (string): Path to the directory with images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_root = images_root
        self.transform = transform
        self.chunk_size = chunk_size
        self.images_list = self._get_images_list()

    def _get_images_list(self):
        with open("/home/fitz_joye/assembly101-temporal-action-segmentation/data/statistic_input.pkl", "rb") as f:
            data = pickle.load(f)
        images_list = []
        for video in data.keys():
            for view in data[video].keys():
                start, end = data[video][view]
                images_list.extend(
                    [(f"{video}/{view}/{i:010d}.jpg", i, min(i+self.chunk_size,end)) for i in range(start, end, self.chunk_size)])
                
        return images_list

    def __len__(self):
        return len(self.images_list)

    def _get_images(self, start_key, start, end):
        video, view = start_key.split('/')[:2]
        keys = [f"{self.images_root}/{video}/{view}/{i:010d}.jpg" for i in range(
            start, end)]
        images =  [Image.open(key).convert('RGB') for key in keys]
        # return [np.array(image) for image in images]
        if self.transform:
            images = [self.transform(image) for image in images]
        images = torch.stack(images, dim=0)
        return images
    
    def __getitem__(self, idx):
        return self._get_images(*self.images_list[idx]), self.images_list[idx][0]

def test_dataset():
    # 使用示例
    dataset = DistillDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    for i in range(len(dataset)):
        data, path = dataset[i]
        print(f"Image {i}: {data.shape} @ {path}")
        break

    for images, paths in dataloader:
        print(f"Batch Images: {images[0].shape} @ {paths[0]}")
        # Batch Images: torch.Size([128, 224, 398, 3]) 
        # @ nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724/C10095_rgb/0000000001.jpg
        

# test_dataset()
dataset = DistillDataset(chunk_size=512)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, collate_fn=lambda x: x[0])

save_path = '/home/fitz_joye/TSM-action-recognition/data/assembly101/semantic_features'

for i, (images,key) in enumerate(tqdm(dataloader, ncols=70)):
    images = images.cuda()
    # print(images.shape)
    with torch.no_grad():
        feature = net.extract_feature(images).squeeze(0)  # [2048]
    # print(feature.shape)
    output_path = os.path.join(save_path, key.replace('.jpg', '.npy'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, feature.cpu().numpy())