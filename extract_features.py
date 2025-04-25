import os
import torch
import sys
import torchvision
from torchvision.transforms import v2
import argparse
from decord import VideoReader, cpu, gpu

import numpy as np
from tqdm import tqdm

from ops.models import TSN

net = TSN(1380, 8, 'RGB', base_model='resnet50', consensus_type='avg', img_feature_dim=256,pretrain='imagenet',is_shift=True,shift_div=8, shift_place='blockres',non_local=False, dropout=0.5, partial_bn=True)
# net = TSN(1380, 1, 'RGB', base_model='resnet50', consensus_type='avg', img_feature_dim=256,pretrain='imagenet',is_shift=False, non_local=False, dropout=0.5, partial_bn=True)

ckpt = torch.load("pretrained_models/TSM_Assembly101_combined_resnet50_shift8_blockres_avg_segment8_e50.pth", weights_only=False)
checkpoint = ckpt['state_dict']

sd = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
# for k,v in list(sd.items()):
#     if 'net.' in k:
#         sd[k.replace('net.','')] = sd.pop(k)
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

def get_args():
    parser = argparse.ArgumentParser(
        'Extract TAD features using the videomae model', add_help=False)

    parser.add_argument(
        '--data_set',
        default='THUMOS14',
        choices=['THUMOS14', 'FINEACTION', 'Assembly101'],
        type=str,
        help='dataset')

    parser.add_argument(
        '--data_path',
        default='YOUR_PATH/thumos14_video',
        type=str,
        help='dataset path')
    parser.add_argument(
        '--data_list',
        default='YOUR_PATH/thumos14_video/THUMOS14_test.txt',
        type=str,
        help='dataset list path')
    parser.add_argument(
        '--save_path',
        default='YOUR_PATH/thumos14_video/th14_vit_g_16_4',
        type=str,
        help='path for saving features')

    return parser.parse_args()


def get_start_idx_range(data_set):

    def thumos14_range(num_frames):
        return range(0, num_frames - 15, 4)

    def fineaction_range(num_frames):
        return range(0, num_frames - 15, 16)
    
    def assembly101_range(num_frames):
        return range(0, num_frames - 47, 4)

    if data_set == 'THUMOS14':
        return thumos14_range
    elif data_set == 'FINEACTION':
        return fineaction_range
    elif data_set == 'Assembly101':
        return thumos14_range
    else:
        raise NotImplementedError()


def extract_feature(args):
    # preparation
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    start_idx_range = get_start_idx_range(args.data_set)
   

    # get video path
    # vid_list = os.listdir(args.data_path)
    # random.shuffle(vid_list)
    with open(args.data_list, 'r') as f:
        vid_list = f.readlines()
    vid_list = [vid.strip() for vid in vid_list]
    print(f'Extracting features for {len(vid_list)} videos')

    # extract feature
    num_videos = len(vid_list)
    for idx, vid_name in enumerate(tqdm(vid_list, ncols=80, desc='Total: ')):
        url = os.path.join(args.save_path, vid_name.split('.')[0] + '.npy')
        os.makedirs(os.path.dirname(url), exist_ok=True)
        if os.path.exists(url):
            continue

        video_path = os.path.join(args.data_path, vid_name)
        vr = VideoReader(video_path, ctx=cpu(0))

        feature_list = []
        for start_idx in tqdm(start_idx_range(len(vr)), ncols=80, desc='Video: '):
            data = vr.get_batch(np.arange(start_idx, start_idx + 48, 6)).asnumpy()
            data = torch.from_numpy(data).to('cuda').permute(0,3,1,2)  # [8, 3, 384, 224]
            images = [transform(frame) for frame in data]  # 每帧单独处理
            input_data = torch.stack(images, dim=0)  # [8, 3, 224, 224]

            with torch.no_grad():
                feature = net.extract_feature(input_data).mean(dim=0)
                feature_list.append(feature.cpu().numpy())

        # [N, C]
        np.save(url, np.vstack(feature_list))
        print(f'[{idx} / {num_videos}]: save feature on {url}')


if __name__ == '__main__':
    args = get_args()
    extract_feature(args)
