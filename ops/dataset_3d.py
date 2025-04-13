# Code for TSM adapted from the original TSM repo:
# https://github.com/mit-han-lab/temporal-shift-module

import os
import os.path
import numpy as np
from numpy.random import randint
import torch
from torch.utils import data
from tqdm import tqdm
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def start_frames(self):
        return int(self._data[1])

    @property
    def num_frames(self):
        return int(self._data[2])

    @property
    def label(self):
        return int(self._data[3])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='frame_{:010d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, 
                 dense_sample=False, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality == 'RGBDiff':
            # Diff needs one more image to calculate diff
            self.new_length += 1

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            #print(os.path.join(directory, self.image_tmpl.format(idx)))
            # return [Image.open(os.path.join(directory, directory.split('/')[-1] + '_' + self.image_tmpl.format(idx))).convert('RGB')]
            return [Image.open(os.path.join(directory, "{:010d}.jpg".format(idx))).convert('RGB')]
            #return [Image.new('RGB', (456, 256), (73, 109, 137))]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, directory.split('/')[-1] + '_' + self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, directory.split('/')[-1] + '_' + self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        for x in tmp:
            x[0] = self.root_path + x[0]
        self.video_list = [VideoRecord(x) for x in tmp]


    def __getitem__(self, index):
        record = self.video_list[index]
        segment_indices = np.arange(record.num_frames - self.new_length + 1) + record.start_frames
        return self.get(record, segment_indices)

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)

    @staticmethod
    def collate_fn(batch):
        # rnn padding
        data = [item[0] for item in batch]
        data = pad_sequence(data, batch_first=True)
        labels = [item[1] for item in batch]
        return data, labels


if __name__ == "__main__":
    # 使用示例
    from transforms import *
    video_dataset = TSNDataSet(
        root_path='/home/fitz_joye/assembly101-action-recognition/TSM-action-recognition/data/images/',
        list_file='/home/fitz_joye/assembly101-action-recognition/TSM-action-recognition/data/test_combined.txt',
        num_segments=8,
        new_length=1,
        modality='RGB',
        transform=torchvision.transforms.Compose([
                       GroupScale(int(224)),
                       GroupCenterCrop(224),
                       Stack(roll=False),
                       ToTorchFormatTensor(div=False),
                       GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                   ]),
        random_shift=True,
        test_mode=False
    )

    dataloader = torch.utils.data.DataLoader(
        video_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=video_dataset.collate_fn
    )

    for i in video_dataset:
        print(i[0].shape)
        break
    # print(video_dataset[0])

    for i in dataloader:
        print(i[0].shape)
        break