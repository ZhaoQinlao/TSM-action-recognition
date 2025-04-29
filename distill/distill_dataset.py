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

# @lru_cache(maxsize=128)
# def get_feature(feature_path):
#     return np.load(feature_path)

transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(224),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class DistillDataset(Dataset):
    def __init__(self, images_root='/home/fitz_joye/TSM-action-recognition/data/images',
                 features_path='/home/fitz_joye/TSM-action-recognition/data/assembly101/TSM_features',
                 transform=transform, chunk_size=128):
        """
        Args:
            images_root (string): Path to the directory with images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.VIEWS = ['C10095_rgb', 'C10115_rgb', 'C10118_rgb', 'C10119_rgb', 'C10379_rgb', 'C10390_rgb', 'C10395_rgb', 'C10404_rgb',
                      'HMC_21176875_mono10bit', 'HMC_84346135_mono10bit', 'HMC_21176623_mono10bit', 'HMC_84347414_mono10bit',
                      'HMC_21110305_mono10bit', 'HMC_84355350_mono10bit', 'HMC_21179183_mono10bit', 'HMC_84358933_mono10bit']
        self.images_root = images_root
        self.transform = transform
        self.chunk_size = chunk_size
        self.images_list = self._get_images_list()
        self.env = {view: lmdb.open(
            f'{features_path}/{view}', readonly=True, lock=False) for view in self.VIEWS}

    def _get_images_list(self):
        with open("/home/fitz_joye/assembly101-temporal-action-segmentation/data/statistic_input.pkl", "rb") as f:
            data = pickle.load(f)
        images_list = []
        for video in data.keys():
            for view in data[video].keys():
                start, end = data[video][view]
                images_list.extend(
                    [f"{video}/{view}/{i:010d}.jpg" for i in range(start, end)])
        return images_list

    def __len__(self):
        return len(self.images_list)

    # def _load_feature(self, idx):
    #     image_path = self.images_list[idx]
    #     feature_path = '/'.join(image_path.split('/')[:-1])+'feature.npy'
    #     offset = int(image_path.split('/')[-1].split('.')[0], base=10)
    #     feature = get_feature(feature_path)

    def _load_feature(self, start_key):
        video, view = start_key.split('/')[:2]
        start_key = start_key.split('/')[-1].split('.')[0]
        start_key = int(start_key, base=10)
        keys = [
            f"{video}/{view}/{view}_{i:010d}.jpg" for i in range(start_key, start_key+self.chunk_size)]
        with self.env[view].begin() as e:
            cursor = e.cursor()
            data_list = cursor.getmulti(
                [key.strip().encode('utf-8') for key in keys])

        elements = []
        for key, data in data_list:
            if data is None:
                print('no available data.')
                exit(2)
            elements.append(np.frombuffer(data, dtype='float32'))

        elements = np.array(elements)
        return elements

    def _get_images(self, start_key):
        video, view = start_key.split('/')[:2]
        start_key = start_key.split('/')[-1].split('.')[0]
        start_key = int(start_key, base=10)
        keys = [f"{self.images_root}/{video}/{view}/{i:010d}.jpg" for i in range(
            start_key, start_key+self.chunk_size)]
        return [Image.open(key).convert('RGB') for key in keys]

    def image_feature_pair(self, idx):
        start_key = self.images_list[idx]
        elements = self._load_feature(start_key)
        images = self._get_images(start_key)
        if self.transform:
            images = [self.transform(np.array(image)) for image in images]
        return np.array(images), elements
    
    def __getitem__(self, idx):
        key = self.images_list[idx]
        video, view = key.split('/')[:2]
        image = Image.open(os.path.join(self.images_root,key)).convert('RGB')
        if self.transform:
            image = self.transform(np.array(image))
        return np.array(image), key



if __name__ == "__main__":
    # 使用示例
    dataset = DistillDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    for i in range(len(dataset)):
        data, path = dataset[i]
        print(f"Image {i}: {data.shape} @ {path}")
        break

    for images, paths in dataloader:
        print(f"Batch Images: {images.shape} @ {paths}")
        break
