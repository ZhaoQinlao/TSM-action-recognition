import os
import torch
from torch.utils import data
import numpy as np
from numpy.random import randint
from PIL import Image
import cv2

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def start_frames(self):
        return int(self._data[1]) if len(self._data) > 1 else 0

    @property
    def num_frames(self):
        return int(self._data[2]) if len(self._data) > 2 else -1

    @property
    def label(self):
        return int(self._data[3]) if len(self._data) > 3 else -1


class VideoTSNDataSet(data.Dataset):
    """直接从视频文件加载帧的TSN数据集"""
    
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 transform=None, random_shift=True, 
                 test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        
        if self.modality == 'RGBDiff':
            # Diff需要多一帧来计算差异
            self.new_length += 1
            
        self._parse_list()
        
    def _parse_list(self):
        """解析视频列表文件"""
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        for x in tmp:
            # 确保视频路径是绝对路径
            if not os.path.isabs(x[0]):
                x[0] = os.path.join(self.root_path, x[0]+'.mp4')
        self.video_list = [VideoRecord(x) for x in tmp]
            
    def _load_frames_opencv(self, video_path, frame_indices):
        """使用OpenCV库加载指定帧"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"错误: 无法打开视频 {video_path}")
            return [Image.new('RGB', (224, 224), (0, 0, 0))]
            
        max_idx = max(frame_indices)
        frame_count = 0
        
        while frame_count <= max_idx:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count in frame_indices:
                # 转BGR到RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
                
            frame_count += 1
            
        cap.release()
        
        # 如果没有获取到足够的帧
        while len(frames) < len(frame_indices):
            frames.append(Image.new('RGB', (224, 224), (0, 0, 0)))
            
        return frames
        
    def _get_frame_count(self, video_path):
        """获取视频总帧数"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count

    def _sample_indices(self, record):
        """采样训练帧索引"""
        num_frames = record.num_frames
        if num_frames <= 0:
            num_frames = self._get_frame_count(record.path)
            
        average_duration = (num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif num_frames > self.num_segments:
            offsets = np.sort(randint(num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        
        return offsets + record.start_frames

    def _get_val_indices(self, record):
        """采样验证帧索引"""
        num_frames = record.num_frames
        if num_frames <= 0:
            num_frames = self._get_frame_count(record.path)
            
        if num_frames > self.num_segments + self.new_length - 1:
            tick = (num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))

        return offsets + record.start_frames

    def _get_test_indices(self, record):
        """采样测试帧索引"""
        num_frames = record.num_frames
        if num_frames <= 0:
            num_frames = self._get_frame_count(record.path)
            
        tick = (num_frames - self.new_length + 1) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets + record.start_frames

    def __getitem__(self, index):
        """获取数据样本"""
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        """获取并处理指定帧索引的图像"""
        images = []
        for seg_ind in indices:
            p = int(seg_ind)
            all_indices = [p + i for i in range(self.new_length) if p + i < record.num_frames]
            
            # 加载指定帧
            frames = self._load_frames_opencv(record.path, all_indices)
                
            images.extend(frames)

        # 数据增强和预处理
        process_data = self.transform(images) if self.transform else images
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)


if __name__ == "__main__":
    # 使用示例
    video_dataset = VideoTSNDataSet(
        root_path='/home/fitz_joye/assembly101-action-recognition/TSM-action-recognition/data/resized',
        list_file='/home/fitz_joye/assembly101-action-recognition/TSM-action-recognition/data/test_combined.txt',
        num_segments=8,
        new_length=1,
        modality='RGB',
        transform=None,  # 示例转换
        random_shift=True,
        test_mode=False
    )

    dataloader = torch.utils.data.DataLoader(
        video_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    for i in video_dataset:
        print(i)
