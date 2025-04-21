import cv2
import os

def extract_video_segment(video_path, start_frame, segment_length, output_name):
    """
    提取视频指定片段并存储
    
    参数:
        video_path (str): 输入视频文件路径
        start_frame (int): 起始帧号(从0开始计数)
        segment_length (int): 要提取的片段长度(帧数)
        output_name (str): 输出视频文件名(无需扩展名)
    
    返回:
        bool: 操作是否成功
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: 无法打开视频文件")
        return False
    
    # 获取视频基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 检查输入参数是否有效
    if start_frame >= total_frames:
        print(f"Error: 起始帧 {start_frame} 超出视频总帧数 {total_frames}")
        return False
    
    end_frame = start_frame + segment_length
    if end_frame > total_frames:
        print(f"Warning: 指定片段长度超出视频范围，将截取到视频末尾")
        segment_length = total_frames - start_frame
    
    # 设置视频编码器 (根据系统选择适合的编码器)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 也可以使用 'XVID' 等
    
    # 创建输出视频写入器
    output_path = f"{output_name}.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 跳转到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 读取并写入指定帧
    frames_written = 0
    while frames_written < segment_length:
        ret, frame = cap.read()
        if not ret:
            break
        
        out.write(frame)
        frames_written += 1
    
    # 释放资源
    cap.release()
    out.release()
    
    # print(f"成功提取 {frames_written} 帧，保存为 {output_path}")
    return True

# 使用示例
# extract_video_segment("input.mp4", 100, 300, "output_segment")


import torch
from torch.utils.data import Dataset, DataLoader

class MAETransform(Dataset):
    def __init__(self, video_source_dir='data/assembly101/resized', output_dir='data/assembly101/videomae', mode='train'):
        super().__init__()
        self.video_source_dir = video_source_dir
        self.output_dir = output_dir
        self.mode = mode
        with open(f"/home/fitz_joye/TSM-action-recognition/data/{mode}_combined.txt") as f:
            lines = f.readlines()
        self.datalist = [line.strip().split(' ') for line in lines]

    def __getitem__(self, index):
        video_path, start, length, label = self.datalist[index]
        src_path = os.path.join(self.video_source_dir, video_path + '.mp4')
        start = int(start)
        length = int(length)
        # label = int(label)

        output_path = os.path.join(self.output_dir, '_'.join([video_path, str(start), str(length)]))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 提取视频片段
        # print(f"Extracting video segment from {src_path} to {output_path} starting at frame {start} for {length} frames.")
        extract_video_segment(src_path, start, length, output_path)
        
        # 这里可以添加更多的数据处理和转换操作
        
        return video_path, start, length, label


    def __len__(self):
        return len(self.datalist)


if __name__ == "__main__":
    from tqdm import tqdm

    trainset = MAETransform(video_source_dir='data/assembly101/resized', output_dir='data/assembly101/videomae_', mode='train')
    trainloader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=8)
    for i in tqdm(trainloader):
        pass

    valset = MAETransform(video_source_dir='data/assembly101/resized', output_dir='data/assembly101/videomae_', mode='validation')
    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=8)
    for i in tqdm(valloader):
        pass

    testset = MAETransform(video_source_dir='data/assembly101/resized', output_dir='data/assembly101/videomae_', mode='test')
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=8)
    for i in tqdm(testloader):
        pass