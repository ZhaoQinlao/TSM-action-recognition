import os
import torch

from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np
from decord import VideoReader, cpu
import gradio as gr
from tqdm import tqdm

from ops.models import TSN

net = TSN(1380, 8, 'RGB', base_model='resnet50', consensus_type='avg', img_feature_dim=256,pretrain='imagenet',is_shift=True,shift_div=8, shift_place='blockres',non_local=False, dropout=0.5, partial_bn=True)
# net = TSN(1380, 1, 'RGB', base_model='resnet50', consensus_type='avg', img_feature_dim=256,pretrain='imagenet',is_shift=False, non_local=False, dropout=0.5, partial_bn=True)

ckpt = torch.load("pretrained_models/TSM_Assembly101_combined_resnet50_shift8_blockres_avg_segment8_e50.pth", weights_only=False)
checkpoint = ckpt['state_dict']


mapping_file = '/home/fitz_joye/TSM-action-recognition/data/assembly101/assembly101-annotations/fine-grained-annotations/actions.csv'
with open(mapping_file, 'r') as file_ptr:
    lines = file_ptr.read().split('\n')[1:-1]
actions_dict = {int(line.split(',')[0]):line.split(',')[4] for line in lines}


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

def extract_frame_feature(frame):
    # H, W, C
    with torch.no_grad():
        # print(frame.shape)
        frame = np.copy(frame)
        frame = torch.from_numpy(frame)
        frame = transform(frame.cuda().permute(2, 0, 1))  # [3, 224, 224]
        # print(frame.shape)
        feature = net.extract_feature(frame).squeeze(0)  # [2048]
        # print(feature.shape)
    return feature.cpu().numpy()

class Extractor():
    def __init__(self):
        self.features = []
        self.frames = []

    def extract_frame(self, frame):
        self.features.append(extract_frame_feature(frame))
        return np.stack(self.features)

    def extract_video(self, video_path:str, progress=gr.Progress()):
        assert os.path.exists(video_path), f"File not found: {video_path}"
        vr = VideoReader(video_path, ctx=cpu(0))

        feature_list = []
        for idx in progress.tqdm(range(len(vr)), desc='Video: '):
            data = vr.get_batch(np.array([idx])).asnumpy()
            data = torch.from_numpy(data).to('cuda').permute(0,3,1,2)  # [8, 3, 384, 224]
            images = [transform(frame) for frame in data]  # 每帧单独处理
            input_data = torch.stack(images, dim=0)  # [8, 3, 224, 224]

            with torch.no_grad():
                feature = net.extract_feature(input_data).mean(dim=0)
                feature_list.append(feature.cpu().numpy())
        feature_list = np.stack(feature_list)
        self.features = feature_list # NOTE: save features
        return feature_list

    def extract_feature_multi(self, video_path:str, progress=gr.Progress()):
        assert os.path.exists(video_path), f"File not found: {video_path}"
        vr = VideoReader(video_path, ctx=cpu(0))

        feature_list = []
        for idx in tqdm(range(len(vr)), desc='Video: '):
            data = vr.get_batch(np.array([idx])).asnumpy()
            data = torch.from_numpy(data).to('cuda').permute(0,3,1,2)  # [8, 3, 384, 224]
            images = [transform(frame) for frame in data]  # 每帧单独处理
            input_data = torch.stack(images, dim=0)  # [8, 3, 224, 224]

            with torch.no_grad():
                feature = net.extract_feature(input_data).mean(dim=0)
                feature_list.append(feature.cpu().numpy())
        feature_list = np.stack(feature_list)
        self.features = feature_list # NOTE: save features
        return feature_list
    
    def recognize_video(self, video_path:str, progress=gr.Progress()):
        assert os.path.exists(video_path), f"File not found: {video_path}"
        vr = VideoReader(video_path, ctx=cpu(0))

        logits = []
        for idx in tqdm(range(0, len(vr)-15, 4), desc='Video: '):
            data = vr.get_batch(np.arange(idx, idx + 16, 2)).asnumpy()
            data = torch.from_numpy(data).to('cuda').permute(0,3,1,2)  # [8, 3, 384, 224]
            images = [transform(frame) for frame in data]  # 每帧单独处理
            input_data = torch.stack(images, dim=0)  # [8, 3, 224, 224]

            with torch.no_grad():
                output = net(input_data)
                recognize = torch.argmax(output, dim=1).detach().cpu().numpy()
                logits.append(recognize)

        recoginize_list = [actions_dict[i.item()] for i in logits]
        return recoginize_list
    
    def recognize_frame(self, frame):
        self.frames.append(frame)
        if len(self.frames) < 8:
            return 'waiting...', '0'
        else:
            input_frames = self.frames[-8:].copy()
            input_frames = torch.from_numpy(np.array(input_frames)).to('cuda').permute(0,3,1,2)  # [8, 3, 384, 224]
            input_frames = [transform(frame) for frame in input_frames]  # 每帧单独处理
            input_frames = torch.stack(input_frames)
            with torch.no_grad():
                output = net(input_frames)
                recognize = torch.argmax(output, dim=1).detach().cpu().numpy()
                output = torch.nn.Softmax(dim=1)(output)
                logit = torch.max(output, dim=1)[0].detach().cpu().numpy()

        recoginize_list = actions_dict[recognize[0].item()]
        return recoginize_list, str(logit[0].item())

    
    def clear(self):
        self.features = []
    

    
if __name__ == '__main__':
    extractor = Extractor()
    # for i in range(3):
    #     dummy_input = torch.randn(224, 224, 3)
    #     output = extractor.extract_frame(dummy_input)
    #     print(output.shape)
    outputs = extractor.recognize_video('/home/fitz_joye/TSM-action-recognition/data/assembly101/resized/nusar-2021_action_both_9022-a18_9022_user_id_2021-02-23_104757/C10115_rgb.mp4')
    print(outputs)