import numpy as np
from tqdm import tqdm
import os

feat_root = '/home/fitz_joye/TSM-action-recognition/data/assembly101/semantic_features' 
output_root = '/home/fitz_joye/TSM-action-recognition/data/assembly101/semantic_features_concat'

for video_dir in tqdm(os.listdir(feat_root), ncols=70):
    video_dir_path = os.path.join(feat_root, video_dir)
    for view_dir in os.listdir(video_dir_path):
        output_view_dir = os.path.join(output_root, video_dir, view_dir)
        output_file_path = os.path.join(output_view_dir, 'features.npy')
        if os.path.exists(output_file_path):
            continue
        view_dir_path = os.path.join(video_dir_path, view_dir)
        print(view_dir_path)
        features = []
        for feature_file in sorted(os.listdir(view_dir_path)):
            feature_file_path = os.path.join(view_dir_path, feature_file)
            feature = np.load(feature_file_path)
            if len(feature.shape):
                feature = feature.reshape(-1, feature.shape[-1])
            features.append(feature)
        features = np.concatenate(features, axis=0)
        
        os.makedirs(output_view_dir, exist_ok=True)
        
        np.save(output_file_path, features)