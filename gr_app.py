import gradio as gr
from demo.app_utils import Extractor
from demo.hand_detector import detect_hand_landmarks
from demo.protas import predict_online_api, predict
import numpy as np
import torch

extractor = Extractor()

def get_features(frame):
    features = extractor.extract_frame(frame)
    # # normalize features
    # features = features / np.linalg.norm(features, axis=1, keepdims=True)
    # # convert to image
    # features = (features * 255).astype(np.uint8)
    return features

def recognize(frame):
    # feat = extractor.extract_video(frame)
    feats = np.load('/home/fitz_joye/TSM-action-recognition/input_x.npy')
    feats = torch.tensor(feats, dtype=torch.float).squeeze(0).permute(1,0)
    feat = feats.cpu().numpy()
    print(feat.shape)
    recognized = predict(feat)
    # recognized = predict_offline_api(extractor.features)
    return recognized



with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            transform = gr.Dropdown(choices=["test", "online"],
                                    value="online", label="Working Mode")
            input_img = gr.Image(sources=["webcam"], type="numpy")
        with gr.Column():
            refresh_btn = gr.Button("Refresh")
            output_img = gr.Image(streaming=True, type="numpy", label="Transformed Image")
            # output_features = gr.Image(streaming=True, type="numpy", label="Features")
            output_action = gr.Textbox(label="Recognized Action")
            logits = gr.Textbox(label="Logits")
        refresh_btn.click(extractor.clear, [], [])
        landmarks = input_img.stream(detect_hand_landmarks, [input_img], [output_img],
                                time_limit=None, stream_every=0.133, concurrency_limit=None)
        # feat = input_img.stream(get_features, [input_img], [output_features],
        #                         time_limit=None, stream_every=0.133, concurrency_limit=None)
        actions = input_img.stream(extractor.recognize_frame, [input_img], [output_action, logits],
                                   time_limit=None, stream_every=0.133, concurrency_limit=None)

# with gr.Blocks() as demo:
#     with gr.Row():
#         with gr.Column():
#             transform = gr.Textbox(value='demo', label="Working Mode")
#             input_video = gr.PlayableVideo(label="Input Video")
#         with gr.Column():
#             with gr.Row():
#                 extract_btn = gr.Button("Extract")
#                 recognize_btn = gr.Button("recognize")
#             features = gr.Image(type="numpy", label="Extracted Features")
#             output_actions = gr.Textbox(label="Recognized Action Sequence")
#     extract_btn.click(recognize, [input_video], [output_actions])
        
demo.launch(server_port=10010)
