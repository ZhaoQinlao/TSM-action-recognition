import gradio as gr
from demo.app_utils import Extractor
from demo.hand_detector import detect_hand_landmarks
from demo.protas import predict_online_api
import numpy as np

extractor = Extractor()

def get_features(frame):
    features = extractor.extract_videos(frame)
    # normalize features
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    # convert to image
    features = (features * 255).astype(np.uint8)
    return features

def recognize(frame):
    feat = get_features(frame)
    recognized = predict_online_api(feat)
    return recognized

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            transform = gr.Dropdown(choices=["cartoon", "edges", "flip"],
                                    value="flip", label="Transformation")
            input_img = gr.Image(sources=["webcam"], type="numpy")
            refresh_btn = gr.Button("Refresh")
        with gr.Column():
            output_img = gr.Image(streaming=True, type="numpy", label="Transformed Image")
            output_features = gr.Image(streaming=True, type="numpy", label="Features")
            output_action = gr.Textbox(label="Recognized Action")
        refresh_btn.click(extractor.clear, [], [])
        feat = input_img.stream(get_features, [input_img], [output_features],
                                time_limit=None, stream_every=0.03, concurrency_limit=None)
        landmarks = input_img.stream(detect_hand_landmarks, [input_img], [output_img],
                                     time_limit=None, stream_every=0.03, concurrency_limit=None)
        actions = input_img.stream(recognize, [input_img], [output_action],
                                   time_limit=None, stream_every=0.03, concurrency_limit=None)

demo.launch()
