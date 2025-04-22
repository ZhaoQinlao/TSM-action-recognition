from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from action_detector import ActionDetector
from face_detector import detect_face_landmarks
from hand_detector import detect_hand_landmarks
from app_utils import Extractor

app = Flask(__name__)
socketio = SocketIO(app)

# 初始化人体动作检测器
detector = ActionDetector()
extractor = Extractor()

@app.route('/')
def index():
    return render_template('index.html')  # 主页，展示视频流

# 处理来自前端的视频流
@socketio.on('video_stream')
def handle_video_stream(data):
    # data 是二进制的字节数据
    np_arr = np.frombuffer(data, np.uint8)  # 这里直接使用二进制数据
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # H, W, C

    # 如果读取成功，进行动作检测
    if frame is not None:
        # 使用动作检测模型处理视频帧
        # frame = detector.detect(frame)
        # frame = detect_face_landmarks(frame)
        # frame = detect_hand_landmarks(frame)
        frame = extractor.extract_videos(frame)
        # print(frame.min(), frame.max(), frame.mean(), frame.std())
        frame = (frame - frame.min()) / (frame.max() - frame.min()) * 255
        frame = frame.astype(np.uint8)
        # print(frame.min(), frame.max(), frame.mean(), frame.std())


        # 将结果编码为JPEG格式，并转换为字节流发送回前端
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_bytes = buffer.tobytes()
            emit('video_response', frame_bytes)


if __name__ == '__main__':
    socketio.run(app, debug=True)
