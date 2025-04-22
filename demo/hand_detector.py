import cv2
import mediapipe as mp
import numpy as np

# 初始化 MediaPipe Hands 模块
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 创建 Hands 对象（可放外部，避免重复初始化）
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def detect_hand_landmarks(frame):
    """
    使用 MediaPipe 对输入的图像帧进行手部关键点检测，并返回绘制后的图像。

    参数:
    frame: 输入的图像帧 (BGR 格式)

    返回:
    frame_with_landmarks: 绘制了关键点和骨架的图像帧
    """
    # 转换为 RGB 图像
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.copy(frame)  # 创建副本以绘制关键点

    # 手部关键点检测
    results = hands.process(rgb_frame)

    # 如果检测到手
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                h, w, _ = frame.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # 可选：绘制骨架连接
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return frame


