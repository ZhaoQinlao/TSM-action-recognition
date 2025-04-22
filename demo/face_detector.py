import cv2
import mediapipe as mp

# 初始化 MediaPipe FaceMesh 模块
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 初始化 FaceMesh 对象
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def detect_face_landmarks(frame):
    """
    使用 MediaPipe 对输入的图像帧进行人脸关键点检测，并返回绘制后的图像。

    参数:
    frame: 输入的图像帧 (BGR 格式)

    返回:
    frame_with_landmarks: 绘制了关键点和轮廓的图像帧
    """
    # 将图像从 BGR 转换为 RGB 格式，因为 MediaPipe 使用 RGB 图像
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 进行人脸关键点检测
    results = face_mesh.process(rgb_frame)

    # 如果检测到人脸
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 绘制每个关键点
            for landmark in face_landmarks.landmark:
                # 获取每个关键点的坐标
                h, w, _ = frame.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                # 绘制关键点
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # 绘制连接线（面部轮廓）
            # mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

    # 返回处理过的帧
    return frame

# 使用例子
if __name__ == "__main__":
    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 调用函数检测人脸关键点
        frame_with_landmarks = detect_face_landmarks(frame)

        # 显示带有关键点和轮廓的图像
        cv2.imshow('Face Mesh', frame_with_landmarks)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()
