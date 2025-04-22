import cv2

class ActionDetector:
    def __init__(self):
        # 这里假设你有一个加载好的动作检测模型，或者可以使用OpenCV的HOG等简化模型
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):        
        # 人体检测，检测结果返回的是bounding box
        # Draw a sample rectangle (x, y, width, height)
        cv2.rectangle(frame, (50, 50), (100, 100), (0, 255, 0), 2)
        
        #
        return frame  # 返回检测结果图像
