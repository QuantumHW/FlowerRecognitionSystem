"""
@author: Huang Wen
@file: train.py
@time: 2024/2/25 20:39
@desc: 
"""

import cv2
from ultralytics import YOLO
from cv2 import getTickCount, getTickFrequency


def train_flowers():
    model = YOLO('./weights/yolov8n.pt')  # 加载预训练模型
    # 训练模型
    results = model.train(data='datasets/flowers/flowers.yaml',
                          epochs=1000, imgsz=640, amp=False)


def inference_camera(path_model):  # 通过摄像头推理
    model = YOLO(path_model)  # 加载模型
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        loop_start = getTickCount()
        ret, frame = cap.read()
        if ret:
            res = model(frame)
            image = res[0].plot()
            # flipped_image = cv2.flip(image, 1)  # 左右翻转画面
            loop_time = getTickCount() - loop_start
            total_time = loop_time / (getTickFrequency())
            fps = int(1 / total_time)
            # 在图像左上角添加FPS文本
            fps_text = f"FPS: {fps:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_position = (10, 30)  # 左上角位置
            cv2.putText(image, fps_text, text_position, font, fontScale=1, color=(0, 0, 255), thickness=2)
            cv2.imshow("Camera", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    cap.release()


def test(path_model):
    model = YOLO(path_model)  # 加载模型
    results = model.predict(source="./img/demo.mp4", save=True, save_txt=False)


if __name__ == '__main__':
    train_flowers()  # 训练
