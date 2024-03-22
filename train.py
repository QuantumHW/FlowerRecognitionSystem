"""
@author: Huang Wen
@file: train.py
@time: 2024/2/25 20:39
@desc: 
"""

import cv2
from ultralytics import YOLO
from cv2 import getTickCount, getTickFrequency


def train():
    model = YOLO('./weights/yolov8n.pt')  # 加载预训练模型
    # 训练模型
    results = model.train(data='datasets/flowers/flowers.yaml',
                          epochs=1000, imgsz=640, amp=False)


def train_MLCA():
    # 训练新增注意力机制后的模型
    model = YOLO('ultralytics/cfg/models/v8/atten_MLCA_yolov8n.yaml')  # 加载网络模型
    # 训练模型
    results = model.train(data='datasets/flowers/flowers.yaml',
                          epochs=1000, imgsz=640, amp=False)


def test(path_model):
    model = YOLO(path_model)  # 加载模型
    results = model.predict(source="./img/demo.mp4", save=True, save_txt=False)


if __name__ == '__main__':
    # train()
    train_MLCA()
