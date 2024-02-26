"""
@author: Huang Wen
@file: main.py
@time: 2024/2/20 17:51
@desc:
"""

import os
import cv2
import sys
import time
import torch
from ultralytics import YOLO
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets
from MainUI import Ui_MainWindow


_translate = QtCore.QCoreApplication.translate
weights_path = None  # 存放权重文件的路径
image_path = ''  # 存放待检测文件的路径
camera_flag = False  # 存储摄像头的状态，初始为关闭（False）
playing = False
model = None  # 存放检测模型
icon_start = QtGui.QIcon()
icon_pause = QtGui.QIcon()


cap = cv2.VideoCapture(0)

my_class = {0: '雏菊', 1: '蒲公英', 2: '玫瑰花', 3: '向日葵', 4: 'champaka', 5: 'chitrak', 6: '五色梅',
            7: '木槿花', 8: '金银花', 9: '梵天花', 10: '琴叶珊瑚', 11: '野牡丹',
            12: '万寿菊', 13: 'shankupushpam', 14: '彼岸花'}


def load_weights():
    file_path, file_type = QFileDialog.getOpenFileNames(MainWindow, '选择权重文件', os.getcwd(),
                                                        "PT文件(*.pt);;所有文件(*)")
    global weights_path, model
    if file_path:
        try:
            weights_path = str(file_path[0])
            model = YOLO(weights_path)
        except Exception as e:
            out_text('Exception:'+str(e))
            return
        out_text('已加载权重：' + weights_path)


def open_file():
    global image_path, cap, playing, icon_pause
    file_path, _ = QFileDialog.getOpenFileNames(MainWindow, '选择需要检测的图像或视频', os.getcwd(),
                                                        "所有文件(*);;图像文件(*.jpg;*.jpeg;*.png);;视频文件(*.mp4;*.avi)")
    if file_path:
        if not weights_path:
            out_text('文件加载失败，请先加载权重文件！')
            return
        ui.button_camera.setText(_translate("MainWindow", "打开摄像头"))
        file_type = file_path[0].split('.')[1]
        image_path = str(file_path[0])
        if file_type in ['mp4', 'avi']:
            try:
                cap = cv2.VideoCapture(image_path)
                if cap.isOpened():
                    out_text('已加载视频：' + image_path)
                    timer_pic.start(5)
                    playing = True
                    ui.button_playing.setVisible(True)
                    ui.button_playing.setText(_translate("MainWindow", "暂停"))
                    ui.button_playing.setIcon(icon_pause)
                else:
                    out_text('视频打开失败！')
            except Exception as e:
                out_text('Exception:'+str(e))
        else:
            try:
                cap.release()  # 如果有正在播放的视频，释放掉
                ui.button_playing.setVisible(False)  # 隐藏按钮
                # img = cv2.imread(image_path)
                out_text('已加载图像：' + image_path)
                if weights_path:  # 如果加载了权重文件，则执行检测功能
                    results = model(image_path)   # 调用检测模型

                    boxes_temp = results[0].boxes
                    boxes = boxes_temp.xyxy.tolist()
                    names = boxes_temp.cls.tolist()
                    conf = boxes_temp.conf.tolist()
                    for i, box in enumerate(boxes):
                        out_text('检测到：'+my_class[names[i]]
                                 + '；位置坐标为：'+str(list(map(int, box)))
                                 + '；置信度为：'+str(round(conf[i], 2)))

                    img = results[0].plot()
                    cur_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # 视频流的长和宽
                    height, width = cur_frame.shape[:2]
                    pixmap = QImage(cur_frame, width, height, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(pixmap)
                    # 获取是视频流和label窗口的长宽比值的最大值，适应label窗口播放，不然显示不全
                    ratio = max(width / ui.label_show.width(), height / ui.label_show.height())
                    pixmap.setDevicePixelRatio(ratio)
                    # 视频流置于label中间部分播放
                    ui.label_show.setAlignment(Qt.AlignCenter)
                    ui.label_show.setPixmap(pixmap)
            except Exception as e:
                out_text('Exception:'+str(e))


def open_camera():
    global camera_flag, weights_path, cap, playing, icon_pause
    if not camera_flag:
        if not weights_path:
            out_text('摄像头打开失败，请先加载权重文件！')
            return
        # 打开摄像头
        camera_flag = True
        playing = True
        ui.button_playing.setVisible(True)
        ui.button_playing.setText(_translate("MainWindow", "暂停"))
        ui.button_playing.setIcon(icon_pause)
        cap = cv2.VideoCapture(0)
        ui.button_camera.setText(_translate("MainWindow", "关闭摄像头"))
        out_text('摄像头已打开')
        timer_pic.start(5)
    else:
        # 关闭摄像头
        camera_flag = False
        playing = False
        timer_pic.stop()
        cap.release()
        ui.button_camera.setText(_translate("MainWindow", "打开摄像头"))
        ui.label_show.setText(_translate("MainWindow", "暂无数据输入"))
        ui.button_playing.setVisible(False)  # 隐藏按钮
        out_text('摄像头已关闭')


def clean():
    global cap, camera_flag
    camera_flag = False
    cap.release()
    timer_pic.stop()
    ui.label_show.setText(_translate("MainWindow", "暂无数据输入"))
    ui.text_output.setHtml(_translate("MainWindow", " "))
    ui.button_camera.setText(_translate("MainWindow", "打开摄像头"))
    ui.button_playing.setVisible(False)  # 隐藏按钮


def pause_and_start():
    global playing, icon_start, icon_pause
    if playing:
        playing = False
        ui.button_playing.setText(_translate("MainWindow", "播放"))
        ui.button_playing.setIcon(icon_start)
        timer_pic.stop()
    else:
        playing = True
        ui.button_playing.setText(_translate("MainWindow", "暂停"))
        ui.button_playing.setIcon(icon_pause)
        timer_pic.start(5)


def show_pic():
    # 执行检测功能
    global cap
    try:
        ret, img = cap.read()
        if ret:
            # 对每一帧进行检测
            with torch.no_grad():
                results = model(img)  # 调用检测模型
                img = results[0].plot()   # 调用检测模型
                boxes_temp = results[0].boxes
                boxes = boxes_temp.xyxy.tolist()
                names = boxes_temp.cls.tolist()
                conf = boxes_temp.conf.tolist()
                for i, box in enumerate(boxes):
                    out_text('检测到：' + my_class[names[i]]
                             + '；位置坐标为：' + str(list(map(int, box)))
                             + '；置信度为：' + str(round(conf[i], 2)))

            cur_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 视频流的长和宽
            height, width = cur_frame.shape[:2]
            pixmap = QImage(cur_frame, width, height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(pixmap)
            # 获取是视频流和label窗口的长宽比值的最大值，适应label窗口播放，不然显示不全
            ratio = max(width / ui.label_show.width(), height / ui.label_show.height())
            pixmap.setDevicePixelRatio(ratio)
            # 视频流置于label中间部分播放
            ui.label_show.setAlignment(Qt.AlignCenter)
            ui.label_show.setPixmap(pixmap)
    except Exception as e:
        out_text('Exception：'+str(e))


def out_text(text):  # 输出内容至“打印输出”区域
    timestamp = '[' + time.strftime("%Y/%m/%d-%H:%M") + ']'
    ui.text_output.append(timestamp + text)


if __name__ == '__main__':
    # train_flowers()  # 训练
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()  # 显示主界面
    icon_start.addPixmap(QtGui.QPixmap("source/start.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    icon_pause.addPixmap(QtGui.QPixmap("source/pause.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)

    ui.button_weights.clicked.connect(load_weights)  # 将按钮button_weights绑定至函数load_weights
    ui.button_file.clicked.connect(open_file)  # 将按钮button_file绑定至函数open_file
    ui.button_camera.clicked.connect(open_camera)  # 将按钮button_camera绑定至函数open_camera
    ui.button_playing.clicked.connect(pause_and_start)  # 将按钮button_playing绑定至函数pause_and_start
    ui.button_clean.clicked.connect(clean)  # 将按钮button_clean绑定至函数clean
    ui.button_playing.setVisible(False)  # 隐藏按钮
    timer_pic = QTimer()
    timer_pic.timeout.connect(show_pic)
    sys.exit(app.exec_())
