#!/usr/bin/env python
# # -*- coding:utf-8 -*-

"""_summary_

============================================
内蔵カメラを使った顔認識アプリ 
============================================

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

print(__doc__)

import os
import cv2

MODEL           = 'haarcascade_frontalface_default.xml'
FACTOR          = 1.1
THICK           = 10
RED             = (0, 0, 255)

def detect_write(cascade, frame, b_faces, scale = FACTOR) :
    """ １つの画像の顔検出を行う
    Arg:
        cas   : cv2.CascadeClassifier
        fname : str : ファイル名
    """
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces    = cascade.detectMultiScale(gray, scaleFactor = scale) # scaleFactorを調べたい
    
    if len(faces) > 0 :   # 顔認識成功
        b_faces = faces
    else :                # 顔認識失敗
        faces = b_faces
    
    for (x, y, w, h) in b_faces :
        cv2.rectangle(frame, (x, y), (x + w, y + h),  color = RED, thickness = THICK)
    
    return frame, b_faces

# カメラのキャプチャを開始
cap = cv2.VideoCapture(0)  # 0は通常、内蔵カメラを指します。必要に応じて変更してください。
b_faces = []  # 顔認識に成功したfacesを保存するリスト

cascade = cv2.CascadeClassifier(MODEL)

if not cap.isOpened():
    print("カメラを開けませんでした。")
    exit()

while True:
    ret, frame = cap.read()  # フレームを読み込む
    if not ret:
        print("フレームを取得できませんでした。")
        break

    frame, b_faces = detect_write(cascade, frame, b_faces, FACTOR)
    cv2.imshow("Camera", frame)  # フレームを表示する

    # 'q'キーが押されたら終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
        
# カメラのキャプチャを解放し、ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
#os._exit(0)