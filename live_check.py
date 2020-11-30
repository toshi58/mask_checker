import keras
import cv2, dlib, pprint, os
import numpy as np
from keras.models import load_model
from beep import beep

# 結果ラベル
res_labels = ['NO MASK!', 'OK']

# 保存した学習データを読む
model = load_model('mask_model.h5')

# Dlib
detector = dlib.get_frontal_face_detector()

# Webカメラから入力を開始
red = (0, 0, 255)
green = (0, 255, 0)
fid = 1
cap = cv2.VideoCapture(0)
while True:
  # カメラの画像を読み込む
  ok, frame = cap.read()
  if not ok: break
  # 画像をリサイズ
  frame = cv2.resize(frame, (500, 300))
  # 顔検出
  dets = detector(frame, 1)
  for k, d in enumerate(dets):
    x1 = int(d.left())
    y1 = int(d.top())
    x2 = int(d.right())
    y2 = int(d.bottom())
    # 顔部分を切り取る
    im = frame[y1:y2, x1:x2]
    im = cv2.resize(im, (50, 50))
    im = im.reshape(-1, 50, 50, 3)
    # 予測
    res = model.predict([im])[0]
    v = res.argmax()
    # 枠を表示
    color = green if v == 1 else red
    border = 2 if v == 1 else 5
    cv2.rectangle(frame,
      (x1, y1), (x2, y2), color,
      thickness=border)
    # テキストを描画
    cv2.putText(frame,
      res_labels[v], (x1, y1-5),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8, color, thickness=1)
    if v == 0:
      beep(2000, 300)
  # ウィンドウに画像を出力
  cv2.imshow('Mask Live Check', frame)
  # Enterキーが押されたらループを抜ける
  k = cv2.waitKey(1)
  if k == 13: break

cap.release()
cv2.destroyAllWindows()