import cv2
import numpy as np
import streamlit as st

# 画像をアップロードする
uploaded_file = st.file_uploader("生徒証用の顔写真を選択してください", type="jpg")

if uploaded_file is not None:
    # 画像を読み込む
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

    # グレースケールに変換する
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 顔の検出器を初期化する
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 顔を検出する
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    # 検出された顔が1つ以上ある場合
    if len(faces) > 0:
        # 最初の顔をトリミングする
        (x, y, w, h) = faces[0]
        cropped_image = image[y:y + h, x:x + w]

        # トリミングされた画像を表示する
        st.image(cropped_image, caption='Cropped Image', use_column_width=True)

    else:
        st.write("顔が検出されませんでした。")
