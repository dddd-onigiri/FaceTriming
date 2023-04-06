import streamlit as st
import cv2

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = cv2.imread(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

# 顔検出
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# 顔をトリミング
for (x,y,w,h) in faces:
    cropped_face = image[y:y+h, x:x+w]

# トリミングされた顔を表示
st.image(cropped_face, caption='Cropped Image.', use_column_width=True)
