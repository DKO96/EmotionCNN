import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

emotion_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img_size = (1920, 1080)
cap = cv2.VideoCapture(0)

if type(img_size) is tuple:
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_size[0])
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size[1])

while True:
  ret, frame = cap.read()
  if not ret:
    break

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

  for (x, y, w, h) in faces:
    roi = gray[y:y+h, x:x+w]

    roi = cv2.resize(roi, (64, 64))
    roi = roi.astype('float')/255.0
    roi = np.expand_dims(roi, axis=0)
    roi = np.expand_dims(roi, axis=-1)

    emotion = emotion_model.predict(roi)
    emotion_index = np.argmax(emotion)
    emotion_text = emotion_labels[emotion_index]

    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

  cv2.imshow('Emotion Recognition', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()

