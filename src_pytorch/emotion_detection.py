import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from emotionCNN import CNN

def preprocess(image):
  transform = transforms.Compose([transforms.ToTensor()])
  TENSOR_IMAGE = transform(image)
  return TENSOR_IMAGE

def main():
  N_CLASSES = 5
  IMAGE_SIZE = 48

  label_map = {0:"angry", 1:"happy", 2:"neutral", 3:"sad", 4:"surprised"}

  # model setup
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = CNN(N_CLASSES).to(device)
  model.load_state_dict(torch.load("model.pt"))

  face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                       'haarcascade_frontalface_default.xml')

  image_size = (1920, 1080)
  cap = cv2.VideoCapture(0)

  cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size[0])
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size[1])

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(30, 30))
  
    for (x, y, w, h) in faces:
      roi = gray[y:y+h, x:x+w]
      roi = cv2.resize(roi, (IMAGE_SIZE, IMAGE_SIZE))
      roi = preprocess(roi).unsqueeze(0)
      roi = roi.to(device)

      with torch.no_grad():
        model.eval()
        output = model(roi)
        prob = torch.nn.functional.softmax(output, dim=1)
        print(prob)
        _, preds = torch.max(output, dim=1)
        label = label_map[preds.item()]

      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
      cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()




if __name__ == "__main__":
  main()
