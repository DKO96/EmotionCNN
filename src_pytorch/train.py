import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import SubsetRandomSampler, DataLoader
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn import metrics

from emotionDataset import EmotionDataset 
from emotionCNN import CNN
cudnn.deterministic = True
random.seed(1000)
torch.manual_seed(1000)
np.random.seed(1000)

class TrainCNN:
  def __init__(self):
    # training hyperparams
    self.N_CLASSES = 5
    self.IMAGE_SIZE = 48
    self.VALIDATION_PER = 0.2
    self.EPOCH_NUMBER = 60
    self.BATCH_SIZE = 64
    self.LEARNING_RATE = 0.0005

    # model setup
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    # self.model = CNN(self.N_CLASSES).to(self.device)
    self.model = models.vgg16(pretrained=False)
    self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features,
                                         self.N_CLASSES)
    self.model = self.model.to(self.device)
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
    print(self.model)

  def loadDataset(self, data_path, label_map):
    paths = []
    labels = []
    for i in range(self.N_CLASSES):
      folder = label_map[i]
      path_i = os.path.join(data_path, folder, "*")
      for file in glob(path_i):
        paths.append(file)
        labels.append(i)
    dataset = EmotionDataset(paths, labels, (self.IMAGE_SIZE, self.IMAGE_SIZE))
    return dataset

  def splitDataset(self, dataset):
    dataset_idx = list(range(0, len(dataset)))
    train_idx, test_idx = train_test_split(dataset_idx, 
                                           test_size=self.VALIDATION_PER,
                                           random_state=42)

    train_split = SubsetRandomSampler(train_idx)   
    train_batches = DataLoader(dataset, batch_size=self.BATCH_SIZE, 
                               sampler=train_split)  

    validation_split = SubsetRandomSampler(test_idx)   
    validation_batches = DataLoader(dataset, batch_size=self.BATCH_SIZE, 
                                    sampler=validation_split)  
    return train_batches, validation_batches

  def _trainLoop(self, train_batches):
    epoch_loss = 0.0
    for data_, target_ in train_batches:
      # load data and model
      target_ = target_.to(self.device)
      data_ = data_.to(self.device)

      # clean up gradients
      self.optimizer.zero_grad()

      # obtain output from model and compute loss
      outputs = self.model(data_)
      loss = self.criterion(outputs, target_)

      # backpropagation and optimizing model
      loss.backward()
      self.optimizer.step()

      # compute loss
      epoch_loss += loss.item()
    return epoch_loss

  def _validationLoop(self, validation_batches):
    val_epoch_loss = 0.0
    for data_, target_ in validation_batches:
        # load data and model
        target_ = target_.to(self.device)
        data_ = data_.to(self.device)

        # obtain output from model and compute loss
        outputs = self.model(data_)
        loss = self.criterion(outputs, target_)

        # compute loss
        val_epoch_loss += loss.item()
    return val_epoch_loss

  def trainModel(self, train_batches, validation_batches):
    train_loss = []
    validation_loss = []
    best_val_loss = float("inf")

    print(f"Epochs: {self.EPOCH_NUMBER}\n"
          f"Batch size: {self.BATCH_SIZE}\n"
          f"Learning rate: {self.LEARNING_RATE}\n"
          )

    start_time = time.time()
    for epoch in range(1, self.EPOCH_NUMBER+1):
      # training loop
      epoch_loss = self._trainLoop(train_batches) 

      # validation loop
      val_epoch_loss = self._validationLoop(validation_batches) 

      training_loss = epoch_loss/len(train_batches)
      train_loss.append(training_loss)
      validating_loss = val_epoch_loss/len(validation_batches)
      validation_loss.append(validating_loss)

      # save model
      if validating_loss < best_val_loss:
        best_val_loss = validating_loss
        torch.save(self.model.state_dict(), "model.pt")
        print(f"Epoch: {epoch}, Training loss: {training_loss:.4f}, \
              Validation loss: {validating_loss:.4f}")

    end_time = time.time()
    print(f"Training time: {end_time - start_time}\n")
    return train_loss, validation_loss

  def tuneModel(self, output_file, train_batches, validation_batches):
    # batch_size = [2, 4, 8, 16, 32, 64, 128, 256]
    # learning_rate = [0.001, 0.005, 0.01, 0.05, 0.01, 0.05, 0.1]
    # gamma = [0.5, 0.6, 0.7, 0.8, 0.9]

    batch_size = [32]
    learning_rate = [0.001, 0.005]
    
    f = open(output_file, "w")
    f.write(f"Batch size\t Train loss\t Validation loss\n")
    for i in batch_size:
      self.BATCH_SIZE = i
      train_loss, validation_loss = self.trainModel(train_batches, 
                                                        validation_batches)
      best_train_loss = min(train_loss)
      best_validation_loss = min(validation_loss)
      f.write(f"{i}\t\t\t\t\t   {best_train_loss:.4f}\t{best_validation_loss:.4f}\n")

    f.write(f"Learning rate\t Train loss\t Validation loss\n")
    for j in learning_rate:
      self.LEARNING_RATE = j
      train_loss, validation_loss = self.trainModel(train_batches, 
                                                        validation_batches)
      best_train_loss = min(train_loss)
      best_validation_loss = min(validation_loss)
      f.write(f"{j}\t\t\t\t\t   {best_train_loss:.4f}\t {best_validation_loss:.4f}\n")
    pass

  def plotLoss(self, train_loss, validation_loss):
    plt.subplots(figsize=(6, 4))
    plt.plot(range(self.EPOCH_NUMBER), train_loss, c="b", label="Training")
    plt.plot(range(self.EPOCH_NUMBER), validation_loss, c="r", label="Validation")
    plt.legend()
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.show()
    

def main():
  trainCNN = TrainCNN()

  # output
  output_file = "training_params.txt"

  # load dataset
  data_path = "../EmotionFace/train"
  label_map = {0:"angry", 1:"happy", 2:"neutral", 3:"sad", 4:"surprised"}  
  dataset = trainCNN.loadDataset(data_path, label_map)

  # split dataset
  train_batches, validation_batches = trainCNN.splitDataset(dataset)

  # train model
  train_loss, validation_loss = trainCNN.trainModel(train_batches, 
                                                    validation_batches)

  # tune model
  # trainCNN.tuneModel(output_file, train_batches, validation_batches)  
  
  # plot loss
  trainCNN.plotLoss(train_loss, validation_loss)


if __name__ == "__main__":
  main()






