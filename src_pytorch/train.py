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
from torch.utils.data import SubsetRandomSampler, DataLoader
from glob import glob
from sklearn.model_selection import train_test_split

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
    self.EPOCH_NUMBER = 100
    self.BATCH_SIZE = 64
    self.LEARNING_RATE = 0.0005

    # model setup
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    self.model = CNN(self.N_CLASSES).to(self.device)
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)

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
    total = 0
    correct = 0
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

      # compute accuracy
      _, pred = torch.max(outputs, 1)
      total += target_.size(0)
      correct += (pred == target_).sum().item()
      train_acc = correct / total

      # compute loss
      epoch_loss += loss.item()
    return epoch_loss, train_acc

  def _validationLoop(self, validation_batches):
    val_epoch_loss = 0.0
    total = 0
    correct = 0
    for data_, target_ in validation_batches:
      # load data and model
      target_ = target_.to(self.device)
      data_ = data_.to(self.device)

      # obtain output from model and compute loss
      outputs = self.model(data_)
      loss = self.criterion(outputs, target_)

      # compute accuracy
      _, pred = torch.max(outputs, 1)
      total += target_.size(0)
      correct += (pred == target_).sum().item()
      val_acc = correct / total

      # compute loss
      val_epoch_loss += loss.item()
    return val_epoch_loss, val_acc

  def trainModel(self, train_batches, validation_batches):
    training_loss = []
    training_acc = []
    validation_loss = []
    validation_acc = []
    best_val_loss = float("inf")

    print(f"Epochs: {self.EPOCH_NUMBER}\n"
          f"Batch size: {self.BATCH_SIZE}\n"
          f"Learning rate: {self.LEARNING_RATE}\n"
          )

    print(f"Epoch\tTrain loss\tTrain acc\tVal loss\tVal acc")

    start_time = time.time()
    for epoch in range(1, self.EPOCH_NUMBER+1):
      # training loop
      epoch_loss, epoch_acc = self._trainLoop(train_batches) 
      training_loss.append(epoch_loss/len(train_batches))
      training_acc.append(epoch_acc)

      # validation loop
      val_epoch_loss, val_acc = self._validationLoop(validation_batches) 
      validation_loss.append(val_epoch_loss/len(validation_batches))
      validation_acc.append(val_acc)

      # save model
      val_loss = val_epoch_loss/len(validation_batches)
      train_loss = epoch_loss/len(train_batches)
      if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(self.model.state_dict(), "model.pt")
        print(f"{epoch:<8}{train_loss:.4f}{'':<10}{epoch_acc:.4f}{'':<10}{val_loss:.4f}{'':<10}{val_acc:.4f}")

    end_time = time.time()
    print(f"Training time: {end_time - start_time}\n")
    return training_loss, training_acc, validation_loss, validation_acc

  def tuneModel(self, train_batches, validation_batches):
    df = pd.DataFrame(np.zeros((81,4)), columns=['fc', 'cv', 'train', 'val'])
    dropout = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"fc \t cv \t train \t validation")
    count = 0
    for i in dropout:
      self.model.do_fc = nn.Dropout(i).to(self.device)
      for j in dropout:
        self.model.do_cv = nn.Dropout(j).to(self.device)

        train_loss, validation_loss = self.trainModel(train_batches, 
                                                          validation_batches)
        best_train_loss = round(min(train_loss), 4)
        best_val_loss = round(min(validation_loss), 4)

        df.loc[count] = [i, j, best_train_loss, best_val_loss]
        print(f"{i} \t {j} \t {best_train_loss}  {best_val_loss}")
        count += 1

    df.to_csv('dropout_tuning.txt', sep='\t', index=False)

  def plotLoss(self, t_loss, t_acc, v_loss, v_acc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    ax1.plot(range(self.EPOCH_NUMBER), t_loss, c="b", label="Training")
    ax1.plot(range(self.EPOCH_NUMBER), v_loss, c="r", label="Validation")
    ax1.legend()
    ax1.set_xlabel("Number of Epochs")
    ax1.set_ylabel("Loss")
    
    ax2.plot(range(self.EPOCH_NUMBER), t_acc, c="b", label="Training")
    ax2.plot(range(self.EPOCH_NUMBER), v_acc, c="r", label="Validation")
    ax2.legend()
    ax2.set_xlabel("Number of Epochs")
    ax2.set_ylabel("Accuracy")
    plt.show()
    

def main():
  trainCNN = TrainCNN()

  # load dataset
  data_path = "../EmotionFace/train"
  label_map = {0:"angry", 1:"happy", 2:"neutral", 3:"sad", 4:"surprised"}  
  dataset = trainCNN.loadDataset(data_path, label_map)

  # split dataset
  train_batches, validation_batches = trainCNN.splitDataset(dataset)

  # train model
  t_loss, t_acc, v_loss, v_acc = trainCNN.trainModel(train_batches, 
                                                     validation_batches)

  # plot loss
  trainCNN.plotLoss(t_loss, t_acc, v_loss, v_acc)
  
  # tune model
  # trainCNN.tuneModel(train_batches, validation_batches)  
  


if __name__ == "__main__":
  main()






