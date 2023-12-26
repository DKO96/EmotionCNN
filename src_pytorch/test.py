import os
import torch
from torch.utils.data import DataLoader
from glob import glob

from emotionDataset import EmotionDataset 
from emotionCNN import CNN


class TestCNN:
  def __init__(self):
    self.N_CLASSES = 5
    self.IMAGE_SIZE = 48

    # model setup
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    self.model = CNN(self.N_CLASSES).to(self.device)
    self.model.load_state_dict(torch.load("model.pt"))

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

  def testModel(self, dataset):
    dataset_idx = list(range(0, len(dataset)))
    test_loader = DataLoader(dataset, batch_size=1)
    print(f"Number of test samples: {len(dataset_idx)}")

    total_true = 0
    total = len(dataset_idx)
    with torch.no_grad():
      self.model.eval()
      for data_, target_ in test_loader:
        data_ = data_.to(self.device)
        target_ = target_.to(self.device)
        print(data_)

        outputs = self.model(data_)
        _, preds = torch.max(outputs, dim=1)
        true = torch.sum(preds == target_).item()
        total_true += true

    test_accuracy = round(100 * total_true/ total, 2)
    print(f"Test accuracy: {test_accuracy}")


def main():
  testCNN = TestCNN()

  # load dataset
  data_path = "../EmotionFace/test"
  label_map = {0:"angry", 1:"happy", 2:"neutral", 3:"sad", 4:"surprised"}  
  dataset = testCNN.loadDataset(data_path, label_map)

  # test model
  testCNN.testModel(dataset)


if __name__ == "__main__":
  main()