import torch
import torchvision.transforms as transforms
from PIL import Image

transform = transforms.Compose([transforms.ToTensor()])

class InvalidDatasetException(Exception):
  def __init__(self, len_of_paths, len_of_labels):
    super().__init__(
      f"""Number of paths ({len_of_paths}) is not compatible with the number
        of labels ({len_of_labels})"""
    )

class EmotionDataset():
  def __init__(self, img_paths, img_labels, size_of_images):
    self.img_paths = img_paths
    self.img_labels = img_labels
    self.size_of_images = size_of_images
    if len(self.img_paths) != len(self.img_labels):
      raise InvalidDatasetException(self.img_paths, self.img_labels)
    
  def __len__(self):
    return len(self.img_paths)
  
  def __getitem__(self, index):
    # images are grayscaled
    PIL_IMAGE = Image.open(self.img_paths[index]).convert('L')\
                .resize(self.size_of_images)

    TENSOR_IMAGE = transform(PIL_IMAGE)
    label = self.img_labels[index]

    return TENSOR_IMAGE, label