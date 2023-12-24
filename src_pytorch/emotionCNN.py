import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
  def __init__(self, n_classes):
    super(CNN, self).__init__()
    # convolutional layers
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    
    # convolutional initialization 
    self.cv1_init = nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
    self.cv2_init = nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
    self.cv3_init = nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
    self.cv4_init = nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='relu')
    self.cv5_init = nn.init.kaiming_normal_(self.conv5.weight, nonlinearity='relu')
    self.cv6_init = nn.init.kaiming_normal_(self.conv6.weight, nonlinearity='relu')
    self.cv7_init = nn.init.kaiming_normal_(self.conv7.weight, nonlinearity='relu')
    self.cv8_init = nn.init.kaiming_normal_(self.conv8.weight, nonlinearity='relu')

    # batch normalization layers
    self.bn1 = nn.BatchNorm2d(32)
    self.bn2 = nn.BatchNorm2d(64)
    self.bn3 = nn.BatchNorm2d(128)
    self.bn4 = nn.BatchNorm2d(256)

    self.bn_fc1 = nn.BatchNorm1d(64)

    # max pooling
    self.mp = nn.MaxPool2d(2, 2)

    # dropout layers
    self.do_cv = nn.Dropout(0.3)
    self.do_fc = nn.Dropout(0.5)

    # fully connected layers
    self.fc1 = nn.Linear(256 * 3 * 3, 64)
    self.fc2 = nn.Linear(64, 64)
    self.fc3 = nn.Linear(64, n_classes)

    # fully connected initializations
    self.fc1_init = nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
    self.fc2_init = nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
    self.fc3_init = nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')


  def forward(self, x):
    # print(f"x0: {x.size()}")
    x = self.conv1(x)
    x = F.relu(x)
    x = self.bn1(x)
    # print(f"x1: {x.size()}")

    x = self.conv2(x)
    x = F.relu(x)
    x = self.bn1(x)
    x = self.mp(x)
    x = self.do_cv(x)
    # print(f"x2: {x.size()}")

    x = self.conv3(x)
    x = F.relu(x)
    x = self.bn2(x)
    x = self.do_cv(x)
    # print(f"x3: {x.size()}")

    x = self.conv4(x)
    x = F.relu(x)
    x = self.bn2(x)
    x = self.mp(x)
    x = self.do_cv(x)
    # print(f"x4: {x.size()}")

    x = self.conv5(x)
    x = F.relu(x)
    x = self.bn3(x)
    # print(f"x5: {x.size()}")

    x = self.conv6(x)
    x = F.relu(x)
    x = self.bn3(x)
    x = self.mp(x)
    x = self.do_cv(x)
    # print(f"x6: {x.size()}")

    x = self.conv7(x)
    x = F.relu(x)
    x = self.bn4(x)
    # print(f"x7: {x.size()}")

    x = self.conv8(x)
    x = F.relu(x)
    x = self.bn4(x)
    x = self.mp(x)
    x = self.do_cv(x)
    # print(f"x8: {x.size()}")

    x = x.view(-1, 256 * 3 * 3)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.bn_fc1(x)
    x = self.do_fc(x)

    x = self.fc2(x)
    x = F.relu(x)
    x = self.bn_fc1(x)
    x = self.do_fc(x)

    x = self.fc3(x)
    x = F.relu(x)
    return x






