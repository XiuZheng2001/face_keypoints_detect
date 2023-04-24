import torch
import torch.nn as nn


class Model(nn.Module):
    """
    input: H/W: 256*256, C: 3
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 96, kernel_size=(3, 3), padding=1)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(96)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(96, 128, kernel_size=(3, 3), padding=1)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(2)

        self.conv6 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)
        self.relu6 = nn.ReLU()
        self.bn6 = nn.BatchNorm2d(512)

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(32768, out_features=512)
        self.dp = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, out_features=88)  # 每幅图像有44个特征点， 分为xy坐标

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.bn5(x)
        x = self.pool5(x)

        x = self.conv6(x)
        x = self.relu6(x)
        x = self.bn6(x)

        x = self.flat(x)
        x = self.fc1(x)
        x = self.dp(x)
        x = self.fc2(x)

        return x
