import torch
import torchvision

from torchvision import transforms, datasets
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import numpy as np
from PIL import Image
import numpy as np
import cv2
conf = {
        "width": 480,
        "height": 480
        }


class Net(nn.Module):
    def __init__(self):  # constructor
        super(Net, self).__init__()  # parent constructor
        # in_channels, out_channels, kernel_size
        self.conv1 = nn.Conv2d(3, 10, 5, padding=2)  # 640 x 480 x 10
        # self.conv2 = nn.Conv2d(10, 16, 9, padding=4)  # 640 x 480 x 16
        self.pool1 = nn.MaxPool2d(2, 2)  # 320 x 240 x 16
        self.conv3 = nn.Conv2d(10, 10, 15, padding=7)  # 320 x 240 x 16
        self.pool2 = nn.MaxPool2d(2, 2)  # 160 x 120 x 16
        # self.conv4 = nn.Conv2d(16, 16, 15, padding=7)  # 160 x 120 x 16
        self.conv5 = nn.Conv2d(10, 10, 15, padding=7)  # 160 x 120 x 16
        self.pool3 = nn.MaxPool2d(4, 4)  # 40 x 30 x 16
        self.conv6 = nn.Conv2d(10, 4, 9, padding=4)  # 40 x 30 x 4
        self.pool4 = nn.MaxPool2d(2, 2)  # 20 x 15 x 4
        self.fc1 = nn.Linear(4 * conf["height"] * conf["width"] // (32 ** 2), 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        # x = F.leaky_relu(self.conv2(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.pool2(x)
        # x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = self.pool3(x)
        x = F.leaky_relu(self.conv6(x))
        x = self.pool4(x)
        x = x.view(-1, 4 * conf["height"] * conf["width"] // (32 ** 2))
        x = self.fc1(x)

        return x


norm = transforms.Compose([transforms.CenterCrop((conf["height"], conf["width"])),  # center crop a square
                           transforms.ToTensor(),  # now turn into a tensor
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize from [0,1] to [-1, 1]
                           ])
classes = ('looking', 'notlooking')
vid = cv2.VideoCapture(0)

net = Net()
net.load_state_dict(torch.load("./alpha.pt", map_location=torch.device('cpu')))

alpha = bravo = 0
with torch.no_grad():
    while True:
        return_value, frame = vid.read()
        # print(type(frame))
        # print(frame)
        frame = norm(Image.fromarray(frame))
        # print(frame.size())
        frame = torch.unsqueeze(frame,0)
        # print(frame.size())
        out = net.forward(frame)
        _, final = torch.max(out.data,1)
        f = final.item()
        print(f)
        if(alpha == bravo == f == 1):
            os.system("xrandr --output eDP-1 --brightness 0.1")
        else:
           os.system("xrandr --output eDP-1 --brightness 1")

        bravo = alpha
        alpha = f
