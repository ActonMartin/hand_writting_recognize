import os
from glob import glob

import matplotlib.patches as patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchsummary import summary
from torchvision import transforms
from torchvision.models.mobilenet import MobileNetV2
from torchvision.models.resnet import resnet34, resnet50
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm


class DigitsMobilenet(nn.Module):
    def __init__(self, class_num=26):
        super(DigitsMobilenet, self).__init__()
        self.net = nn.Sequential(
            MobileNetV2(num_classes=class_num).features,
            nn.AdaptiveAvgPool2d((1, 1)))
        self.fc1 = nn.Linear(1280, class_num)
        self.fc2 = nn.Linear(1280, class_num)
        self.fc3 = nn.Linear(1280, class_num)
        self.fc4 = nn.Linear(1280, class_num)
        self.fc5 = nn.Linear(1280, class_num)
        self.fc6 = nn.Linear(1280, class_num)
        self.fc7 = nn.Linear(1280, class_num)
        self.fc8 = nn.Linear(1280, class_num)
        self.fc9 = nn.Linear(1280, class_num)
        self.fc10 = nn.Linear(1280, class_num)
        self.fc11 = nn.Linear(1280, class_num)
        self.fc12 = nn.Linear(1280, class_num)
        self.fc13 = nn.Linear(1280, class_num)
        self.fc14 = nn.Linear(1280, class_num)
        self.fc15 = nn.Linear(1280, class_num)
        self.fc16 = nn.Linear(1280, class_num)
        self.fc17 = nn.Linear(1280, class_num)
        self.fc18 = nn.Linear(1280, class_num)
        self.fc19 = nn.Linear(1280, class_num)
        self.fc20 = nn.Linear(1280, class_num)
        self.fc21 = nn.Linear(1280, class_num)

    def forward(self, img):
        features = self.net(img).view(-1, 1280)
        fc1 = self.fc1(features)
        fc2 = self.fc2(features)
        fc3 = self.fc3(features)
        fc4 = self.fc4(features)
        fc5 = self.fc5(features)
        fc6 = self.fc6(features)
        fc7 = self.fc7(features)
        fc8 = self.fc8(features)
        fc9 = self.fc9(features)
        fc10 = self.fc10(features)
        fc11 = self.fc11(features)
        fc12 = self.fc12(features)
        fc13 = self.fc13(features)
        fc14 = self.fc14(features)
        fc15 = self.fc15(features)
        fc16 = self.fc16(features)
        fc17 = self.fc17(features)
        fc18 = self.fc18(features)
        fc19 = self.fc19(features)
        fc20 = self.fc20(features)
        fc21 = self.fc21(features)

        return fc1, fc2, fc3, fc4, fc5, fc6, fc7, fc8, fc9, fc10, fc11, fc12, fc13, fc14, fc15, fc16, fc17, fc18, fc19, fc20, fc21


if __name__ == "__main__":
    net = DigitsMobilenet()
    print(net)
    net.to(t.device('cuda'))
    summary(net, input_size=(3, 280, 50), batch_size=1)
