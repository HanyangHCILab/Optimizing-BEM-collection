from torchsummary import summary
import timm
import numpy as np
import torch
import torch.nn as nn  # 신경망들이 포함됨
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import time
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self):
        # super함수는 CNN class의 부모 class인 nn.Module을 초기화
        super(CNN, self).__init__()
        self.model = timm.create_model('xception41p.ra3_in1k', pretrained=True, num_classes=7)
        #self.model = Xception()
        self.upsample2d =nn.Upsample(scale_factor = (1,5))


    def forward(self, x):

        out = self.upsample2d(x)
        out = self.model(out)
        return out
model = CNN()

device = "cpu"
model = model.to(device)
# from torchsummaryX import summary
# summary(model, torch.rand((1, 3, 150,18)).to(device))