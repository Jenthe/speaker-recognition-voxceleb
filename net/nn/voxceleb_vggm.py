import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class SpatialCrossMapLRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, k=1, ACROSS_CHANNELS=True):
        super(SpatialCrossMapLRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        x = x.div(div)
        return x

class Net(nn.Module):
	def __init__(self, class_size):
		super(Net, self).__init__()

		self.class_size = class_size

		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=96, kernel_size=7, stride=2, padding=0),
			nn.ReLU(),
			nn.BatchNorm2d(96),
			# SpatialCrossMapLRN(5, 0.0005, 0.75, 2),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=False)
		)

		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(256),
			# SpatialCrossMapLRN(5, 0.0005, 0.75, 2),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=False)
		)

		self.conv3 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(256),
		)

		self.conv4 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(256),
		)

		self.conv5 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(256),
			nn.MaxPool2d(kernel_size=(5,3), stride=(3,2), padding=0, ceil_mode=False)
		)

		self.fc6 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=(9,1), stride=1, padding=0),
			nn.ReLU(),
			nn.BatchNorm2d(4096),
			# nn.Dropout(0.5)
		)

		self.apool6 = nn.Sequential(
			nn.AdaptiveAvgPool2d(output_size=(1, 1))
			#nn.AvgPool2d(kernel_size=(1,8), stride=1)
		)

		self.fc7 = nn.Sequential(
			nn.Linear(4096,1024),
			nn.ReLU(),
			nn.BatchNorm1d(1024),
			# nn.Dropout(0.5)
		)

		self.fc_b = nn.Sequential(
			nn.Linear(1024,512),
			# nn.ReLU(),
			# nn.BatchNorm1d(1024),
			# nn.Dropout(0.5)
		)



		self.fc8 = nn.Sequential(
			nn.Linear(512, self.class_size)
		)


	def forward(self, x):

		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)

		x = self.fc6(x)
		x = self.apool6(x)

		x = x.view(-1, 4096)

		x = self.fc7(x)

		x = self.fc_b(x)

		x = self.fc8(x)

		return x
