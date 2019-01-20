import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
	def __init__(self):

		super(Net, self).__init__()

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
		)

		self.fc7 = nn.Sequential(
			nn.Linear(4096,1024),
			# nn.ReLU(),
			# nn.BatchNorm1d(1024)
			# nn.Dropout(0.5)
		)

		self.fc8 = nn.Sequential(
			nn.Linear(1024, 1211)
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
		#x = self.fc_b(x)

		#c = 0

		#for l in self.fc7.children():

		#	if (c != 1):
		#		x = l(x)

		#	c += 1

		# x = self.fc7(x)
		# x = self.fc8(x)

		x = x.view(1024)

		return x
