import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import sys
import importlib

from converter import Converter
from visualizer import Visualizer
from nn.voxceleb_vggm import Net
from utils import get_utterance_list
from utils import get_main_directory
from utils import get_speaker_weights
from utils import print_list

# Usage:
# python3 train.py default [ default/epoch_3_bs_64.pth 3 ]

sys.stdout.flush()

# vis = Visualizer()

e = importlib.import_module('constants.' + 'env')
c = importlib.import_module('constants.' + str(sys.argv[1]))
print("Imported constants from:", c)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net(c.CLASS_SIZE).to(device)
net.train()
print("Device:", device)

previous_epoch = -1

if (len(sys.argv) == 4):
	net.load_state_dict(torch.load(os.path.join(e.MODELS_DIR, sys.argv[2])))
	previous_epoch = int(sys.argv[3])

	print('Model:', sys.argv[2])
	print('Epoch:', sys.argv[3])

else:
	print('Model: no previous model loaded')

previous_epoch += 1

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

converter = Converter()
total_batches = 0
running_loss = 0.0

os.makedirs(os.path.join(e.MODELS_DIR, c.ID), exist_ok=True)

input = get_utterance_list(path=e.TRAIN_DIR, shuffle=True, class_size=c.CLASS_SIZE)
labels = torch.empty(c.BATCH_SIZE, dtype=torch.long)
batch = torch.empty(c.BATCH_SIZE, 1, 513, 301)

for epoch in range(c.EPOCHS):

	counter = 1

	for file, label in input:

		file = file[3:] # remove prefix

		spec = converter.audio_file_to_spectogram(
			path=file, sample_rate=c.SAMPLE_RATE, step_size=c.STEP_SIZE, crop_length=c.CROP_LENGTH, fft_length=c.FFT_LENGTH, window_width=c.WINDOW_WIDTH, log_c=c.LOG_C, random_crop=True,
			glob_var_normalized=c.GLOB_VAR_NORMALIZED
		)

		# batch = torch.empty(c.BATCH_SIZE, 1, int(spec.shape[0]), int(spec.shape[1]))
		batch[counter - 1][0] = spec
		labels[counter - 1] = label

		if counter % c.BATCH_SIZE == 0:

			optimizer.zero_grad()

			batch = batch.to(device)
			labels = labels.to(device)

			outputs = net(batch)
			loss = criterion(outputs, labels)

			loss.backward()
			optimizer.step()

			counter = 0
			total_batches += 1

			running_loss += loss.item()
			print(total_batches, ": ",  running_loss)

			# vis.update_learning_rate_graph(total_batches, running_loss)

			running_loss = 0.0

		counter += 1

	model = c.ID + '_epoch_' + str(epoch + previous_epoch) + '_bs_' + str(c.BATCH_SIZE) + '.pth'
	torch.save(net.state_dict(), os.path.join(e.MODELS_DIR, c.ID, model))

torch.save(net.state_dict(), os.path.join(e.MODELS_DIR, c.ID, c.ID + '_epoch_' + 'final' + '_bs_' + str(c.BATCH_SIZE) + '.pth'))

print('Finished Training')
