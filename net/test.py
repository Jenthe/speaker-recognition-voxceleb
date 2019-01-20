import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sys
import os
import importlib

from converter import Converter
from nn.voxceleb_vggm_clean import Net
from utils import get_utterance_list, get_test_utterances
from utils import get_main_directory
from utils import print_list

# Usage:
# python3 test.py default (default | default/epoch0.pth)

sys.stdout.flush()

e = importlib.import_module('constants.' + 'env')
c = importlib.import_module('constants.' + str(sys.argv[1]))
print("Imported constants from:", c)

def test(path, epoch, write_to_file):

	models = []

	if os.path.isdir(os.path.join(e.MODELS_DIR, path)):
		models = [filename for filename in os.listdir(os.path.join(e.MODELS_DIR, path)) if os.path.isfile(os.path.join(e.MODELS_DIR, path, filename)) and filename.lower().endswith('.pth')]
		models.sort()
		print(models)

	else:
		models.append(os.path.join(e.MODELS_DIR, path))

	test_input = get_test_utterances(path=e.TEST_DIR, class_size=c.CLASS_SIZE)
	test_batch_size = 1
	test_labels = torch.empty(test_batch_size, dtype=torch.long)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("Device:", device)

	for model in models:

		net = Net(c.CLASS_SIZE).to(device)
		net.load_state_dict(torch.load(os.path.join(e.MODELS_DIR, path, model)))
		net.eval()

		converter = Converter()

		with torch.no_grad():

			correct = 0
			correct_top_k_5 = 0
			total = 0

			for i in range(len(test_input)):

				predictions = torch.zeros(c.CLASS_SIZE)
				predictions = predictions.to(device)

				for a in range(len(test_input[i])):

					file = test_input[i][a]

					spec = converter.audio_file_to_spectogram(
						path=file, sample_rate=c.SAMPLE_RATE, step_size=c.STEP_SIZE, crop_length=c.CROP_LENGTH_TEST, fft_length=c.FFT_LENGTH, window_width=c.WINDOW_WIDTH, log_c=c.LOG_C, random_crop=False,
					 	glob_var_normalized=c.GLOB_VAR_NORMALIZED
					)

					test_batch = torch.empty(test_batch_size, test_batch_size, int(spec.shape[0]), int(spec.shape[1]))
					test_batch[0][0] = spec
					test_labels[0] = i

					test_batch = test_batch.to(device)
					test_labels = test_labels.to(device)

					test_outputs = net(test_batch)
					test_outputs = torch.nn.functional.softmax(input=test_outputs)

					predictions = predictions + test_outputs

				predictions = predictions / len(test_input[i])
				percentage , predicted = torch.max(predictions, 1)
				_, predicted_top_k_5 = torch.topk(predictions, 5, 1)

				print('Should be: ', i)
				print('Predicted : ', predicted[0].item())
				print('Percentage: ', round(percentage[0].item() * 100, 2), '%')

				total += test_labels.size(0)
				correct += (predicted == test_labels).sum().item()

				if (test_labels[0].item() in np.array(predicted_top_k_5)):
					correct_top_k_5 += 1

				print(correct, '/', total)
				print(correct_top_k_5, '/', total)
				print(" ")

			if (write_to_file):
				with open(os.path.join(e.MODELS_DIR, c.ID, 'results.csv'), 'a+') as file:
					file.write(str(model) + ';' + str(correct) + ';' + str(correct_top_k_5) + ';' + str(total))
					file.write('\n')

if __name__ == "__main__":
	if (len(sys.argv) != 3):
		print('3 arguments needed, check usage')
		sys.exit()

	test(sys.argv[2], epoch=-1, write_to_file=False)
