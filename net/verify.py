import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import sys
import os
import importlib

from converter import Converter
from nn.voxceleb_vggm_verify import Net
from utils import get_verify_utterances

# Usage:
# python3 verify.py default_verification default_verification/default_verification_epoch_10_bs_64.pth

sys.stdout.flush()

e = importlib.import_module('constants.' + 'env')
c = importlib.import_module('constants.' + str(sys.argv[1]))
print("Imported constants from:", c)

class FeatureExtractor():

	def __init__(self, path, model):

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print("Device:", self.device)

		self.net = Net().to(self.device)
		self.net.load_state_dict(torch.load(os.path.join(e.MODELS_DIR, path, model)))
		self.net.eval()

		self.converter = Converter()

	def utterance_to_feature_vector(self, file):

		spec = self.converter.audio_file_to_spectogram(
			path=file, sample_rate=c.SAMPLE_RATE, step_size=c.STEP_SIZE, crop_length=c.CROP_LENGTH_TEST,
			fft_length=c.FFT_LENGTH, window_width=c.WINDOW_WIDTH, log_c=c.LOG_C, random_crop=False,
			glob_var_normalized=c.GLOB_VAR_NORMALIZED
		)

		spec_batch = torch.empty(1, 1, int(spec.shape[0]), int(spec.shape[1]))
		spec_batch[0][0] = spec

		spec_batch = spec_batch.to(self.device)

		return self.net(spec_batch)


def test(path):

	models = []

	if os.path.isdir(os.path.join(e.MODELS_DIR, path)):
		models = [filename for filename in os.listdir(os.path.join(e.MODELS_DIR, path)) if os.path.isfile(os.path.join(e.MODELS_DIR, path, filename)) and filename.lower().endswith('.pth')]
		models.sort()

		print(models)
	else:
		models.append(os.path.join(e.MODELS_DIR, path))

	test_input = get_verify_utterances(path=e.TEST_DIR_VERIFICATION, class_size=c.CLASS_SIZE_VERIFY)

	for model in models:

		extractor = FeatureExtractor(path, model)
		cos = nn.CosineSimilarity(0)

		with torch.no_grad():

			# Calculate feature vectors for each speech utterance of each speaker
			feature_vectors = []

			for speaker_id in range(len(test_input)):

				print("Speaker id: ", speaker_id)

				for vid_id in range(len(test_input[speaker_id])):

					for utt_id in range(len(test_input[speaker_id][vid_id])):

						x = F.normalize(extractor.utterance_to_feature_vector(test_input[speaker_id][vid_id][utt_id]), p=2, dim=0)
						feature_vectors.append((x, speaker_id, str(speaker_id) + str(vid_id) + str(utt_id)))


			# Calculate cosine distance between each pair of vectors
			feature_vectors_sim = []

			for a in feature_vectors:

				for b in feature_vectors:

					# Don't compare the same feature vectors
					if (a[2] != b[2]):
						sim = cos(a[0], b[0])
						feature_vectors_sim.append((a[1], b[1], sim))


			# Calculate FPR and FNR with varying threshold
			t = -1
			while t < 1:

				true_positive = 0
				true_negative = 0
				false_positive = 0
				false_negative = 0

				total = 0

				for a in range(len(feature_vectors_sim)):

					if (feature_vectors_sim[a][2] > t):
						if (feature_vectors_sim[a][0] == feature_vectors_sim[a][1]):
							true_positive += 1
						else:
							false_positive += 1
					else:
						if (feature_vectors_sim[a][0] != feature_vectors_sim[a][1]):
							true_negative += 1
						else:
							false_negative += 1

					total += 1

				FPR = false_positive / (false_positive + true_negative)
				FNR = false_negative / (false_negative + true_positive)

				print(t,';', FPR, ';', FNR)

				# print(true_positive, ';', false_positive, ';', true_negative, ';', false_negative, ';', total)

				with open(os.path.join(e.MODELS_DIR, c.ID, 'results.csv'), 'a+') as file:
					file.write(str(t) + ';' + str(FPR) + ';' + str(FNR))
					file.write('\n')

				t += 0.01


if __name__ == "__main__":
	if (len(sys.argv) != 3):
		print('3 arguments needed, check usage')
		sys.exit()

	test(sys.argv[2])
