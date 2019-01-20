import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import torch

class Converter():

	def calculate_mean(self, input, sample_rate, fft_length, step_size, window_width, log_c):

		y = 0

		result = [0] * 513

		for file, label in input:

			file = file[3:]

			S = self.audio_file_to_spectogram(path=file, sample_rate=sample_rate, step_size=step_size, crop_length=None, fft_length=fft_length, window_width=window_width, log_c=log_c, random_crop=False, normalized=False, glob_var_normalized=False, return_tensor=False)

			i = 0
			y += S.shape[1]

			for row in S:

				for val in row:
					result[i] += val

				i += 1

		for a in range(0, 513):
			result[a] = result[a] / y

		return result


	def calculate_std(self, input, mean, sample_rate, fft_length, step_size, window_width, log_c):

		y = 0

		result = [0] * 513

		for file, label in input:

			file = file[3:]

			S = self.audio_file_to_spectogram(path=file, sample_rate=sample_rate, step_size=step_size, crop_length=None, fft_length=fft_length, window_width=window_width, log_c=log_c, random_crop=False, normalized=False, glob_var_normalized=False, return_tensor=False)

			i = 0
			y += S.shape[1]

			for row in S:

				for val in row:
					val = (val - mean[i]) ** 2
					result[i] += val

				i += 1

		for a in range(0, 513):
			result[a] = math.sqrt(result[a] / y)

		return result


	def get_hamming_window_size(self, fft_length):
		window = np.hamming(fft_length)
		print(np.sum(window)) # Returns 552.5


	def audio_file_to_spectogram(self, path, sample_rate, crop_length, fft_length, step_size, window_width, log_c, random_crop, glob_var_normalized, normalized=True, return_tensor=True):
		offset = 0

		if (random_crop):
			duration = librosa.get_duration(filename=path, sr=sample_rate)
			max_offset = duration - crop_length
			offset = max_offset * random.random()

		audio, sr = librosa.core.load(path, mono=True, sr=sample_rate, duration=crop_length, offset=offset)
		S = librosa.stft(audio, n_fft=fft_length, hop_length=step_size, win_length=window_width, window='hamming')
		S = abs(S)

		S = np.log(S + log_c)
		# S = (S + abs(np.log(c))) / (abs(np.log(c)) + np.log(552.5 + c))

		if (normalized):

			for i in range(0, S.shape[0]):

				mean = np.mean(S[i])
				std = np.std(S[i])

				S[i] = S[i] - mean
				S[i] = S[i] / std

		if (return_tensor):
			S = torch.FloatTensor(S)

		return S
