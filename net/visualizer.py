import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from converter import Converter

converter = Converter()

class Visualizer():

	def __init__(self):
		#plt.ion()
		plt.show()


	def update_learning_rate_graph(self, b, lr):
		plt.plot(b, lr, 'bo')
		plt.pause(0.0001)
		plt.draw()


	def plot_magnitude_frequency(self, array):
		plt.hist(array, bins=40)
		plt.title("Frequency diagram")
		plt.xlabel("Value")
		plt.ylabel("Frequency")
		plt.show()


	def show_spectogram(self, spec):
		plt.figure(figsize=(15, 5))
		librosa.display.specshow(spec.numpy(), sr=16000, x_axis='time', y_axis='linear')
		plt.show()


	def plot_magnitude_distribution(self, input, c):
		result = []

		for file, label in input:
			file = file[3:]
			print(file)
			spec = converter.audio_file_to_spectogram(
				path=file, sample_rate=c.SAMPLE_RATE, step_size=c.STEP_SIZE, crop_length=c.CROP_LENGTH_TEST,
				fft_length=c.FFT_LENGTH, window_width=c.WINDOW_WIDTH, log_c=c.LOG_C, random_crop=False,
				glob_var_normalized=c.GLOB_VAR_NORMALIZED
			)
			for row in spec:
				for val in row:
					result.append(val)

		self.plot_magnitude_frequency(result)


	def show_all_spectograms(self, input):
		for file, label in input:

			file = file[3:]
			spec = converter.audio_file_to_spectogram(file)
			self.show_spectogram(spec)
