import os
import glob
import random
import shutil
import re
import librosa

if os.environ['HOME'] == '/Users/jenthe':

	print('Environment: local computer')

	TRAIN_DIR = '/Users/jenthe/Documents/School/Master/Masterproef/Code/VoxCeleb/data/train/wav'
	TEST_DIR = '/Users/jenthe/Documents/School/Master/Masterproef/Code/VoxCeleb/data/test/wav'
	MODELS_DIR = '/Users/jenthe/Documents/School/Master/Masterproef/Code/VoxCeleb/models'

elif os.environ['HOME'] == '/home/speech/jenthien':

	print('Environment: remote computer')

	TRAIN_DIR = '/home/speech/jenthien/VoxCeleb/data/train_verification_test_subset/wav'
	TEST_DIR = '/home/speech/jenthien/VoxCeleb/data/test/wav'
	MODELS_DIR = '/home/speech/jenthien/VoxCeleb/models'

def generate_test_set():

	dirs = [filename for filename in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, filename))]

	for dir in dirs:

		video_count_min = 5
		video_dir_min = ''

		for video_dir in os.listdir(os.path.join(TRAIN_DIR, dir)):

			video_count = len(os.listdir(os.path.join(TRAIN_DIR, dir, video_dir)))

			if (video_count >= video_count_min):
				video_count_min = video_count
				video_dir_min = video_dir

				os.makedirs(os.path.join(TEST_DIR, dir), exist_ok=True)
				shutil.move(os.path.join(TRAIN_DIR, dir, video_dir_min), os.path.join(TEST_DIR, dir))

				break

generate_test_set()
