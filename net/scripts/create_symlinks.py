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

	DEV_DATA = '/speech/data/VoxCeleb/VoxCeleb1/vox1_dev_wav/wav'

elif os.environ['HOME'] == '/home/speech/jenthien':

	print('Environment: remote computer')

	TRAIN_DIR = '/home/speech/jenthien/VoxCeleb/data/train_verification/wav'
	TEST_DIR = '/home/speech/jenthien/VoxCeleb/data/test_verification/wav'
	MODELS_DIR = '/home/speech/jenthien/VoxCeleb/models'

	DEV_DATA = '/speech/data/VoxCeleb/VoxCeleb1/vox1_test_wav/wav'

def generate_symlinks():

	dirs = [filename for filename in os.listdir(DEV_DATA) if os.path.isdir(os.path.join(DEV_DATA, filename))]

	for dir in dirs:

		os.makedirs(os.path.join(TRAIN_DIR, dir), exist_ok=True)

		for video_dir in os.listdir(os.path.join(DEV_DATA, dir)):

			os.symlink(os.path.join(DEV_DATA, dir, video_dir), os.path.join(TRAIN_DIR, dir, video_dir))

generate_symlinks()
