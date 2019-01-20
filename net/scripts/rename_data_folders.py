import os
import glob
import random
import shutil
import re
import librosa

TRAIN_DIR = '/home/speech/jenthien/VoxCeleb/data/test_verification/wav'

# This function will rename all directories so they become zero-indexed.
# For example: id00270, id00271 => id00000, id00001

def rename_data_folders():

	count = 0

	dirs = [filename for filename in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, filename))]
	dirs.sort()

	for dir in dirs:

		renamed_dir = 'id' + str(count).zfill(5)

		shutil.move(os.path.join(TRAIN_DIR, dir), os.path.join(TRAIN_DIR, renamed_dir))

		count += 1

rename_data_folders()
