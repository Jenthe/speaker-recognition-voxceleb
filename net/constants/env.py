import os

if os.environ['HOME'] == '/Users/jenthe':

	print('Environment: local computer')

	TRAIN_DIR = '/Users/jenthe/Documents/School/Master/Masterproef/Code/VoxCeleb/data/train/wav'
	TEST_DIR = '/Users/jenthe/Documents/School/Master/Masterproef/Code/VoxCeleb/data/test/wav'

	TRAIN_DIR_VERIFICATION = '/Users/jenthe/Documents/School/Master/Masterproef/Code/VoxCeleb/data/train_verification/wav'
	TEST_DIR_VERIFICATION = '/Users/jenthe/Documents/School/Master/Masterproef/Code/VoxCeleb/data/test_verification/wav'

	MODELS_DIR = '/Users/jenthe/Documents/School/Master/Masterproef/Code/VoxCeleb/models'

	DEV_DATA = '/speech/data/VoxCeleb/VoxCeleb1/vox1_dev_wav/wav'

elif os.environ['HOME'] == '/home/speech/jenthien':

	print('Environment: remote computer')

	TRAIN_DIR = '/home/speech/jenthien/VoxCeleb/data/train_verification/wav'
	TEST_DIR = '/home/speech/jenthien/VoxCeleb/data/test/wav'

	TRAIN_DIR_VERIFICATION = '/home/speech/jenthien/VoxCeleb/data/train_verification/wav'
	TEST_DIR_VERIFICATION = '/home/speech/jenthien/VoxCeleb/data/test_verification/wav'

	MODELS_DIR = '/home/speech/jenthien/VoxCeleb/models'

	DEV_DATA = '/speech/data/VoxCeleb/VoxCeleb1/vox1_dev_wav/wav'
