# train.py
BATCH_SIZE = 52 # veelvoud 8
EPOCHS = 50

# test.py
CLASS_SIZE = 1251
CROP_LENGTH_TEST = None

# converter.py
CROP_LENGTH = 3 # s
SAMPLE_RATE = 16000
FFT_LENGTH = 64 # ms 32ms
WINDOW_WIDTH = 25 # ms
STEP_SIZE = 10 # ms, evt (9.5-10ms)
LOG_C = 0.0000001
GLOB_VAR_NORMALIZED = False
STD = None

ID = 'default_original'

# Conversions
WINDOW_WIDTH = int(SAMPLE_RATE / (1000 / WINDOW_WIDTH)) # points
STEP_SIZE = int(SAMPLE_RATE / (1000 / STEP_SIZE)) # points
FFT_LENGTH = int(SAMPLE_RATE / (1000 / FFT_LENGTH)) # points
