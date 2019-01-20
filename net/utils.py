import os
import glob
import random
from numpy import array

main_directory = os.path.dirname(os.path.abspath(__file__))

def get_main_directory():
	return main_directory

def parse_meta_csv(file):

	data = {}

	lines = [line.strip() for line in open(main_directory + file, 'r').readlines()]

	for line in lines:

		line = line.split('\t')

		id = line[0]

		data[id] = {'name': line[1], 'gender': line[2], 'nationality:': line[3], 'set': line[4]}

	return data

def get_verify_utterances(path, class_size):

	data = [0] * class_size

	for i in range(class_size):
		data[i] = []

	dirs = [filename for filename in os.listdir(path) if os.path.isdir(os.path.join(path, filename))]

	for dir in dirs:

		number = int(dir[3:].lstrip("0") or "0")

		vid_dirs = [filename for filename in os.listdir(os.path.join(path, dir)) if os.path.isdir(os.path.join(path, dir))]

		data[number] = [0] * len(vid_dirs)

		for i in range(len(vid_dirs)):
			data[number][i] = []

		count = 0

		for vid_dir in vid_dirs:

			for filename in glob.iglob(path + '/' + dir + '/' + vid_dir + '/*.wav', recursive=True):
					data[number][count].append(filename)

			count += 1

	return data

def get_test_utterances(path, class_size):

	data = [0] * class_size

	for i in range(class_size):
		data[i] = []

	dirs = [filename for filename in os.listdir(path) if os.path.isdir(os.path.join(path, filename))]

	for dir in dirs:

		number = int(dir[3:].lstrip("0") or "0")

		for filename in glob.iglob(path + '/' + dir + '/**/*.wav', recursive=True):
				data[number].append(filename)

	return data

def get_utterance_list(path, shuffle, class_size, max=None, balance=False):

	data = {}

	count = [0] * class_size

	dirs = [filename for filename in os.listdir(path) if os.path.isdir(os.path.join(path, filename))]

	for dir in dirs:

		number = int(dir[3:].lstrip("0") or "0")
		round = 0

		for filename in glob.iglob(path + '/' + dir + '/**/*.wav', recursive=True):
			count[number] += 1
			if max == None or count[number] <= max:
				data["@" + str(round) + "@" + filename] = number

		round += 1

		while (balance and count[number] <= max):

			for filename in glob.iglob(path + '/' + dir + '/**/*.wav', recursive=True):
				count[number] += 1
				if max == None or count[number] <= max:
					data["@" + str(round) + "@" + filename] = number

			round += 1

	if shuffle == True:
		data = data.items()
		data = list(data)
		random.shuffle(data)

		return data

	return data.items()

def get_speaker_weights(input, n):
	weights = [0] * n
	total = 0

	for file, label in input:
		weights[int(label)] += 1
		total += 1

	weights = array(weights)

	return weights/total


def print_list(list):

	for file, label in list:
		print(file, ' => ', label)


def read_files(directory):

	speakers = parse_meta_csv('/data/meta/vox1_meta.csv')

	for dir in os.listdir(main_directory +  directory):
		print(dir + ' => ' + speakers[dir]['name'])
