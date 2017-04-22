import os
import shutil

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

DIR = './'
TRAIN_DIR = 'train/'
TEST_DIR = 'validation/'
IMAGE_DIR = 'images/'

images = pd.read_csv(DIR + 'images.txt', delim_whitespace=True, header=None, names=['id', 'path'])
datasetType = pd.read_csv(DIR + 'train_test_split.txt', delim_whitespace=True, header=None, names=['id', 'set'])
labels = pd.read_csv(DIR + 'image_class_labels.txt', delim_whitespace=True, header=None, names=['id', 'label'])
boxes = pd.read_csv(DIR + 'bounding_boxes.txt', delim_whitespace=True, header=None, names=['id', 'x', 'y', 'width', 'height'])

# id (int) | training or testing (int) | class label (int) | x | y | width | height
data = pd.concat([images, datasetType['set'], labels['label'], boxes['x'], boxes['y'], boxes['width'], boxes['height']], axis=1)

train = data[data['set'] == 1].reset_index(drop=True)
test = data[data['set'] == 0].reset_index(drop=True)

for index, row in train.iterrows():
	path = row['path'] 

	directories = os.path.dirname(path)

	src = IMAGE_DIR + path
	dst = TRAIN_DIR + path

	try:
		os.makedirs(TRAIN_DIR + directories)
		shutil.copy(src, dst)
	except OSError:
		shutil.copy(src, dst)

for index, row in test.iterrows():
	path = row['path'] 

	directories = os.path.dirname(path)

	src = IMAGE_DIR + path
	dst = TEST_DIR + path

	try:
		os.makedirs(TEST_DIR + directories)
		shutil.copy(src, dst)
	except OSError:
		shutil.copy(src, dst)

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)