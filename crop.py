import os
import shutil

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import skimage.io

DIR = './'
TRAIN_DIR = 'train/'
TEST_DIR = 'validation/'

CROP_TRAIN_DIR = 'crop_train/'
CROP_TEST_DIR = 'crop_validation/'

train_CSV = pd.read_csv(DIR + 'train.csv')
test_CSV = pd.read_csv(DIR + 'test.csv')

for index, row in train_CSV.iterrows():
	path = row['path']
	x = int(row['x'])
	y = int(row['y'])
	width = int(row['width'])
	height = int(row['height'])


	directories = os.path.dirname(path)
	src = TRAIN_DIR + path
	dst = CROP_TRAIN_DIR + path

	pic = skimage.io.imread(src)
	cropped = pic[y:y+height, x:x+width]

	try:
		os.makedirs(CROP_TRAIN_DIR + directories)
		skimage.io.imsave(dst, cropped)
	except OSError:
		skimage.io.imsave(dst, cropped)

for index, row in test_CSV.iterrows():
	path = row['path']
	x = int(row['x'])
	y = int(row['y'])
	width = int(row['width'])
	height = int(row['height'])

	directories = os.path.dirname(path)
	src = TEST_DIR + path
	dst = CROP_TEST_DIR + path

	pic = skimage.io.imread(src)
	cropped = pic[y:y+height, x:x+width]

	try:
		os.makedirs(CROP_TEST_DIR + directories)
		skimage.io.imsave(dst, cropped)
	except OSError:
		skimage.io.imsave(dst, cropped)


