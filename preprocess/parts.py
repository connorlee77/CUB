import os
import shutil

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from skimage import transform as tf 
from scipy.misc import bytescale
import cv2


DIR = './'

def parse_parts(save=False):
	PARTS_DIR = 'parts/'
	parts_csv = pd.read_csv(DIR + PARTS_DIR + 'part_locs.txt', delim_whitespace=True, header=None, names=['image_id', 'part_id', 'x', 'y', 'visible'])
	parts_csv.drop(parts_csv[parts_csv.visible == 0].index, inplace=True)

	centroids = parts_csv.groupby(['part_id'], as_index=False)[['x', 'y']].mean().rename(columns={'x':'muX', 'y':'muY'})

	dontwant = [1, 3, 4, 8, 9, 12, 13, 14]
	for x in dontwant:
		centroids.drop(centroids[centroids.part_id == x].index, inplace=True)

	pX = np.array([185, 100, 138, 117, 60, 151, 141])
	pY = np.array([105, 63, 76, 104, 160, 102, 139])

	centroids['pX'] = pd.Series(np.array(pX), index=centroids.index)
	centroids['pY'] = pd.Series(np.array(pY), index=centroids.index)

	return parts_csv.merge(centroids, on=['part_id'])

def get_dataset():
	DIR = './'
	
	images = pd.read_csv(DIR + 'images.txt', delim_whitespace=True, header=None, names=['id', 'path'])
	datasetType = pd.read_csv(DIR + 'train_test_split.txt', delim_whitespace=True, header=None, names=['id', 'set'])
	data = pd.concat([images, datasetType['set']], axis=1).rename(columns={'id':'image_id'})

	parts = parse_parts()
	df = data.merge(parts, on=['image_id'])
	return df 

def to_canonical(height=224, width=224):

	WARPED_TRAIN_DIR = 'warp_train/'
	WARPED_TEST_DIR = 'warp_validation/'

	TRAIN_DIR = 'train/'
	TEST_DIR = 'validation/'

	CROP_TRAIN_DIR = 'crop_train/'
	CROP_TEST_DIR = 'crop_validation/'

	df = get_dataset()
	have = []
	for name, group in df.groupby('image_id'):

		if name % 100 == 0:
			print name
		have.append(name)

		path, isTrain = group.iloc[0]['path'], group.iloc[0]['set']

		# Get source file
		src = None
		if isTrain:
			src = TRAIN_DIR + path
		else:
			src = TEST_DIR + path
		

		img = cv2.imread(src)

		A = []
		B = []
		for i, row in group.iterrows():
			n, x, y, px, py =  row['part_id'], row['x'], row['y'], row['pX'], row['pY']

			A.append(np.array([row['x'], row['y']] ))
			B.append(np.array([row['pX'], row['pY']]))

		if len(A) < 3:
			have.pop()
			continue

		A = np.float32(A)
		B = np.float32(B)

		tform = tf.estimate_transform('affine', A, B)
		warped_img = tf.warp(img, inverse_map=tform.inverse) 

		warped_img = tf.resize(warped_img[:height, :width], (height, width))
		assert warped_img.shape == (height, width, 3)
		warped_img = bytescale(warped_img)

		#Create directory + file storage
		directories = os.path.dirname(path)
		dst = None
		directory_path = None
		
		if isTrain:
			dst = WARPED_TRAIN_DIR + path
			directory_path = WARPED_TRAIN_DIR + directories
		else:
			dst = WARPED_TEST_DIR + path
			directory_path = WARPED_TEST_DIR + directories

		try:
			os.makedirs(directory_path)
			cv2.imwrite(dst, warped_img)
		except OSError:
			cv2.imwrite(dst, warped_img)



	missing = []
	i = 1
	while i < 11788:
		if i not in have:
			if len(df[df['image_id'] == i]) == 0:
				i += 1
				continue

			path = df[df['image_id'] == i]['path'].iloc[0]
			isTrain = df[df['image_id'] == i]['set'].iloc[0]

			src = None
			if isTrain:
				src = CROP_TRAIN_DIR + path
			else:
				src = CROP_TEST_DIR + path

			directories = os.path.dirname(path)
			dst = None
			directory_path = None
			
			if isTrain:
				dst = WARPED_TRAIN_DIR + path
				directory_path = WARPED_TRAIN_DIR + directories
			else:
				dst = WARPED_TEST_DIR + path
				directory_path = WARPED_TEST_DIR + directories

			try:
				os.makedirs(directory_path)
				shutil.copy(src, dst)
			except OSError:
				shutil.copy(src, dst)

			missing.append(i)
		i += 1
	

		
to_canonical()