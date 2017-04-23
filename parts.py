import os
import shutil

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import skimage.io

DIR = './'
PARTS_DIR = 'parts/'


parts_csv = pd.read_csv(DIR + PARTS_DIR + 'part_locs.txt', delim_whitespace=True, header=None, names=['image_id', 'part_id', 'x', 'y', 'visible'])
parts_csv.drop(parts_csv[parts_csv.visible == 0].index, inplace=True)


centroids = parts_csv.groupby(['part_id'])[['x', 'y']].sum() / len(parts_csv)
print centroids
centroids.to_csv('centroids.csv')



