#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Prepares data for the Arthisto1960 project
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.03.18
'''

#%% MODULES
import numpy as np
import pandas as pd

from arthisto1960_utilities import *
from os import path
from scipy import ndimage, stats
from sklearn import ensemble, impute
from skimage import color, feature, segmentation, morphology

paths = dict(
    images ='../data_1960/images', 
    labels ='../data_1960/labels',
    desktop='/Users/clementgorin/Desktop'
)

#%% FUNCTIONS

# Computes the median intensity of each segment for each channel
def segment_mean(image:np.ndarray, segment:np.ndarray) -> np.ndarray:
    nsegment = len(np.unique(segment))
    channels = np.dsplit(image, image.shape[-1])
    means    = [ndimage.mean(channel, labels=segment, index=np.arange(0, nsegment)) for channel in channels]
    means    = np.column_stack(means)
    return means

def segment_variance(image:np.ndarray, segment:np.ndarray) -> np.ndarray:
    nsegment  = len(np.unique(segment))
    channels  = np.dsplit(image, image.shape[-1])
    variances = [ndimage.variance(channel, labels=segment, index=np.arange(0, nsegment)) for channel in channels]
    variances = np.column_stack(variances)
    return variances

def segment_argmax(label:np.ndarray, segment:np.ndarray) -> np.ndarray:
    table  = pd.DataFrame({'segment': segment.flatten(), 'label': label.flatten()})
    table  = pd.crosstab(table.segment, table.label)
    argmax = table.idxmax(axis=1).to_numpy()
    return argmax

def check(array:np.ndarray, label:str='', display=True, dtype:str='uint8') -> None:
    outfile = path.join(paths['desktop'], 'label_0650_6870_{}.tif'.format(label))
    if array.ndim == 2:
        array = np.expand_dims(array, 2)
    write_raster(array, imagefile, outfile, dtype=dtype)
    if display:
        os.system('open {}'.format(outfile))

#%% DATA
imagefile = path.join(paths['images'],  'image_0650_6870.tif')
labelfile = path.join(paths['desktop'], 'label_0650_6870.tif')
image     = read_raster(imagefile)

#%% SEGMENTATION

# Quickshift segmentation
segment = segmentation.quickshift(image, ratio=1, kernel_size=5,  max_dist=10, sigma=0, convert2lab=True, random_seed=1)
check(segment, 'segment', 'uint32')

# Sieving and harmonize identifiers
os.system('gdal_sieve.py {srcfile} {outfile} -st 4'.format(srcfile=labelfile.replace('.tif', '_segment.tif'), outfile=labelfile.replace('.tif', '_sieved.tif')))
segment = read_raster(labelfile.replace('.tif', '_sieved.tif'), dtype='uint32')
rank    = stats.rankdata(segment.flatten(), method='dense') - 1
segment = rank.reshape(segment.shape)
del rank

#%% LABELS AND VARIABLES
# Variables
image_lab  = color.rgb2lab(image)
image_lbp  = np.dstack([feature.local_binary_pattern(color.rgb2gray(image), P=R*8, R=R) for R in [1, 2, 3]])
image_var  = np.dstack((image_lab, image_lbp))
del image_lab, image_lbp 

value = np.column_stack((
    segment_mean(image_var, segment),
    segment_variance(image_var, segment),
    np.bincount(segment.flatten())
))
imputer = impute.SimpleImputer(strategy='mean')
value   = imputer.fit_transform(value)
del imputer
# check(value[...,6][segment], 'size')

#%% PREDICT LABELS
os.system('gdal_rasterize {srcfile} {outfile} -q -a class -te 650000 6845000 675000 6870000 -tr 5 5 -a_nodata 0 -ot Byte'.format(srcfile=labelfile.replace('.tif', '.gpkg'), outfile=labelfile))
label = read_raster(labelfile)
label = segment_argmax(label, segment)

# Training sample
train = np.where(label > 0)
label_train = label[train]
label_train = np.where(label_train == 2, 0, label_train)
value_train = value[train]
del train

model = ensemble.RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', n_jobs=-1)
model.fit(value_train, label_train)
del value_train, label_train

pred = model.predict(value)
pred = pred[segment]
check(pred, '', display=False)
# %%
