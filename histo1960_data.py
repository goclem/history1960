#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Utilities for the Arthisto1960 project
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.01.24
'''

#%% MODULES

# Modules
import argparse
import cv2
import imgaug
import numpy as np
import rasterio
import os
import keras
import matplotlib.pyplot as plt
import histo1960_functions as foo 

from easydict import EasyDict as edict
from itertools import compress

#%% PARAMETERS

# Parameters
params = edict({"seed": 1, "workers": 5, "gpu": True})
paths  = foo.set_paths()

# GPU computing
if params.gpu:
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

#%% Functions

def clahe(X, params):
    clahe   = cv2.createCLAHE(**params)
    X_clahe = []
    for channel in cv2.split(X):
        X_clahe.append(clahe.apply(channel))
    X_clahe = cv2.merge(X_clahe)
    return X_clahe

def compare(original, transformed):
    fig, ax = plt.subplots(1,2, figsize=(10, 20))
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].set_title('Original')
    ax[1].set_title(f'Transformed')
    ax[0].imshow(original)
    ax[1].imshow(transformed)
    plt.show()
    
def make_blocks(X, block_shape=(572, 572)):
    blocks = []
    for row in range(0, X.shape[0], block_shape[0]):
        for col in range(0, X.shape[1], block_shape[1]):
            blocks.append(X[row:row + block_shape[0], col:col + block_shape[1]])
    return blocks

def preprocess(X):
    X = clahe(X, params_clahe)
    X = cv2.cvtColor(X, cv2.COLOR_RGB2LAB)
    X = cv2.normalize(X, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return X

def crop_centre(img, block_shape=(572, 572)):
    xsize, ysize, _ = img.shape
    xmin  = int(xsize / 2 - block_shape[0] / 2)
    xmax  = int(xsize / 2 + block_shape[0] / 2)
    ymin  = int(ysize / 2 - block_shape[1] / 2)
    ymax  = int(ysize / 2 + block_shape[1] / 2)
    block_img = img[xmin:xmax, ymin:ymax, :]
    return block_img
    
#%% RASTERISE TRAINING DATA

files = foo.search_files(paths.yraw, pattern='\\d{4}_\\d{4}\\.gpkg', path=True, extension=False)
for file in files:
    print(foo.file_name(file))
    args = edict({
        'srcVecPath': file + '.gpkg',
        'srcRstPath': file + '.tif',
        'outRstPath': os.path.join(paths.y, 'y_{}.tif'.format(foo.file_name(file)))
    })
    foo.rasterise(**args)
del(files, args)

#%% RASTERISE LEGEND
tids = foo.get_tids(paths.yraw)
for tid in tids:
    print(tid)
    tile = foo.tile(tid)
    args = edict({
        'srcVecPath': os.path.join(paths.data, 'tiles', 'legends_1960.gpkg'),
        'srcRstPath': tile.paths().Xraw,
        'outRstPath': tile.paths().mask,
        'burnField' : 'ledgendid'
    })
    foo.rasterise(**args)
del(tids, tile, args)

# COMPUTE SUBTILES
'''
Tile:           5000 x 5000
Subtile input:  572 x 572
Subtile output: 388 x 388
Subtile margin: 92 each side
'''

'''
Data augmentation:
- Shift (from randomness)
- Rotate (i.e. 0-360)
- Flips (i.e. horizontal, vertical)
- Scaling (i.e. zoom)
- Brightness
- Noise (e.g. gaussian)
'''


tile  = foo.tile('1025_6320')
Xraw  = tile.Xraw.values()
block = Xraw[4000:4572, 3500:4072]

import skimage.transform
import skimage.util
import skimage.filters
import skimage.color
import skimage.exposure
import cv2

# BLOCKS
crop   = crop_centre(Xraw, (4576, 4576))
blocks = skimage.util.shape.view_as_blocks(crop, (572, 572, 3))
blocks = np.reshape(blocks, (8, 8, 572, 572, 3)) # Drop colour dimensions

foo.plot(blocks[0,0,...])
foo.plot(blocks[0,1,...])
foo.plot(blocks[0,1,...])
blocks.reshape(25, 572, 572, 3)
blocks.shape
25*572*572*3*572
# PRE-PROCESSING
lab   = cv2.cvtColor(block, cv2.COLOR_BGR2LAB)
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(10,10)) # Check parameters
lab[...,0] = clahe.apply(lab[...,0])
lab = skimage.exposure.rescale_intensity(lab)
foo.plot(lab)

# AUGMENTATION

# Spatial
rotate  = skimage.transform.rotate(block, 45, resize=True)
rotate  = crop_centre(rotate) # Solve block size


tensorflow.keras.layers.preprocessing.RandomRotation

rescale = skimage.transform.rescale(block, 2, multichannel=True)
rescale = crop_centre(rescale)
flip    = np.flipud(block)
flip    = np.fliplr(block)

# Noise
noise  = skimage.util.random_noise(block, var=0.01)
smooth = skimage.filters.gaussian(block)
compare(block, lab)
foo.plot(block)
foo.plot(lab)
foo.plot(rescale)
foo.plot(rotate)
foo.plot(flip)
foo.plot(noise)


# Make the pre-processing part of the model
# The training sample does not change

# layers.experimental.preprocessing.RandomRotation(factor=0.4, fill_mode="wrap"),
# layers.experimental.preprocessing.RandomTranslation(height_factor=0.2, width_factor=0.2, fill_mode="wrap"),
# layers.experimental.preprocessing.RandomFlip("horizontal"),
# layers.experimental.preprocessing.RandomContrast(factor=0.2),
# layers.experimental.preprocessing.RandomHeight(factor=0.2),
# layers.experimental.preprocessing.RandomWidth(factor=0.2)


#%% TESTS

tids  = foo.get_tids(paths.Xraw, prefix='_')
tiles = dict(zip([tid for tid in tids], [foo.tile(tid) for tid in tids]))
tile  = tiles["0875_6270"]
tile.y.plot()
tile.Xraw.table()
tile.X.values()
src.properties()
X = src.raw.values()
X = make_blocks(X)


for i, block in enumerate(X):
    path = os.path.join(paths.tmp, src.tid + "_" + str(i) + ".tif")
    cv2.imwrite(path, block)







X = preprocess(X)



    

foo.plot(X[10, ...])

y = tiles["0880_6260"].ohs.values()
y = np.where(y == 1, 1, 0)
y = make_blocks(y)



