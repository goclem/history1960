#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Performs predictions for the Arthisto 1960 project
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.05.30
'''

#%% HEADER

# Modules
import numpy as np

from arthisto1960_utilities import *
from keras import layers, models
from os import path

#%% PREDICTS NEW TILES

# Loads model
# tensorflow.compat.v1.logging.get_verbosity()
# tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)
model = models.load_model(path.join(paths['models'], 'unet64_baseline.h5'))


# Lists batches
batch_size = 3
batches = search_files(paths['images'], pattern='tif$')
#batches = filter_identifiers(batches, search_files(paths['predictions'], pattern='tif$'))
batches = [batches[i:i + batch_size] for i in range(0, len(batches), batch_size)]
del batch_size

# Computes predictions
def predict_tiles(model, files):
    images  = np.array([read_raster(file) for file in files])
    images  = layers.Rescaling(1./255)(images)
    blocks1 = images_to_blocks(images=images, imagesize=images.shape, blocksize=(256, 256), shift=False)
    blocks2 = images_to_blocks(images=images, imagesize=images.shape, blocksize=(256, 256), shift=True)
    probas1 = model.predict(blocks1, verbose=1)
    probas2 = model.predict(blocks2, verbose=1)
    probas1 = blocks_to_images(blocks=probas1, imagesize=images.shape[:3] + (1,), blocksize=(256, 256), shift=False)
    probas2 = blocks_to_images(blocks=probas2, imagesize=images.shape[:3] + (1,), blocksize=(256, 256), shift=True)
    probas  = (probas1 + probas2) / 2
    return probas

# Computes predictions
for i, files in enumerate(batches):
    print('Batch {i:d}/{n:d}'.format(i=i + 1, n=len(batches)))
    probas   = predict_tiles(model=model, files=files)
    outfiles = [path.join(paths['predictions'], path.basename(file).replace('image', 'proba')) for file in files]
    for proba, file, outfile in zip(probas, files, outfiles):
        write_raster(array=proba, source=file, destination=outfile, nodata=None, dtype='float32')
del i, files, file, probas, proba, outfiles, outfile
# %%
