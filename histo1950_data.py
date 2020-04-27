#!/usr/bin/env conda run -n arthisto1950 python
# -*- coding: utf-8 -*-

"""
@description: Extracts buildings from SCAN50 maps
@author: Clement Gorin
@contact: gorin@gate.cnrs.fr
@date: April 2020
"""

#%% Utilities

# Modules
import argparse
import cv2
import imgaug
import numpy as np
import rasterio
import os
from histo1950_functions import *

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--seed",      type=int,   default=1,  help="Session seed")
parser.add_argument("--cliplimit", type=float, default=3,  help="CLAHE clip limit")
parser.add_argument("--gridsize",  type=int,   default=10, help="CLAHE gridsize")
parser.add_argument("--workers",   type=int,   default=6,  help="Maximum number of workers")
params = parser.parse_args()
paths  = utils.set_paths()

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend" # GPU backend


#%% Tile class

"""
- Define an abstract class tile
- Every tile in the dataset would be an instance
- Methods could include "show" tile.show() to plot it
- Methods could include "path"
- Methods could include "log" log.update()
"""

raw
paths = makePaths()
"0880_6260"

class tile():
    def show()
    def raw()
    def X()
    def y()
    def yh()
    def processed() # Plots the tile
    
    





#%% Functions

def clahe(X):
    clahe   = cv2.createCLAHE(params.cliplimit, (params.gridsize, params.gridsize))
    X_clahe = []
    for channel in cv2.split(X):
        X_clahe.append(clahe.apply(channel))
    X_clahe = cv2.merge(X_clahe)
    return X_clahe

def plot(X):
    import matplotlib.pyplot as plt
    plt.imshow(X)
    plt.show()
    
def make_blocks(X, block_shape=(500, 500)):
    blocks = []
    for row in range(0, X.shape[0], block_shape[0]):
        for col in range(0, X.shape[1], block_shape[1]):
            blocks.append(X[row:row + block_shape[0], col:col + block_shape[1]])
    return blocks

#%% Data

Xtile = utils.searchFiles(paths.X, "0880_6260.*tif$")[0]
ytile = utils.searchFiles(paths.y, "0880_6260.*tif$")[0]

Xds = rasterio.open(Xtile)
Xds = reshape_as_image(Xds.read())
yds = rasterio.open(ytile).read(1)

#%% Pre-processing

X = cv2.normalize(Xds, None, 0, 255, cv2.NORM_MINMAX)
X = clahe(X)
X = cv2.cvtColor(X, cv2.COLOR_RGB2LAB)
X = make_blocks(X)
y = make_blocks(yds)

_ = [np.any(np.nonzero(i)) for i in y]
X = [i for (i, j) in zip(X, _) if j]
X = np.stack(X)
y = [i for (i, j) in zip(y, _) if j]
y = [np.where(i != 1, 0, i) for i in y]
y = np.stack(y)



#%% Augmentation
# https://towardsdatascience.com/image-augmentation-for-deep-learning-using-keras-and-histogram-equalization-9329f6ae5085
# https://keras.io/preprocessing/image/

datagen = imgaug.ImageDataGenerator(rotation_range=90)

# fit parameters from data
datagen.fit(x_train)





def unet(input_shape):
    inputs = Input(input_shape)
    conv1  = Conv2D(filters=64, kernel_size=3, activation="relu", padding="same", kernel_initializer="he_normal")(inputs)
    conv1  = Conv2D(filters=64, kernel_size=3, activation="relu", padding="same", kernel_initializer="he_normal")(conv1)
    pool1  = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2  = Conv2D(filters=128, kernel_size=3, activation="relu", padding="same", kernel_initializer="he_normal")(pool1)
    conv2  = Conv2D(filters=128, kernel_size=3, activation="relu", padding="same", kernel_initializer="he_normal")(conv2)
    pool2  = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3  = Conv2D(filters=256, kernel_size=3, activation="relu", padding="same", kernel_initializer="he_normal")(pool2)
    conv3  = Conv2D(filters=256, kernel_size=3, activation="relu", padding="same", kernel_initializer="he_normal")(conv3)
    pool3  = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4  = Conv2D(filters=512, kernel_size=3, activation="relu", padding="same", kernel_initializer="he_normal")(pool3)
    conv4  = Conv2D(filters=512, kernel_size=3, activation="relu", padding="same", kernel_initializer="he_normal")(conv4)
    drop4  = Dropout(0.5)(conv4)
    pool4  = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5  = Conv2D(filters=1024, kernel_size=3, activation="relu", padding="same", kernel_initializer="he_normal")(pool4)
    conv5  = Conv2D(filters=1024, kernel_size=3, activation="relu", padding="same", kernel_initializer="he_normal")(conv5)
    drop5  = Dropout(0.5)(conv5)
    up6    = UpSampling2D(size=(2, 2))(drop5)
    up6    = Conv2D(filters=512, kernel_size=2, activation="relu", padding="same", kernel_initializer="he_normal")(up6)
    merge6 = Concatenate(axis=3)([drop4, up6])
    conv6  = Conv2D(filters=512, kernel_size=3, activation="relu", padding="same", kernel_initializer="he_normal")(merge6)
    conv6  = Conv2D(filters=512, kernel_size=3, activation="relu", padding="same", kernel_initializer="he_normal")(conv6)
    up7    = UpSampling2D(size=(2, 2))(conv6)
    up7    = Conv2D(filters=256, kernel_size=2, activation="relu", padding="same", kernel_initializer="he_normal")(up7)
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7  = Conv2D(filters=256, kernel_size=3, activation="relu", padding="same", kernel_initializer="he_normal")(merge7)
    conv7  = Conv2D(filters=256, kernel_size=3, activation="relu", padding="same", kernel_initializer="he_normal")(conv7)
    up8    = UpSampling2D(size=(2, 2))(conv7)
    up8    = Conv2D(filters=128, kernel_size=2, activation="relu", padding="same", kernel_initializer="he_normal")(up8)
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8  = Conv2D(filters=128, kernel_size=3, activation="relu", padding="same", kernel_initializer="he_normal")(merge8)
    conv8  = Conv2D(filters=128, kernel_size=3, activation="relu", padding="same", kernel_initializer="he_normal")(conv8)
    up9    = UpSampling2D(size=(2, 2))(conv8)
    up9    = Conv2D(filters=64, kernel_size=2, activation="relu", padding="same", kernel_initializer="he_normal")(up9)
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9  = Conv2D(filters=64, kernel_size=3, activation="relu", padding="same", kernel_initializer="he_normal")(merge9)
    conv9  = Conv2D(filters=64, kernel_size=3, activation="relu", padding="same", kernel_initializer="he_normal")(conv9)
    conv9  = Conv2D(filters=2, kernel_size=3, activation="relu", padding="same", kernel_initializer="he_normal")(conv9)
    conv10 = Conv2D(filters=1, kernel_size=1, activation="sigmoid")(conv9)
    model  = Model(inputs=inputs, outputs=conv10)
    return model

#%% Augmentation
# https://towardsdatascience.com/image-augmentation-for-deep-learning-using-keras-and-histogram-equalization-9329f6ae5085
# https://keras.io/preprocessing/image/

datagen = imgaug.ImageDataGenerator(rotation_range=90)

# fit parameters from data
datagen.fit(x_train)





