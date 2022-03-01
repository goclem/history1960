#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@description: UNet models for arthisto1950
@author: Clement Gorin
@contact: gorin@gate.cnrs.fr
@date: March 2021
"""

"""
Pre-trained models: https://github.com/qubvel/segmentation_models
Satellite UNet: https://github.com/karolzak/keras-unet/blob/master/keras_unet/models/satellite_unet.py
"""
#%% Utilities

import os
import argparse
import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int,  default=1,    help="Session seed")
parser.add_argument("--gpu",  type=bool, default=True, help="Uses GPU")
params = parser.parse_args()

# GPU backend
if params.gpu: 
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

#%% Model 1

def unet(input_shape=(256, 256, 1), pretrained_weights=None):
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
    model.compile(optimizer = Adam(lr = 1e-4), loss = "binary_crossentropy", metrics = ["accuracy"])
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)
    return model

#%% Model 2

def bn_conv_relu(input, filters, bachnorm_momentum, **conv2d_args):
    x = BatchNormalization(momentum=bachnorm_momentum)(input)
    x = Conv2D(filters, **conv2d_args)(x)
    return x

def bn_upconv_relu(input, filters, bachnorm_momentum, **conv2d_trans_args):
    x = BatchNormalization(momentum=bachnorm_momentum)(input)
    x = Conv2DTranspose(filters, **conv2d_trans_args)(x)
    return x

def satellite_unet(input_shape, num_classes=1, output_activation='sigmoid', num_layers=4):

    inputs = Input(input_shape)
    filters        = 64
    upconv_filters = 96
    kernel_size    = (3,3)
    activation     = "relu"
    strides        = (1,1)
    padding        = "same"
    kernel_initializer = 'he_normal'

    conv2d_args = {
        'kernel_size':kernel_size,
        'activation':activation, 
        'strides':strides,
        'padding':padding,
        'kernel_initializer':kernel_initializer
        }

    conv2d_trans_args = {
        'kernel_size':kernel_size,
        'activation':activation, 
        'strides':(2,2),
        'padding':padding,
        'output_padding':(1,1)
        }

    bachnorm_momentum = 0.01

    pool_size = (2,2)
    pool_strides = (2,2)
    pool_padding = 'valid'

    maxpool2d_args = {
        'pool_size':pool_size,
        'strides':pool_strides,
        'padding':pool_padding,
        }
    
    x = Conv2D(filters, **conv2d_args)(inputs)
    c1 = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)    
    x = bn_conv_relu(c1, filters, bachnorm_momentum, **conv2d_args)
    x = MaxPooling2D(**maxpool2d_args)(x)

    down_layers = []

    for l in range(num_layers):
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        down_layers.append(x)
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = MaxPooling2D(**maxpool2d_args)(x)

    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    x = bn_upconv_relu(x, filters, bachnorm_momentum, **conv2d_trans_args)

    for conv in reversed(down_layers):        
        x = concatenate([x, conv])  
        x = bn_conv_relu(x, upconv_filters, bachnorm_momentum, **conv2d_args)
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = bn_upconv_relu(x, filters, bachnorm_momentum, **conv2d_trans_args)

    x = concatenate([x, c1])
    x = bn_conv_relu(x, upconv_filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
           
    outputs = Conv2D(num_classes, kernel_size=(1,1), strides=(1,1), activation=output_activation, padding='valid') (x)       
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model


#%% Augmentation
# https://towardsdatascience.com/image-augmentation-for-deep-learning-using-keras-and-histogram-equalization-9329f6ae5085
# https://keras.io/preprocessing/image/

datagen = imgaug.ImageDataGenerator(rotation_range=90)

# fit parameters from data
datagen.fit(x_train)




#%%
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
