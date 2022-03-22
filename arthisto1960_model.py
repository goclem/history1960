#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Initialises model for the Arthisto 1960 project
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.03.22
'''

#%% HEADER

# Modules
import numpy as np
from pandas import DataFrame
from os import path
from tensorflow.keras import callbacks, layers, models, utils

# Paths
paths = dict(
    models='../data_1960/models',
    statistics='../data_1960/statistics'
)

'''
Notes:
- Check the number of filters for transpose, we should maintian dimensionality
- Compare the model summary with the original U-net to make sure everything is ok
- May be better https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/
'''

#%% UNET MODEL 1

def convolutional_block(input, filters:int, dropout:float=0, kernel_size:dict=(3, 3), padding:str='same', initializer:str='he_normal', name:str=''):
    convolution   = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, kernel_initializer=initializer, use_bias=False, name=f'{name}_convolution1')(input)
    activation    = layers.Activation(activation='relu', name=f'{name}_activation1')(convolution)
    normalisation = layers.BatchNormalization(name=f'{name}_normalisation1')(activation)
    convolution   = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, kernel_initializer=initializer, use_bias=False, name=f'{name}_convolution2')(normalisation)
    activation    = layers.Activation(activation='relu', name=f'{name}_activation2')(convolution)
    normalisation = layers.BatchNormalization(name=f'{name}_normalisation2')(activation)
    dropout       = layers.SpatialDropout2D(rate=dropout, name=f'{name}_dropout')(normalisation)
    return dropout

def deconvolutional_block(input, skip, filters:int, kernel_size:dict=(3, 3), padding:str='same', strides:dict=(2, 2), dropout:float=0, name:str=''):
    transpose     = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=f'{name}_transpose')(input)
    concatenation = layers.concatenate(inputs=[transpose, skip], axis=3, name=f'{name}_concatenation')
    convblock     = convolutional_block(input=concatenation, filters=filters, dropout=dropout, name=name)
    return convblock

def init_unet1(n_classes:int, input_size:tuple, filters:int):
    # Input
    inputs = layers.Input(shape=input_size, name='input')
    # Contraction
    convblock1 = convolutional_block(input=inputs, filters=filters*1, dropout=0.1, name='convblock1')
    maxpool1   = layers.MaxPool2D(pool_size=(2, 2), name='convblock1_maxpool')(convblock1)
    convblock2 = convolutional_block(input=maxpool1, filters=filters*2, dropout=0.1, name='convblock2')
    maxpool2   = layers.MaxPool2D(pool_size=(2, 2), name='convblock2_maxpool')(convblock2)
    convblock3 = convolutional_block(input=maxpool2, filters=filters*4, dropout=0.2, name='convblock3')
    maxpool3   = layers.MaxPool2D(pool_size=(2, 2), name='convblock3_maxpool')(convblock3)
    convblock4 = convolutional_block(input=maxpool3, filters=filters*8, dropout=0.2, name='convblock4')
    maxpool4   = layers.MaxPool2D(pool_size=(2, 2), name='convblock4_maxpool')(convblock4)
    # Bottleneck
    convblock5 = convolutional_block(input=maxpool4, filters=filters*16, dropout=0.3, name='convblock5')
    # Extension
    deconvblock1 = deconvolutional_block(input=convblock5,   skip=convblock4, filters=filters*8, dropout=0.3, name='deconvblock1')
    deconvblock2 = deconvolutional_block(input=deconvblock1, skip=convblock3, filters=filters*4, dropout=0.2, name='deconvblock2')
    deconvblock3 = deconvolutional_block(input=deconvblock2, skip=convblock2, filters=filters*2, dropout=0.2, name='deconvblock3')
    deconvblock4 = deconvolutional_block(input=deconvblock3, skip=convblock1, filters=filters*1, dropout=0.1, name='deconvblock4')
    # Output
    output = layers.Conv2D(n_classes, kernel_size=(1, 1), activation='sigmoid', name='output')(deconvblock4)
    # Model
    model   = models.Model(inputs=inputs, outputs=output, name='Unet')
    return model

unet1 = init_unet1(n_classes=1, input_size=(256, 256, 3), filters=16)
unet1.compile(optimizer='adam', loss='binary_focal_crossentropy', metrics=['accuracy', 'Recall', 'Precision'])
del convolutional_block, deconvolutional_block, init_unet1

# Summary
# utils.plot_model(unet, to_file=path.join(paths['models'], 'unet_structure.pdf'), show_shapes=True)
# summary = DataFrame([dict(Name=layer.name, Type=layer.__class__.__name__, Shape=layer.output_shape, Params=layer.count_params()) for layer in unet.layers])
# summary.style.to_html(path.join(paths['models'], 'unet_structure.html'), index=False) 
# del summary

#%% UNET MODEL 2
'''
Note: Not working
'''

def convolutional_block(input, filters:int, dropout:float=0, kernel_size:dict=(3, 3), padding:str='same', initializer:str='he_normal', name:str=''):
    convolution   = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, kernel_initializer=initializer, use_bias=False, name=f'{name}_convolution1')(input)
    activation    = layers.Activation(activation='relu', name=f'{name}_activation1')(convolution)
    normalisation = layers.BatchNormalization(name=f'{name}_normalisation1')(activation)
    convolution   = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, kernel_initializer=initializer, use_bias=False, name=f'{name}_convolution2')(normalisation)
    activation    = layers.Activation(activation='relu', name=f'{name}_activation2')(convolution)
    normalisation = layers.BatchNormalization(name=f'{name}_normalisation2')(activation)
    return normalisation

def downsample_block(input, filters:int, dropout:float=0, kernel_size:dict=(3, 3), padding:str='same', initializer:str='he_normal', pool_size:dict=(2, 2), name:str=''):
    convolutions = convolutional_block(input, filters=filters, dropout=dropout, kernel_size=kernel_size, padding=padding, initializer=initializer, name=name)
    pooling      = layers.MaxPool2D(pool_size=pool_size, name=f'{name}_maxpool')(convolutions)
    dropout      = layers.SpatialDropout2D(rate=dropout, name=f'{name}_dropout')(pooling)
    return dropout

def upsample_block(input, skip, filters:int, kernel_size:dict=(3, 3), padding:str='same', strides:dict=(2, 2), dropout:float=0, name:str=''):
    transpose     = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=f'{name}_transpose')(input)
    concatenation = layers.concatenate(inputs=[transpose, skip], axis=3, name=f'{name}_concatenation')
    convblock     = convolutional_block(input=concatenation, filters=filters, dropout=dropout, name=name)
    return convblock

def init_unet2(n_classes:int, input_size:tuple, filters:int):
    # Input
    inputs = layers.Input(shape=input_size, name='input')
    # Down-sampling
    down_block1 = downsample_block(input=inputs,      filters=filters*1,  dropout=0.1, name='down_block1')
    down_block2 = downsample_block(input=down_block1, filters=filters*2,  dropout=0.1, name='down_block2')
    down_block3 = downsample_block(input=down_block2, filters=filters*4,  dropout=0.2, name='down_block3')
    down_block4 = downsample_block(input=down_block3, filters=filters*8,  dropout=0.2, name='down_block4')
    down_block5 = downsample_block(input=down_block4, filters=filters*16, dropout=0.3, name='down_block5')
    # Up-sampling
    up_block1   = upsample_block(input=down_block5, skip=down_block4, filters=filters*8, dropout=0.3, name='up_block1')
    up_block2   = upsample_block(input=up_block1,   skip=down_block3, filters=filters*4, dropout=0.2, name='up_block2')
    up_block3   = upsample_block(input=up_block2,   skip=down_block2, filters=filters*2, dropout=0.2, name='up_block3')
    up_block4   = upsample_block(input=up_block3,   skip=down_block1, filters=filters*1, dropout=0.1, name='up_block4')
    # Output
    output = layers.Conv2D(n_classes, kernel_size=(1, 1), activation='sigmoid', name='output')(up_block4)
    # Model
    model   = models.Model(inputs=inputs, outputs=output, name='Unet')
    return model

unet2 = init_unet2(n_classes=1, input_size=(256, 256, 3), filters=16)
unet2.compile(optimizer='adam', loss='binary_focal_crossentropy', metrics=['accuracy', 'Recall', 'Precision'])
del convolutional_block, downsample_block, upsample_block, init_unet2

# Summary
# utils.plot_model(unet, to_file=path.join(paths['models'], 'unet_structure.pdf'), show_shapes=True)
# summary = DataFrame([dict(Name=layer.name, Type=layer.__class__.__name__, Shape=layer.output_shape, Params=layer.count_params()) for layer in unet.layers])
# summary.style.to_html(path.join(paths['models'], 'unet_structure.html'), index=False) 
# del summary

#%% TESTING

# Data
images_test = np.load(path.join(paths['statistics'], 'images_test.npy'))
labels_test = np.load(path.join(paths['statistics'], 'labels_test.npy'))

# Training
training = unet2.fit(images_test, labels_test, epochs=2, verbose=1)

# %%
