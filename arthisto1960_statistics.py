#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Computes statistics statistics for the Arthisto 1960 project
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
from pandas import DataFrame
from skimage import segmentation

# Samples
cities   = dict(paris='0625_6870|0650_6870', marseille='0875_6245|0875_6270', lyon='0825_6520|0825_6545', toulouse='0550_6295|0575_6295')
training = identifiers(search_data(paths['labels']), regex = True)

#%% FUNCTIONS

def compute_sets(label_test:np.ndarray, label_pred:np.ndarray) -> np.ndarray:
    '''Computes prediction sets'''
    label_test = label_test.astype(bool)
    label_pred = label_pred.astype(bool)
    set_tp = np.logical_and(label_test, label_pred)
    set_tn = np.logical_and(np.invert(label_test), np.invert(label_pred))
    set_fp = np.logical_and(np.invert(label_test), label_pred)
    set_fn = np.logical_and(label_test, np.invert(label_pred))
    sets   = np.array([set_tp, set_tn, set_fp, set_fn])
    return sets

def mask_borders(sets:np.ndarray, label_test:np.ndarray) -> np.ndarray:
    '''Computes subset without borders'''
    subset = np.invert(segmentation.find_boundaries(label_test))
    subset = np.tile(subset, (sets.shape[0], 1, 1, 1))
    masked = np.where(subset, sets, False)
    return masked 

def compute_statistics(sets:np.ndarray):
    '''Computes prediction statistics'''
    tp, tn, fp, fn = np.sum(sets, axis=(1, 2, 3))
    with np.errstate(divide='ignore', invalid='ignore'): # Returns Inf when dividing by 0
        recall    = np.divide(tp, (tp + fn)) # Among the building pixels, {recall}% are classified as building
        precision = np.divide(tp, (tp + fp)) # Among the pixels classified as buildings, {precision}% are in fact buildings
        accuracy  = np.divide((tp + tn), (tp + tn + fp + fn))
    statistics = dict(tp=tp, tn=tn, fp=fp, fn=fn, recall=recall, precision=precision, accuracy=accuracy)
    return statistics

def display_statistics(image:np.ndarray, sets:np.ndarray, colour=(255, 255, 0)) -> None:
    '''Displays prediction masks'''
    counts = np.sum(sets, axis=(1, 2, 3))
    titles = ['True positive ({:d})', 'True negative ({:d})', 'False positive ({:d})', 'False negative ({:d})']
    titles = list(map(lambda title, count: title.format(count), titles, counts))
    images = [np.where(np.tile(mask, (1, 1, 3)), colour, image) for mask in sets]
    fig, axs = pyplot.subplots(2, 2, figsize=(10, 10))
    for image, title, ax in zip(images, titles, axs.ravel()):
        ax.imshow(image)
        ax.set_title(title, fontsize=20)
        ax.axis('off')
    pyplot.tight_layout(pad=2.0)
    pyplot.show()

#%% FORMATS DATA

# Loads model and test data
labels_test = np.load(path.join(paths['statistics'], 'labels_test.npy'))
images_test = np.load(path.join(paths['statistics'], 'images_test.npy'))
model       = models.load_model(path.join(paths['models'], 'unet64_220609.h5'))

# Predicts test data
probas_pred = layers.Rescaling(1./255)(images_test)
probas_pred = model.predict(probas_pred, verbose=1)
labels_pred = probas_pred >= 0.5

# Subsets data
subset = np.logical_and(
    np.sum(labels_test, axis=(1, 2, 3)) > 0,
    np.sum(labels_pred, axis=(1, 2, 3)) > 0
)
images_test = images_test[subset]
labels_test = labels_test[subset]
probas_pred = probas_pred[subset]
labels_pred = labels_pred[subset]
del subset

#%% COMPUTES PREDICTION STATISTICS

# Compute sets and removes border
sets = np.array(list(map(compute_sets, labels_test, labels_pred)))
sets = np.array(list(map(mask_borders, sets, labels_test)))

# Aggregated statistics
stats = np.sum(sets, axis=0)
stats = compute_statistics(stats)

# Statistics per tile
stats = list(map(compute_statistics, sets))
stats = DataFrame.from_dict(stats)

#%% DISPLAYS PREDICTION STATISTICS

# Displays statistics distribution
fig = pyplot.figure(figsize=(10,5))
stats.hist(['precision', 'recall'], bins=100, ax=fig.gca())
del fig

# Displays image statistics
subset = stats.sort_values(by='fp', ascending=False).index[:10]
for image, set in zip(images_test[subset], sets[subset]):
    display_statistics(image, set)
del subset

# %%
