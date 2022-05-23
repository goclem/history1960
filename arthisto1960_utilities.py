#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Utilities for the Arthisto1960 project
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.05.23
'''

#%% MODULES

import geopandas
import numpy as np
import rasterio
import os
import re
import shutil

from itertools import compress
from matplotlib import pyplot
from rasterio import features

#%% FILES UTILITIES

def search_data(pattern:str='.*', directory:str='../data_1960') -> list:
    '''Sorted list of files in a directory matching a regular expression'''
    files = list()
    for root, _, file_names in os.walk(directory):
        for file_name in file_names:
            files.append(os.path.join(root, file_name))
    files = list(filter(re.compile(pattern).search, files))
    files.sort()
    return files

def identifiers(files:list, regex:bool=False, extension:str='tif') -> list:
    '''Extracts file identifiers'''
    identifiers = [os.path.splitext(os.path.basename(file))[0] for file in files]
    identifiers = [identifier[identifier.find('_') + 1:] for identifier in identifiers]
    identifiers.sort()
    if regex:
        identifiers = '({identifiers})\\.{extension}$'.format(identifiers='|'.join(identifiers), extension=extension)
    return identifiers

def filter_identifiers(files:list, filter:list) -> list:
    '''Filters by identifiers'''
    subset = np.isin(identifiers(files), identifiers(filter), invert=True)
    subset = list(compress(files, subset))
    return subset

def initialise_directory(directory:str, remove:bool=False):
    '''Initialises a directory'''
    if not os.path.exists(directory):
        os.mkdir(directory)
    if os.path.exists(directory) and remove is True:
        shutil.rmtree(directory)
        os.mkdir(directory)

#%% RASTER UTILITIES

def read_raster(source:str, band:int=None, window=None, dtype:str=None) -> np.ndarray:
    '''Reads a raster as a numpy array'''
    raster = rasterio.open(source)
    if band is not None:
        image = raster.read(band, window=window)
        image = np.expand_dims(image, 0)
    else: 
        image = raster.read(window=window)
    image = image.transpose([1, 2, 0]).astype(dtype)
    return image

def write_raster(array:np.ndarray, profile, destination:str, nodata:int=None, dtype:str='uint8') -> None:
    '''Writes a numpy array as a raster'''
    if array.ndim == 2:
        array = np.expand_dims(array, 2)
    array = array.transpose([2, 0, 1]).astype(dtype)
    bands, height, width = array.shape
    if isinstance(profile, str):
        profile = rasterio.open(profile).profile
    profile.update(driver='GTiff', dtype=dtype, count=bands, nodata=nodata)
    with rasterio.open(fp=destination, mode='w', **profile) as raster:
        raster.write(array)
        raster.close()

def rasterise(source, profile, attribute:str=None, dtype:str='uint8') -> np.ndarray:
    '''Tranforms vector data into raster'''
    if isinstance(source, str): 
        source = geopandas.read_file(source)
    if isinstance(profile, str): 
        profile = rasterio.open(profile).profile
    geometries = source['geometry']
    if attribute is not None:
        geometries = zip(geometries, source[attribute])
    image = features.rasterize(geometries, out_shape=(profile['height'], profile['width']), transform=profile['transform'])
    image = image.astype(dtype)
    return image

#%% DISPLAY UTILITIES
    
def display(image:np.ndarray, title:str='', cmap:str='gray') -> None:
    '''Displays an image'''
    fig, ax = pyplot.subplots(1, figsize=(10, 10))
    ax.imshow(image, cmap=cmap)
    ax.set_title(title, fontsize=20)
    ax.set_axis_off()
    pyplot.tight_layout()
    pyplot.show()

def compare(images:list, titles:list=['Image'], cmaps:list=['gray']) -> None:
    '''Displays multiple images'''
    nimage = len(images)
    if len(titles) == 1:
        titles = titles * nimage
    if len(cmaps) == 1:
        cmaps = cmaps * nimage
    fig, axs = pyplot.subplots(nrows=1, ncols=nimage, figsize=(10, 10 * nimage))
    for ax, image, title, cmap in zip(axs.ravel(), images, titles, cmaps):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=15)
        ax.set_axis_off()
    pyplot.tight_layout()
    pyplot.show()