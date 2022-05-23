#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Utilities for the Arthisto1960 project
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.03.07
'''

#%% MODULES

import numpy as np
import rasterio
import os
import re
import shutil

from itertools import compress
from matplotlib import pyplot
from osgeo import gdal, ogr, osr

#%% FILES

# Lists files in a directory that match a regular expression
def search_files(directory:str, pattern:str='.') -> list:
    files = list()
    for root, _, file_names in os.walk(directory):
        for file_name in file_names:
            files.append(os.path.join(root, file_name))
    files = list(filter(re.compile(pattern).search, files))
    files.sort()
    return files

# Extracts file identifiers
def identifiers(files:list, regex:bool=False, extension:str='tif') -> list:
    identifiers = [os.path.splitext(os.path.basename(file))[0] for file in files]
    identifiers = [identifier[identifier.find('_') + 1:] for identifier in identifiers]
    identifiers.sort()
    if regex:
        identifiers = '({identifiers})\\.{extension}$'.format(identifiers='|'.join(identifiers), extension=extension)
    return identifiers

# Filters by identifiers
def filter_identifiers(files:list, filter:list) -> list:    
    subset = np.isin(identifiers(files), identifiers(filter), invert=True)
    subset = list(compress(files, subset))
    return subset

# Initialises a directory
def initialise_directory(directory:str, remove:bool=False):
    if not os.path.exists(directory):
        os.mkdir(directory)
    if os.path.exists(directory) and remove is True:
        shutil.rmtree(directory)
        os.mkdir(directory)

#%% DISPLAY
    
# Displays an image
def display(image:np.ndarray, title:str='', cmap:str='gray') -> None:
    fig, ax = pyplot.subplots(1, figsize=(10, 10))
    ax.imshow(image, cmap=cmap)
    ax.set_title(title, fontsize=20)
    ax.set_axis_off()
    pyplot.tight_layout()
    pyplot.show()

# Displays multiple images
def compare(images:list, titles:list=['Image'], cmaps:list=['gray']) -> None:
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

#%% RASTERS

# Reads a raster as an array
def read_raster(source:str, dtype:type=np.uint8) -> np.ndarray:
    raster = rasterio.open(source)
    raster = raster.read()
    image  = raster.transpose([1, 2, 0]).astype(dtype)
    return image

# Writes an array as a raster
def write_raster(array:np.ndarray, source:str, destination:str, nodata:int=0, dtype:str='uint8') -> None:
    raster = array.transpose([2, 0, 1]).astype(dtype)
    bands, height, width = raster.shape
    profile = rasterio.open(source).profile
    profile.update(driver='GTiff', dtype=dtype, count=bands, nodata=nodata)
    with rasterio.open(fp=destination, mode='w', **profile) as dest:
        dest.write(raster)
        dest.close()