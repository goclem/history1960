#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@description: Utilities for arthisto1950
@author: Clement Gorin
@contact: gorin@gate.cnrs.fr
@date: April 2020
"""

#%% Dependencies

import argparse
import gdal
import numpy
import os
import pandas
import re
import shutil

#%% Paths

def makePaths():
    base  = os.path.join(os.environ["HOME"], "Dropbox", "research", "arthisto")
    code  = os.environ["PWD"]
    data  = os.path.join(base, "data_1950")
    paths = argparse.Namespace(
        code = code,
        data = data,
        temp = os.path.join(os.environ["HOME"], "temp_1950"),
        desk = os.path.join(os.environ["HOME"], "Desktop")
        raw  = os.path.join(data, "scan50"),
        # X    = os.path.join(data, "X"),
        # y    = os.path.join(data, "y"),
        fh   = os.path.join(data, "fh"),
        yh   = os.path.join(data, "yh"),
        X    = os.path.join(base, "data_1850", "scem"),
        y    = os.path.join(base, "data_1850", "datasets", "y")
    )
    return paths

#%% Files

# File name from path
def fileName(path, extension = False):
    file = os.path.basename(path)
    if extension is False:
        file = os.path.splitext(file)[0]
    return file

def plot(r, figsize = (10, 10), dpi = 100):
    # import matplotlib.pyplot as plt
    fig = plt.figure(dpi = dpi, figsize = figsize)
    axs = fig.add_subplot()
    axs.imshow(r, cmap = "gray", vmin = 0, vmax = 255)
    axs.axis("off")
    plt.show()

# List files
def searchFiles(directory, pattern = ".", path = True, extension = True):
    # import re, os
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if path is True:
                files.append(os.path.join(root, filename))
            else:
                files.append(filename)
    files = list(filter(re.compile(pattern).search, files))
    if extension is False:
        files = [os.path.splitext(os.path.basename(file))[0] for file in files]
    return files

# List folders
def searchFolds(directory, pattern = ".", path = True):
    # import re, os
    folds = list(filter(re.compile(pattern).search, os.listdir(directory)))
    if path == True:
        folds = [os.path.join(path, fold) for fold in folds]
    return folds

# Rests a folder without confirmation
def resetFold(path, remove = False):
    if remove is True:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
    if remove is False:
        if not os.path.exists(path):
            os.mkdir(path)

# Trims a list of strings
def trimStrings(strings, what = "suffix", cut = "_"):
    if what is "suffix":
        trimmed = [string[:string.rfind(cut)] for string in strings]
    if what is "prefix":
        trimmed = [string[string.find(cut) + 1:] for string in strings]
    return trimmed

def getTiles(folder1, pattern1=".tif$", folder2=None, pattern2=".tif$", operator="difference", prefix1=None, suffix1="_", prefix2=None, suffix2="_"):
    tiles = searchFiles(folder1, pattern1, fullPath=False, extension=False)
    if prefix1 is not None:
        tiles = trimStrings(tiles, "prefix", prefix1)
    if suffix1 is not None:
        tiles = trimStrings(tiles, "suffix", suffix1)
    if folder2 is not None:
        filt = searchFiles(folder2, pattern2, fullPath=False, extension=False)
        if prefix2 is not None:
            filt = trimStrings(filt, "prefix", prefix2)
        if suffix2 is not None:
            filt = trimStrings(filt, "suffix", suffix2)
        if operator is "difference": 
            tiles = list(set(tiles) - set(filt))
        if operator is "intersection":
            tiles = list(set(tiles) & set(filt))
    tiles.sort()
    print(str(len(tiles)) + " tiles")
    return tiles

#%% Summary statistics

def tab(x):
    print(pandas.value_counts(x.flatten()))