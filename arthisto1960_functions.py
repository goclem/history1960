#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Utilities for the Arthisto1960 project
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.01.24
'''

#%% MODULES

import argparse
import easydict
import gdal
import numpy
import ogr
import os
import osr
import pandas
import rasterio
import rasterio.plot
import re
import shutil
import matplotlib.pyplot

#%% PATHS

def set_paths():
    home  = os.environ['HOME']
    base  = os.path.join(home, 'Dropbox', 'research', 'arthisto')
    code  = os.path.join(base, 'code_1960')
    data  = os.path.join(base, 'data_1960')
    paths = easydict.EasyDict({
        'home': home,
        'base': base,
        'code': code,
        'data': data,
        'Xraw': os.path.join(home, 'Dropbox', 'data', 'ign_scan50'),
        'yraw': os.path.join(base, 'shared_ras', 'training1960'),
        'mask': os.path.join(data, 'mask'),
        'X'   : os.path.join(data, 'X'),
        'y'   : os.path.join(data, 'y'),
        'yh'  : os.path.join(data, 'yh'),
        'tmp' : os.path.join(home, 'tmp'),
        'dsk' : os.path.join(home, 'Desktop')
    })
    return paths

#%% SPATIAL

def rasterise(srcVecPath:str, srcRstPath:str, outRstPath:str, driver:str='GTiff', burnField:str=None, burnValue:int=1, noDataValue:int=0, dataType:int=gdal.GDT_Byte):
    '''
    Description:
        Converts a vector file to a raster file
    
    Parameters:
        srcVecPath  (str): Path to source vector file 
        srcRstPath  (str): Path to source raster file
        outRstPath  (str): Path to output raster file
        driver      (str): GDAL driver
        burnField   (str): Field to burn (overrides burnValue)
        burnValue   (int): Fixed value to burn
        noDataValue (int): Missing value in raster
        dataType    (int): GDAL data type
        
    Returns:
        Rasterised vector file in raster format
    '''
    # import gdal, ogr
    srcRst = gdal.Open(srcRstPath, gdal.GA_ReadOnly)
    srcVec = ogr.Open(srcVecPath)
    srcLay = srcVec.GetLayer()
    rstDrv = gdal.GetDriverByName('GTiff')
    outRst = rstDrv.Create(outRstPath, srcRst.RasterXSize, srcRst.RasterYSize, 1, dataType)
    outRst.SetGeoTransform(srcRst.GetGeoTransform())
    outRst.SetProjection(srcRst.GetProjection())
    outBnd = outRst.GetRasterBand(1)
    outBnd.Fill(noDataValue)
    outBnd.SetNoDataValue(noDataValue)
    if burnField is not None:
        gdal.RasterizeLayer(outRst, [1], srcLay, options = ['ATTRIBUTE=' + burnField])
    else:
        gdal.RasterizeLayer(outRst, [1], srcLay, burn_values = [burnValue])
    outRst = None
    
# Raster to vector
def vectorise(srcRstPath, outVecPath, driver = 'ESRI Shapefile', fieldIndex = 1, connectedness = 4, dataType = ogr.OFTIntegerList):
    # import gdal, ogr, osr
    srcRst = gdal.Open(srcRstPath)
    srcBnd = srcRst.GetRasterBand(1)
    driver = ogr.GetDriverByName(driver)
    if os.path.exists(outVecPath):
        driver.DeleteDataSource(outVecPath)
    outVec = driver.CreateDataSource(outVecPath)
    srs    = osr.SpatialReference(srcRst.GetProjection())
    outLay = outVec.CreateLayer(outVecPath, srs)
    # outFld = ogr.FieldDefn('FID', dataType) 
    # outLay.CreateField(outFld)
    gdal.Polygonize(srcBnd, srcBnd, outLay, fieldIndex, ['8CONNECTED=' + str(connectedness)], callback = None)
    del srcRst, srcBnd, outVec, outLay;

#%% PLOTS

def plot(array, band=None, cmap='viridis', figsize=(10, 10)):
    '''Plots a single-band or multi-band raster.'''
    # import matplotlib.pyplot, rasterio, rasterio.plot
    if band is not None: array = array[..., band]
    if band is None and len(array.shape) > 2: array = rasterio.plot.reshape_as_raster(array)
    fig, ax = matplotlib.pyplot.subplots(1, figsize=figsize)
    ax.axis('off')
    rasterio.plot.show(array, cmap=cmap, ax=ax)

def plot_rgb(path, r=1, g=2, b=3, figsize=(15, 5), dpi=100):
    '''Plots the RGB channels of a multi-band raster separately.'''
    # import matplotlib.pyplot, rasterio, rasterio.plot
    raster = rasterio.open(path)
    name   = file_name(raster.name)
    fig, (axr, axg, axb) = matplotlib.pyplot.subplots(1, 3, figsize=figsize, dpi=dpi)
    axr.axis('off')
    axg.axis('off')
    axb.axis('off')
    rasterio.plot.show((raster, r), ax=axr, cmap='Reds',   title='%s (%s)' % (name, r))
    rasterio.plot.show((raster, g), ax=axg, cmap='Greens', title='%s (%s)' % (name, g))
    rasterio.plot.show((raster, b), ax=axb, cmap='Blues',  title='%s (%s)' % (name, b))

def plot_compare(path1, path2, band1=None, band2=None, cmap='viridis', figsize=(15, 7.5), dpi=100):
    '''Plots two single-band or multi-band rasters side-by-side.'''
    # import matplotlib.pyplot, rasterio, rasterio.plot
    raster1 = rasterio.open(path1)
    raster2 = rasterio.open(path2)
    fig, (ax1, ax2) = matplotlib.pyplot.subplots(1, 2, figsize=figsize, dpi=dpi)
    rasterio.plot.show((raster1, band1), ax=ax1, cmap=cmap, title=file_name(raster1.name))
    rasterio.plot.show((raster2, band2), ax=ax2, cmap=cmap, title=file_name(raster2.name))

def plot_hist(path, band=None, figsize=(10, 10)):
    '''Plots a single-band or multi-band raster.'''
    # import matplotlib.pyplot, rasterio, rasterio.plot
    raster  = rasterio.open(path)
    fig, ax = matplotlib.pyplot.subplots(1, figsize=figsize)
    if raster.count == 1: band = 1
    rasterio.plot.show_hist((raster, band), ax=ax, title=file_name(raster.name))

#%% FILES

def file_name(path, extension=False):
    '''Removes the directory and possibly the extension from a path.'''
    # import os
    file = os.path.basename(path)
    if extension is False:
        file = os.path.splitext(file)[0]
    return file

def search_files(directory:str, pattern:str='.', path:bool=True, extension:bool=True):
    '''Returns a list of files matching a regex in the directory tree.'''
    # import re, os
    files = []
    for root, _, file_names in os.walk(directory):
        for file_name in file_names:
            if path is True:
                files.append(os.path.join(root, file_name))
            else:
                files.append(file_name)
    files = list(filter(re.compile(pattern).search, files))
    if path is False:
        files = [os.path.basename(file) for file in files]
    if extension is False:
        files = [os.path.splitext(file)[0] for file in files]
    return files

def reset_folder(path, remove=False):
    '''Creates a folder if it doesn't exist, removes it eventually.'''
    # import os, shutil
    if remove is True:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
    if remove is False:
        if not os.path.exists(path):
            os.mkdir(path)

def trim_strings(strings, side='suffix', cutpoint='_'):
    '''Trims the prefix or the suffix of a list of strings.'''
    if side is 'suffix':
        trimmed = [string[:string.rfind(cutpoint)] for string in strings]
    if side is 'prefix':
        trimmed = [string[string.find(cutpoint) + len(cutpoint):] for string in strings]
    return trimmed

def get_tids(path, pattern='.tif$', path2=None, pattern2='.tif$', operator='difference', prefix=None, suffix=None, prefix2=None, suffix2=None):
    '''Returns a list of tile identifiers based on one of two folders.'''
    tiles = search_files(path, pattern, path=False, extension=False)
    if prefix is not None:
        tiles = trim_strings(tiles, 'prefix', prefix)
    if suffix is not None:
        tiles = trim_strings(tiles, 'suffix', suffix)
    if path2 is not None:
        filt = search_files(path2, pattern2, path=False, extension=False)
        if prefix2 is not None:
            filt = trim_strings(filt, 'prefix', prefix2)
        if suffix2 is not None:
            filt = trim_strings(filt, 'suffix', suffix2)
        if operator is 'difference': 
            tiles = list(set(tiles) - set(filt))
        if operator is 'intersection':
            tiles = list(set(tiles) & set(filt))
    tiles.sort()
    print(str(len(tiles)) + ' tiles')
    return tiles

#%% DESCRIPTIVE STATISTICS
def table(array):
    '''Returns a frequency table of a numpy array.'''
    # import pandas
    print(pandas.value_counts(array.flatten()))

#%% CLASSES

class tile_path(object):
    
    def __init__(self, path, log=0):
        '''Sets attributes.'''
        self.path = path
        self.log  = log
    
    def exists(self):
        '''Indicates whether the raster attribute exists.'''
        return os.path.exists(self.path)
    
    def check(self):
        if not os.path.isfile(self.path):
            raise IOError('File does not exist')
    
    def properties(self):
        '''Returns a dictionary with the properties of the raster attribute.'''
        self.check()
        return easydict.EasyDict(rasterio.open(self.path).meta)
    
    def values(self, band=None):
        '''Returns the raster attribute as a numpy array .'''
        self.check()
        source = rasterio.open(self.path)
        values = source.read(band)
        values = rasterio.plot.reshape_as_image(values)
        if source.count == 1: values = values[..., 0]
        return values
    
    def plot(self, band=None, cmap='viridis', figsize=(10, 10)):
        '''Plots the raster attribute.'''
        self.check()
        source = rasterio.open(self.path)
        if source.count == 1: band = 1
        fig, ax = matplotlib.pyplot.subplots(1, figsize=figsize)
        ax.axis('off')
        rasterio.plot.show((source, band), cmap=cmap, ax=ax, title=file_name(source.name))

    def histogram(self, band=None, figsize=(10, 10)):
        '''Returns an histogram of the raster attribute.'''
        self.check()
        source = rasterio.open(self.path)
        if source.count == 1: band = 1
        fig, ax = matplotlib.pyplot.subplots(1, figsize=figsize)
        ax.axis('off')
        rasterio.plot.show_hist((source, band), ax=ax, title=file_name(source.name))
            
    def table(self, band=None):
        '''Returns a frequency table of the raster attribute.'''
        self.check()
        values = rasterio.open(self.path)
        values = values.read(band)
        table  = pandas.value_counts(values.flatten())
        return table

class tile(object):
    
    def __init__(self, tid):
        '''Sets attributes as object of class tile_path.'''
        paths     = set_paths()
        self.tid  = tid
        self.Xraw = tile_path(os.path.join(paths.Xraw, 'sc50_{}.tif'.format(tid)))
        self.mask = tile_path(os.path.join(paths.mask, 'mask_{}.tif'.format(tid)))
        self.X    = tile_path(os.path.join(paths.X,    'X_{}.tif'.format(tid)))
        self.y    = tile_path(os.path.join(paths.y,    'y_{}.tif'.format(tid)))
        self.yh   = tile_path(os.path.join(paths.yh,   'yh_{}.tif'.format(tid)))
    
    def paths(self):
        '''Returns a dictionary with the path of every attribute.'''
        items = dict(vars(self))
        items.pop('tid', None)
        out   = dict(zip(items.keys(), [value.path for value in items.values()]))
        return easydict.EasyDict(out)
    
    def logs(self):
        '''Returns a dictionary with the log of every attribute.'''
        items = dict(vars(self))
        items.pop('tid', None)
        out   = dict(zip(items.keys(), [value.log for value in items.values()]))
        return easydict.EasyDict(out)
    
    def exist(self):
        '''Returns a dictionary indicating wheter attributes exist.'''
        items = dict(vars(self))
        items.pop('tid', None)
        out   = dict(zip(items.keys(), [os.path.exists(value.path) for value in items.values()]))
        return easydict.EasyDict(out)