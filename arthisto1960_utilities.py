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

from matplotlib import pyplot
from osgeo import gdal, ogr, osr

#%% FUNCTIONS

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
def identifiers(files:list, regex:bool=False) -> list:
    identifiers = [os.path.splitext(os.path.basename(file))[0] for file in files]
    identifiers = [identifier[identifier.find('_') + 1:] for identifier in identifiers]
    identifiers.sort()
    if regex:
        identifiers = '({})'.format('|'.join(identifiers))
    return identifiers

# Initialises a directory
def initialise_directory(directory:str, remove:bool=False):
    if not os.path.exists(directory):
        os.mkdir(directory)
    if os.path.exists(directory) and remove is True:
        shutil.rmtree(directory)
        os.mkdir(directory)

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

# Displays a random sample of images
def display_sample(images:np.ndarray, gridsize:int=3, seed:int=None) -> None:
    np.random.seed(seed)
    sample  = np.random.choice(range(images.shape[0]), gridsize * gridsize)
    images  = images[sample]
    fig, axs = pyplot.subplots(nrows=gridsize, ncols=gridsize, figsize=(10, 10))
    for image, ax in zip(images, axs.ravel()):
        ax.imshow(image)
        ax.axis('off')
    fig.suptitle('Sampled images', fontsize=20)
    pyplot.tight_layout()
    pyplot.show()

# Converts vector to raster
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
    
# Converts raster to vector
def vectorise(srcRstPath, outVecPath, driver='ESRI Shapefile', fieldIndex=1, connectedness=4, dataType=ogr.OFTIntegerList):
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
    gdal.Polygonize(srcBnd, srcBnd, outLay, fieldIndex, ['8CONNECTED=' + str(connectedness)], callback=None)
    del srcRst, srcBnd, outVec, outLay
