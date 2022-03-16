#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Workflow for the semantic segmentation example
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.03.15
'''

# Modules
from arthisto1960_utilities import *
from os import path

#%% CLEAN IGN RASTERS

paths = dict(
    ign    = '/Users/clementgorin/Dropbox/data/ign_scan50',
    images = '../data_1960/images'
)

srcfiles = search_files(paths['ign'], 'tif$')

for srcfile in srcfiles:
    print(path.basename(srcfile))
    outfile = path.basename(srcfile).replace('sc50', 'image')
    outfile = path.join(paths['images'], outfile)
    if not path.exists(outfile):
        os.system('gdal_translate -ot byte {srcfile} {outfile}'.format(srcfile=srcfile, outfile=outfile))





files = foo.search_files(paths.Yraw, '\\.gpkg$')
srcVecPath=files[3]
srcRstPath='/Users/clementgorin/Dropbox/research/arthisto/data_1950/training_1950/training/lille/lille_rasters/SC50_HISTO1950_0700_7070_L93.tif'
outRstPath=os.path.join(paths.desktop, 'try.tif')
outVecPath=os.path.join(paths.desktop, 'try.gpkg')
# Computes training rasters





def rasterise(srcRstPath, srcVecPath, outRstPath, burnValue=1, burnField=None, noData=0, dataType=gdal.GDT_Byte):
    # Drivers
    vecDrv = ogr.GetDriverByName('GPKG')
    rstDrv = gdal.GetDriverByName('GTiff')
    # Source data
    srcVec = vecDrv.Open(srcVecPath, 0)
    srcRst = gdal.Open(srcRstPath, 0)
    srcLay = srcVec.GetLayer()
    # Output raster
    outRst = rstDrv.Create(outRstPath, srcRst.RasterXSize, srcRst.RasterYSize, 1, dataType)
    outRst.SetGeoTransform(srcRst.GetGeoTransform())
    outRst.SetProjection(srcRst.GetProjection())
    # Output band
    outBnd = outRst.GetRasterBand(1)
    outBnd.Fill(noData)
    outBnd.SetNoDataValue(noData)        
    # Rasterise
    if burnField is not None:
        gdal.RasterizeLayer(outRst, [1], srcLay, options=["ATTRIBUTE=" + burnField])
    else:
        gdal.RasterizeLayer(outRst, [1], srcLay, burn_values=[burnValue])
    outRst = outBnd = None