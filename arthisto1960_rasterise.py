#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Prepares data for the Arthisto1860 project
@author: Clement Gorin
@contact: gorinclem@gmail.com
@date: March 2021
'''

#%% MODULES

import argparse
import concurrent.futures
import histo1950_functions as foo
import gdal
import ogr
import numpy as np
import pandas as pd
import os
import re
import shutil

#%% DATA

paths = argparse.Namespace(
    src = '/Users/clementgorin/Dropbox/research/arthisto/shared_ras/training1960',
    out = '/Users/clementgorin/Desktop/vectors'
    )

foo.reset_folder(paths.out)
srcRsts = foo.search_files(paths.src, "\\d{4}_\\d{4}\\.tif$")
srcVecs = foo.search_files(paths.src, "\\d{4}_\\d{4}\\.gpkg$")
outRsts = [os.path.join(paths.out, os.path.basename(file)) for file in srcRsts]
outVecs = [file.replace(".tif", ".gpkg") for file in outRsts]

#%% COMPUTATIONS

for srcRst, srcVec, outRst, outVec in zip(srcRsts, srcVecs, outRsts, outVecs):
    foo.rasterise(srcRst, srcVec, outRst)
    foo.vectorise(outRst, outVec)

srcRstPath=outRst
outVecPath=outVec





# Loads the layer in memory
srcDs  = ogr.GetDriverByName('GPKG').Open(paths.src, 0)
memDs  = ogr.GetDriverByName('Memory').CreateDataSource('memDs')
memDs.CopyLayer(srcDs.GetLayer(), 'memLay', ['OVERWRITE=YES'])
memLay = memDs.GetLayer('memLay')

# Maps clc classes to cls
memLay.CreateField(ogr.FieldDefn('cls_id', ogr.OFTReal))
for feature in memLay:
    value = clc2cls.loc[clc2cls.clc_id == feature.GetField("code_18")].iloc[0].cls_id
    feature.SetField('cls_id', value)
    memLay.SetFeature(feature)

# Single tile
i      = 10
srcRst = srcRsts[i]
outRst = outRsts[i]
rasterise(srcRst, outRst)
shutil.copy(paths.ohs + '/ohs_0080_6860.qml', outRst.replace('.tif', ".qml"))

# Loop version
for srcRst, outRst in zip(srcRsts, outRsts):
    rasterise(srcRst, outRst)
    shutil.copy(paths.ohs + '/ohs_0080_6860.qml', outRst.replace('.tif', ".qml"))
    
# Parallel version
with concurrent.futures.ProcessPoolExecutor(max_workers = 4) as executor:
    executor.map(rasterise, srcRsts, outRsts)