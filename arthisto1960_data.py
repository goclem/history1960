import histo1950_functions as foo 
import ogr
import gdal

paths = foo.set_paths()
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