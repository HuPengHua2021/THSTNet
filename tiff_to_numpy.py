import numpy as np
from osgeo import gdal

def read_tiff_to_numpy(filename):
    dataset = gdal.Open(filename)
    shape = [dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount]
    outputfile = np.zeros((shape[2], shape[0], shape[1]))
    for i in range(shape[2]):
        outputfile[i, :, :] = dataset.GetRasterBand(i + 1).ReadAsArray()
    outpath = filename.split('.')[0]
    outname = outpath + '.npy'
    np.save(outname, outputfile)

if __name__=='__main__':
    read_tiff_to_numpy(r'D:\SwinT_FYMOLA\landsat\lansat_fine.tif')
    read_tiff_to_numpy(r'D:\SwinT_FYMOLA\MODIS\modis_fine.tif')
