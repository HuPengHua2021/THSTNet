from osgeo import gdal
from sewar import rmse
import numpy as np

from math import sqrt
from tools import ssim_numpy
from sklearn.metrics import r2_score

def GetGeoInformation(tif_file):
    dataset = gdal.Open(tif_file)
    im_proj = dataset.GetProjection()  # 读取投影
    im_Geotrans = dataset.GetGeoTransform()  # 读取仿射变换
    del dataset
    return im_proj, im_Geotrans


def WriteTiff(newpath, im_data, im_Geotrans, im_proj, datatype):
    if len(im_data.shape) == 3:
        im_height, im_width, im_bands = im_data.shape
    else:
        (im_height, im_width), im_bands = im_data.shape, 1
    diver = gdal.GetDriverByName('GTiff')
    new_dataset = diver.Create(newpath, im_width, im_height, im_bands, datatype)
    new_dataset.SetGeoTransform(im_Geotrans)
    new_dataset.SetProjection(im_proj)

    if im_bands == 1:
        new_dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            new_dataset.GetRasterBand(i + 1).WriteArray(im_data[:, :, i])
    del new_dataset

def read_im(path):
    dataset = gdal.Open(path)
    shape = [dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount]
    image_data = np.zeros(shape)
    for j in range(shape[2]):
        image_data[:, :, j] = dataset.GetRasterBand(j + 1).ReadAsArray()

    return image_data


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    else:
        return 20 * np.log10(100 / np.sqrt(mse))


def pearson_correlation_coefficient(x, y):

    mean_x = np.mean(x)
    mean_y = np.mean(y)


    covariance = np.sum((x - mean_x) * (y - mean_y))


    std_dev_x = np.sqrt(np.sum((x - mean_x) ** 2))
    std_dev_y = np.sqrt(np.sum((y - mean_y) ** 2))


    correlation_coefficient = covariance / (std_dev_x * std_dev_y)

    return correlation_coefficient


def cal_seven():
    Landsat = read_im(r'D:\lansat_fine.tif')
    THSTNET = read_im(r'D:\THSTNET.tif')
    STTFN = read_im(r'D:\STTFN.tif')
    EDSTFN = read_im(r'D:\edcstfn.tif')
    GAN = read_im(r'D:\GAN.tif')
    Msnet = read_im(r'D:\MSnet.tif')
    starfm = read_im(r'D:\STARFM.tif')
    estarfm = read_im(r'D:\ESTARFM.tif')

    band1 = [8, 12, 15, 23, 28, 41]
    band2 = [i for i in range(6)]
    RMSE = np.zeros([6, 7])
    MAE = np.zeros([6, 7])
    SSIM = np.zeros([6, 7])
    PSNR = np.zeros([6, 7])
    R2 = np.zeros([6, 7])

    for i in range(6):
        Landsat_i = Landsat[:, :, band1[i]]
        THSTNET_i = THSTNET[:, :, band1[i]]
        STTFN_i = STTFN[:, :, band1[i]]
        EDSTFN_i = EDSTFN[:, :, band1[i]]
        GAN_i = GAN[:, :, band1[i]]
        Msnet_i = Msnet[:, :, band1[i]]
        starfm_i = starfm[:, :, band2[i]]
        estarfm_i = estarfm[:, :, band2[i]]

        predic = np.array([starfm_i, estarfm_i, STTFN_i, EDSTFN_i, GAN_i, Msnet_i, THSTNET_i])
        for j in range(7):
            label = Landsat_i
            image = predic[j]
            index = np.where(abs(image - label) > 10)
            image[index] = 0
            label[index] = 0

            PSNR[i, j] = psnr(image, label)
            SSIM[i, j] = ssim_numpy(label - 250, image - 250, val_range=50)

            y_test = np.reshape(label, [-1])
            y_predict = np.reshape(image, [-1])

            y_mean = np.mean(y_test)

            mse = np.sum((y_test - y_predict) ** 2) / len(y_test)
            rmse = sqrt(mse)
            mae = np.sum(np.absolute(y_test - y_predict)) / len(y_test)
            r = pearson_correlation_coefficient(y_test, y_predict)

            RMSE[i, j] = rmse
            MAE[i, j] = mae
            R2[i, j] = r ** 2

    np.savetxt(r'.\RMSE.txt', RMSE)
    np.savetxt(r'.\MAE.txt', MAE)
    np.savetxt(r'.\SSIM.txt', SSIM)
    np.savetxt(r'.\PSNR.txt', PSNR)
    np.savetxt(r'.\R2.txt', R2)


def cal_two():
    Landsat = read_im(r'D:\lansat_fine.tif')
    without_tt = read_im(r'D:\without_tt.tif')
    THSTNET = read_im(r'D:\THSTNET.tif')
    band = [8, 12, 15, 23, 28, 41]
    RMSE = np.zeros([6, 3])
    MAE = np.zeros([6, 3])
    SSIM = np.zeros([6, 3])
    x, y, bands = Landsat.shape
    without_tt_error = np.zeros([x, y, 6])
    THST_error = np.zeros([x, y, 6])
    for i in range(6):
        Landsat_i = Landsat[:, :, band[i]]
        THSTNET_i = THSTNET[:, :, band[i]]
        st1_i = without_tt[:, :, band[i]]

        predic = np.array([st1_i, THSTNET_i])
        without_tt_error[:, :, i] = abs(Landsat_i - st1_i)
        THST_error[:, :, i] = abs(Landsat_i - THSTNET_i)
        for j in range(2):
            label = Landsat_i
            image = predic[j]
            index = np.where(abs(image - label) > 10)
            image[index] = 0
            label[index] = 0

            SSIM[i, j] = ssim_numpy(label - 250, image - 250, val_range=50)

            y_test = np.reshape(label, [-1])
            y_predict = np.reshape(image, [-1])

            mse = np.sum((y_test - y_predict) ** 2) / len(y_test)
            rmse = sqrt(mse)
            mae = np.sum(np.absolute(y_test - y_predict)) / len(y_test)

            RMSE[i, j] = rmse
            MAE[i, j] = mae

    im_proj, im_Geotrans = GetGeoInformation(r'D:\lansat_fine.tif')
    WriteTiff(r'D:\without_tt_error.tif', without_tt_error, im_Geotrans, im_proj, gdal.GDT_Float32)
    WriteTiff(r'D:\THST_error.tif', THST_error, im_Geotrans, im_proj, gdal.GDT_Float32)

    np.savetxt(r'D:\RMSE.txt', RMSE)
    np.savetxt(r'D:\SSIM.txt', SSIM)

if __name__ == '__main__':
    cal_seven()
