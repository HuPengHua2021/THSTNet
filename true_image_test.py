import torch
import argparse
import numpy as np

from osgeo import gdal
from sewar import rmse
from datasets import transform_image
from tools import cal_patch_index, test_fill_index,ssim_numpy

from stage_one import stage_one
from stage_two import stage_two
import time

'''
对待测试的MODIS与Landsat影像对进行预测，每一景用前一对影像作为参考，对应的Landsat作为检验
'''

def get_args_parser():
    parser = argparse.ArgumentParser(description='test model')
    parser.add_argument('--image_size', default=[1152, 1734], type=int, help='the image size (height, width)')

    parser.add_argument('--patch_size', default=256, type=int, help='training sample size')
    parser.add_argument('--in_dim', default=32, type=int, help='image feature encoding dimension')

    parser.add_argument('--data_number', default=23, type=int, help='The number of images to predict')

    parser.add_argument('--LAN_path', default=r'D:\lansat_test.tif',
                        help='Landsat image storage location to be tested')
    parser.add_argument('--MOD_path', default=r'D:\MODIS\modis_test.tif',
                        help='MODIS image storage location to be tested')

    parser.add_argument('--stage_one_para', default=r'D:\stage_two\stage_one_parameters.pth',
                        help='Pretrained parameters of stage one')
    parser.add_argument('--stage_two_para', default=r'D:\stage_two\stage_two_parameters.pth',
                        help='Pretrained parameters of stage two')

    parser.add_argument('--out_path', default=r'D:\test_result\THSTNET.tif',
                        help='Predicted result output position')
    return parser


def load_model_para(model, path):
    model_dict = model.state_dict()
    model_parameters = torch.load(path)
    pretained_dict = {k: v for k, v in model_parameters.items() if k in model_dict}

    model_dict.update(pretained_dict)
    model.load_state_dict(model_dict)
    model.cuda()
    return model


def read_im(path):
    dataset = gdal.Open(path)
    shape = [dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount]
    outputfile = np.zeros((shape[2], shape[0], shape[1]))
    for i in range(shape[2]):
        outputfile[i, :, :] = dataset.GetRasterBand(i + 1).ReadAsArray()

    return outputfile


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

    if len(im_data.shape) == 2:
        new_dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            new_dataset.GetRasterBand(i + 1).WriteArray(im_data[:, :, i])
    del new_dataset


def load_image_pair(cur_date, ref_date, LAN_path, MOD_path):
    file_LAN = read_im(LAN_path)
    file_MOD = read_im(MOD_path)
    size = file_MOD.shape

    paths = np.zeros((4, 1, size[1], size[2]))
    paths[0] = np.expand_dims(file_MOD[cur_date - 1, :, :], axis=0)  # 目标时刻MODIS路径
    paths[1] = np.expand_dims(file_LAN[cur_date - 1, :, :], axis=0)  # 目标时刻LANDSAT路径
    paths[2] = np.expand_dims(file_MOD[ref_date - 1, :, :], axis=0)  # 参考时刻MODIS路径
    paths[3] = np.expand_dims(file_LAN[ref_date - 1, :, :], axis=0)  # 参考时刻LANDSAT路径

    images = []
    for p in range(4):
        images.append(paths[p])

    return images  # 返回[0,1,2,3]分别对应目标时刻MODIS、LANDSAT，参考时刻MODIS、LANDSAT


if __name__ == '__main__':
    opt = get_args_parser().parse_args()
    st_one = stage_one(patchsize=opt.patch_size, in_dim=opt.in_dim)
    st_two = stage_two(patchsize=opt.patch_size, in_dim=opt.in_dim)

    st_one = load_model_para(st_one, opt.stage_one_para)
    st_two = load_model_para(st_two, opt.stage_two_para)
    # 加载训练好的模型参数

    st_one.eval()
    st_two.eval()

    h_list, w_list = cal_patch_index(opt.patch_size, opt.image_size,opt.patch_size//2)
    data_num = opt.data_number

    pred_im = np.zeros([opt.image_size[0], opt.image_size[1], data_num])

    start=time.time()

    for num in [image_index + 1 for image_index in range(data_num)]:
        if num == 1:  # 第一景用最后一景参考
            cur_date = num
            ref_date = data_num
        else:
            cur_date = num  # 其它景用前一景参考
            ref_date = num - 1
        images = load_image_pair(cur_date, ref_date, opt.LAN_path, opt.MOD_path)
        output_image = np.zeros(images[1].shape)
        image_mask = np.ones(images[1].shape)
        for i in range(4):
            negtive_mask = np.where(images[i] < 250)
            inf_mask = np.where(images[i] > 350)
            image_mask[negtive_mask] = 0
            image_mask[inf_mask] = 0

        for i in range(len(h_list)):
            for j in range(len(w_list)):
                h_start = h_list[i]
                w_start = w_list[j]

                input_lr = images[0][:, h_start: h_start + opt.patch_size, w_start: w_start + opt.patch_size]
                ref_lr = images[2][:, h_start: h_start + opt.patch_size, w_start: w_start + opt.patch_size]
                ref_hr = images[3][:, h_start: h_start + opt.patch_size, w_start: w_start + opt.patch_size]

                flip_num = 0
                rotate_num0 = 0
                rotate_num = 0
                input_lr, _ = transform_image(input_lr, flip_num, rotate_num0, rotate_num)
                ref_lr, _ = transform_image(ref_lr, flip_num, rotate_num0, rotate_num)
                ref_hr, _ = transform_image(ref_hr, flip_num, rotate_num0, rotate_num)

                input_lr = input_lr.unsqueeze(0).cuda()
                ref_lr = ref_lr.unsqueeze(0).cuda()
                ref_hr = ref_hr.unsqueeze(0).cuda()

                down, up, output_fine = st_one(ref_lr, ref_hr, input_lr)
                output = st_two(ref_lr, ref_hr, input_lr, down, up)
                output = output.squeeze()

                fill_in, patch_in = test_fill_index(i, j, h_start, w_start, h_list, w_list, opt.patch_size)

                output_image[:, fill_in[0]: fill_in[1], fill_in[2]: fill_in[3]] = \
                    output[patch_in[0]: patch_in[1], patch_in[2]: patch_in[3]].cpu().detach().numpy()

        real_im = images[1] * image_mask
        real_output = (output_image) * 100 + 250
        real_im[0][np.where(real_im[0] < 250)] = real_output[0][np.where(real_im[0] < 250)]

        print(num, rmse(real_im[0], real_output[0]))
        print(num, ssim_numpy(real_im[0]-270, real_output[0]-270, val_range=40))

        pred_im[:, :, num - 1] = real_output[0]

    end=time.time()
    print(end - start)
    print((end - start) / 43)
    outtif_path = opt.out_path
    im_proj, im_Geotrans = GetGeoInformation(opt.LAN_path)
    WriteTiff(outtif_path, pred_im, im_Geotrans, im_proj, gdal.GDT_Float32)
