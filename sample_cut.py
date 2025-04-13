import os
import argparse
import numpy as np
from tqdm import tqdm
from tools import cal_patch_index


def get_args_parser():
    parser = argparse.ArgumentParser(description='Obtain training sample')
    parser.add_argument('--data_number', default=43, type=int, help='The number of images to train')
    parser.add_argument('--test_index', default=17, type=int, help='The index of image for val')
    parser.add_argument('--image_size', default=[1152, 1734], type=int, help='the image size (height, width)')
    parser.add_argument('--patch_size', default=256, type=int, help='training images crop size')
    parser.add_argument('--LAN_path', default=r'D:\SwinT_FYMOLA\landsat\lansat_fine.npy',
                        help='All landsat images storage location')
    parser.add_argument('--MOD_path', default=r'D:\SwinT_FYMOLA\MODIS\modis_fine.npy',
                        help='All MODIS images storage location')
    parser.add_argument('--sample_dir', default=r'D:\SwinT_FYMOLA\model_train', help='Sample storage location')
    return parser


if __name__ == "__main__":
    np.random.seed(2023)
    opt = get_args_parser().parse_args()
    MOD = np.load(opt.MOD_path)
    LAN = np.load(opt.LAN_path)

    all_index = [i + 1 for i in range(opt.data_number)]  # 所有影像的序列索引(下标从1开始)

    h_list, w_list = cal_patch_index(opt.patch_size, opt.image_size, opt.patch_size // 2)
    total_index = 0
    sample_dir = opt.sample_dir

    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # 挑出一景影像作为验证，其余参与训练
    train_images = np.zeros((len(all_index) - 1, 2, 1, opt.image_size[0], opt.image_size[1]))

    train_num = 0
    for k in tqdm(range(len(all_index))):
        if all_index[k] not in [opt.test_index]:  # 非验证图像索引(下标从1计数)
            train_images[train_num, 1] = np.expand_dims(LAN[k, :, :], axis=0)  # 所有的LANDSAT样本数据
            train_images[train_num, 0] = np.expand_dims(MOD[k, :, :], axis=0)  # 所有的MODIS样本数据
            train_num += 1

    for k in tqdm(range(len(all_index) - 1)):  # 影像对数目
        for i in range(len(h_list)):  # 行分块数
            for j in range(len(w_list)):  # 列分块数
                h_start = h_list[i]
                w_start = w_list[j]

                ref_index = np.random.choice(len(all_index) - 1)  # 训练日期中随机挑选一天
                if ref_index == k:
                    if ref_index == 0:
                        ref_index += 1
                    else:
                        ref_index -= 1  # 参考日期与预测日期不重复

                images = []
                images.append(train_images[k, 0])
                images.append(train_images[k, 1])
                images.append(train_images[ref_index, 0])
                images.append(train_images[ref_index, 1])

                input_images = []
                for im in images:
                    input_images.append(im[:, h_start: h_start + opt.patch_size, w_start: w_start + opt.patch_size])
                input_images_all = np.concatenate(input_images, axis=0)
                # save the patch for training
                np.save(os.path.join(sample_dir, str(total_index) + '.npy'), input_images_all)
                total_index += 1

    assert total_index == (len(all_index) - 1) * len(h_list) * len(w_list)
