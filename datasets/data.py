import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tools import cal_patch_index


def get_pair_path(cur_date, ref_date, LAN_path, MOD_path):
    file_LAN = np.load(LAN_path)
    file_MOD = np.load(MOD_path)

    size = file_MOD.shape

    paths = np.zeros((4, 1, size[1], size[2]))

    paths[0] = np.expand_dims(file_MOD[cur_date - 1, :, :], axis=0)  # 目标时刻MODIS路径

    paths[1] = np.expand_dims(file_LAN[cur_date - 1, :, :], axis=0)  # 目标时刻LANDSAT路径

    paths[2] = np.expand_dims(file_MOD[ref_date - 1, :, :], axis=0)  # 参考时刻MODIS路径

    paths[3] = np.expand_dims(file_LAN[ref_date - 1, :, :], axis=0)  # 参考时刻LANDSAT路径

    return paths


def load_image_pair(cur_date, ref_date, LAN_path, MOD_path):
    paths = get_pair_path(cur_date, ref_date, LAN_path, MOD_path)
    images = []
    for p in range(4):
        images.append(paths[p])

    return images  # 返回[0,1,2,3]分别对应目标时刻MODIS、LANDSAT，参考时刻MODIS、LANDSAT


def transform_image(image, flip_num, rotate_num0, rotate_num):
    image_mask = np.ones(image.shape)

    negtive_mask = np.where(image < 250)
    inf_mask = np.where(image > 350)

    image = (image - 250) / 100  # 对温度执行放缩

    image_mask[negtive_mask] = 0.0
    image_mask[inf_mask] = 0.0
    image[negtive_mask] = 0.0
    image[inf_mask] = 0.0
    image = image.astype(np.float32)

    if flip_num == 1:
        image = image[:, :, ::-1]

    C, H, W = image.shape
    if rotate_num0 == 1:
        # -90
        if rotate_num == 2:
            image = image.transpose(0, 2, 1)[::-1, :]
        # 90
        elif rotate_num == 1:
            image = image.transpose(0, 2, 1)[:, ::-1]
        # 180
        else:
            image = image.reshape(C, H * W)[:, ::-1].reshape(C, H, W)

    image = torch.from_numpy(image.copy())
    image_mask = torch.from_numpy(image_mask)

    return image, image_mask


# Data Augment, flip、rotate
class PatchSet(Dataset):
    def __init__(self, root_dir, image_dates, image_size, patch_size):
        super(PatchSet, self).__init__()
        self.root_dir = root_dir
        h_list, w_list = cal_patch_index(patch_size, image_size, patch_size // 2)
        self.total_index = len(image_dates) * len(h_list) * len(w_list)

    def __getitem__(self, item):
        images = []

        im = np.load(os.path.join(self.root_dir, str(item) + '.npy'))
        for i in range(4):
            images.append(np.expand_dims(im[i, :, :], axis=0))
        patches = [None] * len(images)
        masks = [None] * len(images)

        flip_num = np.random.choice(2)
        rotate_num0 = np.random.choice(2)
        rotate_num = np.random.choice(3)
        for i in range(len(patches)):
            im = images[i]
            im, im_mask = transform_image(im, flip_num, rotate_num0, rotate_num)
            patches[i] = im
            masks[i] = im_mask

        # 将参考图像对差异过大的地方去除，差异值可调整
        image_mask = np.ones(patches[2].shape)
        ref_diff = abs(patches[2] - patches[3])
        diff_indx = np.where(ref_diff > 0.08)
        image_mask[diff_indx] = 0.0

        gt_mask = masks[0] * masks[1] * masks[2] * masks[3] * image_mask

        return patches[0], patches[1], patches[2], patches[3], gt_mask

    def __len__(self):
        return self.total_index
