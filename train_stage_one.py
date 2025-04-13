import os
import torch
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from sewar import rmse
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from datasets import PatchSet, load_image_pair, transform_image
from tools import cal_patch_index, test_fill_index, ReconstructionLoss, Average, ssim_numpy

from stage_one import stage_one


def get_args_parser():
    parser = argparse.ArgumentParser(description='Train stage one')
    parser.add_argument('--image_size', default=[1152, 1734], type=int, help='the image size (height, width)')

    parser.add_argument('--patch_size', default=256, type=int, help='training sample size')
    parser.add_argument('--in_dim', default=32, type=int, help='image feature encoding dimension')

    parser.add_argument('--num_epochs', default=80, type=int, help='train epoch number')
    parser.add_argument('--batch_size', default=8, type=int, help='training batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='parameters learning rate')

    parser.add_argument('--data_number', default=43, type=int, help='The number of images for train')
    parser.add_argument('--test_index', default=17, type=int, help='The index of image for val')

    parser.add_argument('--LAN_path', default=r'D:\lansat_fine.npy',
                        help='All landsat images storage location')
    parser.add_argument('--MOD_path', default=r'D:\MODIS\modis_fine.npy',
                        help='All MODIS images storage location')
    parser.add_argument('--train_dir', default=r'D:\model_train', help='Sample storage location')

    parser.add_argument('--save_dir', default=r'D:\stage_one', help='Save training parameters')
    return parser


def draw_fig(list, name, epoch, save_dir):
    x1 = range(1, epoch + 1)
    y1 = list
    if name == "loss":
        plt.cla()
        plt.title('Train loss vs. epoch', fontsize=20)
        plt.plot(x1, y1, 'r-.d')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train loss', fontsize=20)
        plt.savefig(save_dir + '/Train_loss_stage1.png')
        plt.show()
    elif name == "rmse":
        plt.cla()
        plt.title('RMSE vs. epoch', fontsize=20)
        plt.plot(x1, y1, 'g-.+')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train accuracy', fontsize=20)
        plt.savefig(save_dir + '/Train_RMSE_stage1.png')
        plt.show()
    elif name == "ssim":
        plt.cla()
        plt.title('SSIM  vs. epoch', fontsize=20)
        plt.plot(x1, y1, 'b-.*')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train SSIM', fontsize=20)
        plt.savefig(save_dir + '\Train_SSIM_stage1.png')
        plt.show()


def train(opt, train_dates, test_dates):
    train_set = PatchSet(opt.train_dir, train_dates, opt.image_size, opt.patch_size)
    train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=opt.batch_size, shuffle=True)

    model = stage_one(patchsize=opt.patch_size,in_dim=opt.in_dim)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('There are %d trainable parameters in stage one.' % n_params)

    loss_function = ReconstructionLoss()

    model.cuda()
    loss_function.cuda()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=0)  # transformer使用该学习率
    scheculer = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_RMSE = 100.0
    best_epoch = -1
    save_dir = opt.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    epoch_loss = []
    epoch_rmse = []
    epoch_ssim = []  # 存储每次迭代的平均loss，以及验证集的RMSE和SSIM
    for epoch in tqdm(range(opt.num_epochs)):
        model.train()
        g_loss, batch_time = Average(), Average()
        batches = len(train_loader)

        for item, (data, target, ref_lr, ref_target, gt_mask) in tqdm(enumerate(train_loader)):

            t_start = timer()

            data = data.cuda()
            target = target.cuda()
            ref_lr = ref_lr.cuda()
            ref_target = ref_target.cuda()
            gt_mask = gt_mask.float().cuda()

            down, up, predict_fine = model(ref_lr, ref_target, data)

            optimizer.zero_grad()
            l_total = loss_function(predict_fine * gt_mask, target * gt_mask)
            l_total.backward()
            optimizer.step()
            # optimizer.zero_grad()

            g_loss.update(l_total.cpu().item())

            t_end = timer()
            batch_time.update(round(t_end - t_start, 4))

            if item % 100 == 99:
                print('[%d/%d][%d/%d] G-Loss: %.4f Batch_Time: %.4f' % (
                    epoch + 1, opt.num_epochs, item + 1, batches, g_loss.avg, batch_time.avg,
                ))
        print('[%d/%d][%d/%d] G-Loss: %.4f Batch_Time: %.4f' % (
            epoch + 1, opt.num_epochs, batches, batches, g_loss.avg, batch_time.avg,
        ))

        final_ssim, final_rmse = test(model, test_dates, opt)

        epoch_loss.append(g_loss.avg)
        epoch_rmse.append(final_rmse)
        epoch_ssim.append(final_ssim)

        scheculer.step(final_rmse)
        if final_rmse < best_RMSE:
            best_RMSE = final_rmse
            best_epoch = epoch
            torch.save(model.state_dict(), save_dir + '/stage_one_parameters.pth')

        torch.save(model.state_dict(), save_dir + '/epoch_%d.pth' % (epoch + 1))
        print('Best Epoch is %d' % (best_epoch + 1), 'SSIM is %.4f' % best_RMSE)
        print('------------------')

    data_save = np.zeros([len(epoch_loss), 3])
    data_save[:, 0] = np.array(epoch_loss)
    data_save[:, 1] = np.array(epoch_rmse)
    data_save[:, 2] = np.array(epoch_ssim)

    result_save = pd.DataFrame(data=data_save, index=None, columns=['loss', 'rmse', 'ssim'])
    result_save.to_csv(save_dir + '/stage_one_result.csv')

    draw_fig(epoch_loss, 'loss', opt.num_epochs, save_dir)
    draw_fig(epoch_ssim, 'ssim', opt.num_epochs, save_dir)
    draw_fig(epoch_rmse, 'rmse', opt.num_epochs, save_dir)


def test(model, test_dates, opt):
    model.eval()

    h_list, w_list = cal_patch_index(opt.patch_size, opt.image_size, opt.patch_size // 2)

    final_ssim = 0.0
    final_rmse = 0.0
    for cur_date in test_dates:
        ref_date = cur_date - 1
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
                rotate_num = 0  # 预测时不进行增强操作
                input_lr, _ = transform_image(input_lr, flip_num, rotate_num0, rotate_num)
                ref_lr, _ = transform_image(ref_lr, flip_num, rotate_num0, rotate_num)
                ref_hr, _ = transform_image(ref_hr, flip_num, rotate_num0, rotate_num)

                input_lr = input_lr.unsqueeze(0).cuda()
                ref_lr = ref_lr.unsqueeze(0).cuda()
                ref_hr = ref_hr.unsqueeze(0).cuda()

                down, up, output = model(ref_lr, ref_hr, input_lr)
                output = output.squeeze()

                fill_in, patch_in = test_fill_index(i, j, h_start, w_start, h_list, w_list, opt.patch_size)

                output_image[:, fill_in[0]: fill_in[1], fill_in[2]: fill_in[3]] = \
                    output[patch_in[0]: patch_in[1], patch_in[2]: patch_in[3]].cpu().detach().numpy()

        output_image = (output_image) * 100 + 250
        real_im = images[1] * image_mask
        real_output = output_image * image_mask
        # plt.imshow(real_output[0],cmap='jet')
        # plt.show()
        final_ssim = ssim_numpy(real_im[0] - 270, real_output[0] - 270, val_range=40)
        final_rmse = rmse(real_im[0], real_output[0])
        print('[%s/%s] RMSE: %.4f SSIM: %.4f' % (
            cur_date, ref_date, final_rmse, final_ssim))

    return final_ssim, final_rmse


if __name__ == '__main__':

    np.random.seed(2023)
    torch.manual_seed(2023)
    torch.cuda.manual_seed_all(2023)
    torch.backends.cudnn.deterministic = True
    opt = get_args_parser().parse_args()

    train_dates = []
    test_dates = []
    all_index = [i + 1 for i in range(opt.data_number)]  # 训练与测试的图像总数,下标从1开始
    for i in range(len(all_index)):
        if all_index[i] not in [opt.test_index]:  # 挑选一张作为测试，与样本裁剪时一致
            train_dates.append(all_index[i])  # 待训练的时刻
        else:
            test_dates.append(all_index[i])  # 待预测的个时刻

    train(opt, train_dates, test_dates)
