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
from stage_two import stage_two


def get_args_parser():
    parser = argparse.ArgumentParser(description='Train stage two')
    parser.add_argument('--image_size', default=[1152, 1734], type=int, help='the image size (height, width)')

    parser.add_argument('--patch_size', default=256, type=int, help='training sample size')
    parser.add_argument('--in_dim', default=32, type=int, help='image feature encoding dimension')

    parser.add_argument('--num_epochs', default=80, type=int, help='train epoch number')
    parser.add_argument('--batch_size', default=2, type=int, help='training batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='parameters learning rate')

    parser.add_argument('--data_number', default=43, type=int, help='The number of images for train')
    parser.add_argument('--test_index', default=17, type=int, help='The index of image for val')

    parser.add_argument('--stage_one_para', default=r'D:\stage_one\epoch_80.pth',
                        help='Pretrained parameters of stage one')

    parser.add_argument('--LAN_path', default=r'D:\lansat_fine.npy',
                        help='All landsat images storage location')
    parser.add_argument('--MOD_path', default=r'D:\MODIS\modis_fine.npy',
                        help='All MODIS images storage location')
    parser.add_argument('--train_dir', default=r'D:\model_train', help='Sample storage location')

    parser.add_argument('--save_dir', default=r'D:\stage_two', help='Save training parameters and results')
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
        plt.savefig(save_dir + '/Train_loss_stage2.png')
        plt.show()
    elif name == "rmse":
        plt.cla()
        plt.title('RMSE vs. epoch', fontsize=20)
        plt.plot(x1, y1, 'g-.+')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train accuracy', fontsize=20)
        plt.savefig(save_dir + '/Train_RMSE_stage2.png')
        plt.show()
    elif name == "ssim":
        plt.cla()
        plt.title('SSIM  vs. epoch', fontsize=20)
        plt.plot(x1, y1, 'b-.*')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train SSIM', fontsize=20)
        plt.savefig(save_dir + '\Train_SSIM_stage2.png')
        plt.show()


def train(opt, train_dates, test_dates):
    train_set = PatchSet(opt.train_dir, train_dates, opt.image_size, opt.patch_size)
    train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=opt.batch_size, shuffle=True)

    st_one = stage_one(patchsize=opt.patch_size, in_dim=opt.in_dim)
    st_two = stage_two(patchsize=opt.patch_size, in_dim=opt.in_dim)

    n_params = sum(p.numel() for p in st_two.parameters() if p.requires_grad)
    print('There are %d trainable parameters in stage_two.' % n_params)

    loss_fuction = ReconstructionLoss()

    # 加载stage one训练完成的参数
    st_one_dict = st_one.state_dict()
    st_one_parameters = torch.load(opt.stage_one_para)
    pretained_dict = {k: v for k, v in st_one_parameters.items() if k in st_one_dict}
    st_one_dict.update(pretained_dict)
    st_one.load_state_dict(st_one_dict)

    st_one.cuda()
    st_two.cuda()
    loss_fuction.cuda()

    # optimizer_one = optim.Adam(st_one.parameters(), lr=1e-4, weight_decay=0)  # stage_one使用该学习率
    # scheculer_one = optim.lr_scheduler.ReduceLROnPlateau(optimizer_one, mode='max', factor=0.5, patience=3)

    optimizer_two = optim.Adam(st_two.parameters(), lr=opt.lr, weight_decay=0)  # stage_two使用该学习率
    scheculer_two = optim.lr_scheduler.ReduceLROnPlateau(optimizer_two, mode='max', factor=0.5, patience=3)

    best_ssim = 0.0
    best_epoch = -1
    save_dir = opt.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    epoch_loss = []
    epoch_rmse = []
    epoch_ssim = []  # 存储每次迭代的平均loss，以及验证集的RMSE和SSIM
    for epoch in tqdm(range(opt.num_epochs)):

        st_one.eval()  # stage one为预测模式
        st_two.train()

        g_loss, batch_time = Average(), Average()
        batches = len(train_loader)

        for item, (data, target, ref_lr, ref_target, gt_mask) in tqdm(enumerate(train_loader)):

            t_start = timer()

            data = data.cuda()
            target = target.cuda()
            ref_lr = ref_lr.cuda()
            ref_target = ref_target.cuda()
            gt_mask = gt_mask.float().cuda()


            with torch.no_grad():
                down, up, predict_fine = st_one(ref_lr, ref_target, data)
            out_fine = st_two(ref_lr, ref_target, data, down, up)

            # optimizer_one.zero_grad()
            optimizer_two.zero_grad()

            l_total = loss_fuction(out_fine * gt_mask, target * gt_mask)

            l_total.backward()

            optimizer_two.step()
            # optimizer_one.step()
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

        final_ssim, final_rmse = test(st_one, st_two, test_dates, opt)

        epoch_loss.append(g_loss.avg)
        epoch_rmse.append(final_rmse)
        epoch_ssim.append(final_ssim)

        scheculer_two.step(final_ssim)
        if final_ssim > best_ssim:
            best_ssim = final_ssim
            best_epoch = epoch
            torch.save(st_two.state_dict(), save_dir + '/stage_two_parameters.pth')
            torch.save(st_one.state_dict(), save_dir + '/stage_one_parameters.pth')

        torch.save(st_two.state_dict(), save_dir + '/epoch_%d.pth' % (epoch + 1))
        print('Best Epoch is %d' % (best_epoch + 1), 'SSIM is %.4f' % best_ssim)
        print('------------------')

    data_save = np.zeros([len(epoch_loss), 3])
    data_save[:, 0] = np.array(epoch_loss)
    data_save[:, 1] = np.array(epoch_rmse)
    data_save[:, 2] = np.array(epoch_ssim)

    result_save = pd.DataFrame(data=data_save, index=None, columns=['loss', 'rmse', 'ssim'])
    result_save.to_csv(save_dir + '/stage_two_result.csv')

    draw_fig(epoch_loss, 'loss', opt.num_epochs, save_dir)
    draw_fig(epoch_ssim, 'ssim', opt.num_epochs, save_dir)
    draw_fig(epoch_rmse, 'rmse', opt.num_epochs, save_dir)


def test(st_one, st_two, test_dates, opt):
    st_one.eval()
    st_two.eval()

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
            image_mask[inf_mask] = 0  # 将不符合温度区间的异常值移除

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

                down, up, predict_one = st_one(ref_lr, ref_hr, input_lr)
                out_fine = st_two(ref_lr, ref_hr, input_lr, down, up)

                output = out_fine.squeeze()

                fill_in, patch_in = test_fill_index(i, j, h_start, w_start, h_list, w_list, opt.patch_size)

                output_image[:, fill_in[0]: fill_in[1], fill_in[2]: fill_in[3]] = \
                    output[patch_in[0]: patch_in[1], patch_in[2]: patch_in[3]].cpu().detach().numpy()

        output_image = (output_image) * 100 + 250
        real_im = images[1] * image_mask
        real_output = output_image * image_mask

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
    all_index = [i + 1 for i in range(opt.data_number)]
    for i in range(len(all_index)):
        if all_index[i] not in [opt.test_index]:
            train_dates.append(all_index[i])  # 训练的时刻
        else:
            test_dates.append(all_index[i])  # 验证的时刻

    train(opt, train_dates, test_dates)
