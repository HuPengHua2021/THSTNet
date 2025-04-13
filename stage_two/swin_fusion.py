import torch
import torch.nn as nn
from base_models import BasicLayer
from .Texture_Transformer import Texture_trans


class Cross_Stage_Feature_Fusion(nn.Module):
    def __init__(self, in_dim_down, in_dim_up, out_dim, resolution):
        super(Cross_Stage_Feature_Fusion, self).__init__()
        self.convdown = nn.Conv2d(in_dim_down, out_dim, 3, 1, 1)
        self.convup = nn.Conv2d(in_dim_up, out_dim, 3, 1, 1)
        self.project = nn.Conv2d(out_dim * 3, out_dim, 3, 1, 1)
        self.patch = resolution
        self.down_dim = in_dim_down
        self.up_dim = in_dim_up

    def forward(self, down, up, nextup):
        B, L, C = up.shape
        down = down.transpose(1, 2).view(B, self.down_dim, self.patch, self.patch)
        up = up.transpose(1, 2).view(B, self.up_dim, self.patch, self.patch)
        nextup = nextup.transpose(1, 2).view(B, self.up_dim, self.patch, self.patch)
        down = self.convdown(down)
        up = self.convup(up)
        next = torch.cat([down, up, nextup], dim=1)
        next = self.project(next)
        next = next.flatten(2).transpose(1, 2)

        return next


class Upsample(nn.Module):
    def __init__(self, patchsize=256, in_dim=32, block_num=(2, 2, 6, 2)):
        super(Upsample, self).__init__()
        self.patchsize = patchsize
        self.up1 = UpBlock(in_dim * 8 * 2, in_dim * 4, patchsize // 16, block_num[3])
        self.up2 = UpBlock(in_dim * 4 * 2, in_dim * 2, patchsize // 8, block_num[2])
        self.up3 = UpBlock(in_dim * 2 * 2, in_dim, patchsize // 4, block_num[1])
        self.up4 = UpBlock(in_dim * 2, in_dim, patchsize // 2, block_num[0])

        self.CSFF16 = Cross_Stage_Feature_Fusion(in_dim * 8, in_dim * 4, in_dim * 4, patchsize // 16)
        self.CSFF8 = Cross_Stage_Feature_Fusion(in_dim * 4, in_dim * 2, in_dim * 2, patchsize // 8)
        self.CSFF4 = Cross_Stage_Feature_Fusion(in_dim * 2, in_dim * 1, in_dim * 1, patchsize // 4)
        self.CSFF2 = Cross_Stage_Feature_Fusion(in_dim * 1, in_dim * 1, in_dim * 1, patchsize // 2)

        self.out = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(in_dim, 1, 3, 1, 1),
            nn.Tanh()
        )
        self.TT = Texture_trans(in_dim * 8, 2, patchsize//32)

    def forward(self, c_fea0, f_fea0, c_fea1, down_feature, up_feature):
        # x0 = f_fea0[4]
        x0 = self.TT(c_fea0[4], f_fea0[4], c_fea1[4]) + f_fea0[4]

        x1 = self.up1(c_fea0[3], f_fea0[3], c_fea1[3], x0)
        x1 = self.CSFF16(down_feature[0], up_feature[0], x1)

        x2 = self.up2(c_fea0[2], f_fea0[2], c_fea1[2], x1)
        x2 = self.CSFF8(down_feature[1], up_feature[1], x2)

        x3 = self.up3(c_fea0[1], f_fea0[1], c_fea1[1], x2)
        x3 = self.CSFF4(down_feature[2], up_feature[2], x3)

        x4 = self.up4(c_fea0[0], f_fea0[0], c_fea1[0], x3)
        x4 = self.CSFF2(down_feature[3], up_feature[3], x4)

        B, L, C = x4.shape

        x4 = x4.transpose(1, 2).view(B, C, self.patchsize // 2, self.patchsize // 2)
        output_fine = self.out(x4)
        return output_fine


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resolution, block_num):
        super(UpBlock, self).__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.up = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2 * 4, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.TT = Texture_trans(in_channels // 2, 2, resolution)

        self.layer = BasicLayer(dim=out_channels, input_resolution=(resolution, resolution),
                                depth=block_num, num_heads=out_channels // 32, window_size=8, mlp_ratio=1,
                                qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.,
                                norm_layer=nn.LayerNorm)

        self.Lin = nn.Linear(in_channels // 2 * 3, out_channels)

    def forward(self, x_c0, x_f0, x_c1, x_f1):
        B, L, C = x_f1.shape  # 上一阶段纹理变换的结果

        x_f1 = x_f1.transpose(1, 2).view(B, C, self.resolution // 2, self.resolution // 2)
        x_f1 = self.up(x_f1).flatten(2).transpose(1, 2)

        x_f1_pre = self.TT(x_c0, x_f0, x_c1) + x_f0
        x = torch.cat([x_c1, x_f1_pre, x_f1], dim=2)
        x = self.Lin(x)
        x = self.layer(x)

        return x
