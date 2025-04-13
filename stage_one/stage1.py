import torch
import torch.nn as nn
from base_models import BasicLayer, PatchEmbed, PatchMerging


class stage_one(nn.Module):
    def __init__(self, patchsize=256, in_dim=32):
        super(stage_one, self).__init__()
        self.stage_one = ST_mapping_fusion(patchsize=patchsize, in_dim=in_dim)

    def forward(self, c0, f0, c1):
        down_fea, up_fea, out = self.stage_one(c0, c1, f0)
        return down_fea, up_fea, out


class ST_mapping(nn.Module):
    def __init__(self, winsize, patchsize):
        super(ST_mapping, self).__init__()
        self.winsize = winsize
        self.resolution = patchsize

    def patch_cut(self, c0, c1, f0):
        winsize = self.winsize
        f0 = f0.permute(0, 2, 3, 1)
        c0 = c0.permute(0, 2, 3, 1)
        c1 = c1.permute(0, 2, 3, 1)

        B, H, W, C = f0.shape
        f0 = f0.view(B, H // winsize, winsize, W // winsize, winsize, C)
        c0 = c0.view(B, H // winsize, winsize, W // winsize, winsize, C)
        c1 = c1.view(B, H // winsize, winsize, W // winsize, winsize, C)

        win_f0 = f0.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, winsize, winsize, C)
        win_c0 = c0.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, winsize, winsize, C)
        win_c1 = c1.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, winsize, winsize, C)

        return win_f0, win_c0, win_c1

    def cal_F(self, input1, input2):
        input1 = input1.permute(0, 3, 1, 2)
        input2 = input2.permute(0, 3, 1, 2)
        # 中间变量
        intermediate1 = input2 @ input1.transpose(-2, -1)
        intermediate2 = input1 @ input1.transpose(-2, -1)

        # 解算F
        F_val = intermediate1 @ torch.linalg.pinv(intermediate2)

        return F_val

    def predict(self, input, F):
        input = input.permute(0, 3, 1, 2)
        output = F @ input
        return output

    def shape_reduction(self, input):
        window_size = self.winsize
        B = int(input.shape[0] / (self.resolution * self.resolution / window_size / window_size))

        x = input.view(B, -1, self.resolution // window_size,
                       self.resolution // window_size, window_size, window_size)

        x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, self.resolution, self.resolution)
        return x

    def forward(self, c0, c1, f0):
        win_f0, win_c0, win_c1 = self.patch_cut(c0, c1, f0)

        F_time = self.cal_F(win_c0, win_c1)
        F_spatial = self.cal_F(win_c0, win_f0)

        f1_time = self.predict(win_f0, F_time)
        f1_spatial = self.predict(win_c1, F_spatial)

        f1_time = self.shape_reduction(f1_time)
        f1_spatial = self.shape_reduction(f1_spatial)

        # [B, 2C, H, W]
        f1 = torch.cat([f1_time, f1_spatial], dim=1)

        return f1


class ST_mapping_fusion(nn.Module):
    def __init__(self, patchsize=256, in_dim=32, block_num=(2, 2, 6, 2)):
        super(ST_mapping_fusion, self).__init__()
        self.pathsize = patchsize
        self.ST_mapping = ST_mapping(16, patchsize)

        self.PatchEmbed = PatchEmbed(img_size=patchsize, patch_size=2, in_chans=2, embed_dim=in_dim,
                                     norm_layer=nn.LayerNorm)
        self.down4 = swin_down(in_dim, in_dim * 2, patchsize // 2, block_num[0])
        self.down8 = swin_down(in_dim * 2, in_dim * 4, patchsize // 4, block_num[1])
        self.down16 = swin_down(in_dim * 4, in_dim * 8, patchsize // 8, block_num[2])
        self.down32 = swin_down(in_dim * 8, in_dim * 8, patchsize // 16, block_num[3])

        self.up32 = up_fusion_swin(in_dim * 8, in_dim * 4, patchsize // 16, block_num[3])
        self.up16 = up_fusion_swin(in_dim * 4, in_dim * 2, patchsize // 8, block_num[2])
        self.up8 = up_fusion_swin(in_dim * 2, in_dim * 1, patchsize // 4, block_num[1])
        self.up4 = up_fusion_swin(in_dim * 1, in_dim, patchsize // 2, block_num[0])

        self.out = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(in_dim, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, c0, c1, f0):
        f1 = self.ST_mapping(c0, c1, f0)

        x1 = self.PatchEmbed(f1)  # [B,128*128,C]
        x2 = self.down4(x1)  # [B,64*64,2C]
        x3 = self.down8(x2)  # [B,32*32,4C]
        x4 = self.down16(x3)  # [B,16*16,8C]
        x5 = self.down32(x4)  # [B,8*8,8C]

        up_16 = self.up32(x5, x4)  # [B,16*16,4C]
        up_8 = self.up16(up_16, x3)  # [B,32*32,2C]
        up_4 = self.up8(up_8, x2)  # [B,64*64,C]
        up_2 = self.up4(up_4, x1)  # [B,128*128,C]

        B, L, C = up_2.shape
        up = up_2.transpose(1, 2).view(B, C, self.pathsize // 2, self.pathsize // 2)
        output = self.out(up)

        down_feature = [x4, x3, x2, x1]
        up_feature = [up_16, up_8, up_4, up_2]

        return down_feature, up_feature, output


class swin_down(nn.Module):
    def __init__(self, in_channels, out_channels, input_size, block_num):
        super(swin_down, self).__init__()
        self.layer = BasicLayer(dim=in_channels, input_resolution=(input_size, input_size), depth=block_num,
                                num_heads=in_channels // 32, window_size=8, mlp_ratio=1, qkv_bias=True,
                                qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm)
        self.downsample = PatchMerging((input_size, input_size), in_channels, out_channels)

    def forward(self, x):
        x = self.layer(x)
        x = self.downsample(x)
        return x


class up_fusion_swin(nn.Module):
    def __init__(self, in_dims, out_dims, input_size, block_num):
        super(up_fusion_swin, self).__init__()
        self.resolution = input_size
        self.up = nn.Sequential(
            nn.Conv2d(in_dims, in_dims * 4, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.fusion = nn.Linear(in_dims * 2, out_dims)
        self.transformer = BasicLayer(dim=out_dims, input_resolution=(self.resolution, self.resolution),
                                      depth=block_num, num_heads=out_dims // 32, window_size=8, mlp_ratio=1,
                                      qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.,
                                      norm_layer=nn.LayerNorm)

    def forward(self, f1, f1_last):
        B, L, C = f1.shape
        f1 = f1.transpose(1, 2).view(B, C, self.resolution // 2, self.resolution // 2)
        f1 = self.up(f1).flatten(2).transpose(1, 2)
        f1 = torch.cat([f1_last, f1], dim=2)
        f1 = self.fusion(f1)
        f1 = self.transformer(f1)

        return f1


if __name__ == '__main__':
    model = stage_one(patchsize=64)
    c0 = torch.randn([1, 1, 64, 64])
    c1 = torch.randn([1, 1, 64, 64])
    f0 = torch.randn([1, 1, 64, 64])

    down, up, out = model(c0, f0, c1)
    print(out.shape)
