import torch.nn as nn
from base_models import BasicLayer, PatchEmbed, PatchMerging


class Down_featuere(nn.Module):
    def __init__(self, patchsize=256, in_dim=32, block_num=(2, 2, 6, 2)):
        super(Down_featuere, self).__init__()
        self.PatchEmbed = PatchEmbed(img_size=patchsize, patch_size=2, in_chans=1, embed_dim=in_dim,
                                     norm_layer=nn.LayerNorm)
        self.down4 = Down_Block(in_dim, in_dim * 2, patchsize // 2, block_num[0])
        self.down8 = Down_Block(in_dim * 2, in_dim * 4, patchsize // 4, block_num[1])
        self.down16 = Down_Block(in_dim * 4, in_dim * 8, patchsize // 8, block_num[2])
        self.down32 = Down_Block(in_dim * 8, in_dim * 8, patchsize // 16, block_num[3])

    def forward(self, x):
        x_down2 = self.PatchEmbed(x)
        x_down4 = self.down4(x_down2)
        x_down8 = self.down8(x_down4)
        x_down16 = self.down16(x_down8)
        x_down32 = self.down32(x_down16)

        return x_down2, x_down4, x_down8, x_down16, x_down32


class Down_Block(nn.Module):
    def __init__(self, in_channels, out_channels, resolution, block_num):
        super(Down_Block, self).__init__()
        self.layer = BasicLayer(dim=in_channels, input_resolution=(resolution, resolution), depth=block_num,
                                num_heads=in_channels // 32, window_size=8, mlp_ratio=1, qkv_bias=True, qk_scale=None,
                                drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm)

        self.downsample = PatchMerging((resolution, resolution), in_channels, out_channels)

    def forward(self, x):
        x = self.layer(x)
        x = self.downsample(x)
        return x
