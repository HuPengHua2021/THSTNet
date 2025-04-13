import torch.nn as nn
from stage_two.down_block import Down_featuere
from stage_two.swin_fusion import Upsample
from stage_one import stage_one
import torch

'''
A Two-stage Hierarchical Spatiotemporal fusion network 
for Land Surface Temperature with Transformer
'''


class stage_two(nn.Module):
    def __init__(self, patchsize=256, in_dim=32):
        super(stage_two, self).__init__()
        self.stage2 = THSTNet(patchsize=patchsize, in_dim=in_dim)

    def forward(self, c0, f0, c1, down_feature, up_feature):
        out = self.stage2(c0, f0, c1, down_feature, up_feature)
        return out


class THSTNet(nn.Module):
    def __init__(self, patchsize, in_dim):
        super(THSTNet, self).__init__()
        self.c_down = Down_featuere(patchsize=patchsize, in_dim=in_dim)
        self.f_down = Down_featuere(patchsize=patchsize, in_dim=in_dim)
        self.up_swin = Upsample(patchsize=patchsize, in_dim=in_dim)

    def forward(self, c0, f0, c1, down_feature, up_feature):
        c_fea0 = self.c_down(c0)
        c_fea1 = self.c_down(c1)
        f_fea0 = self.f_down(f0)

        output_fine = self.up_swin(c_fea0, f_fea0, c_fea1, down_feature, up_feature)

        return output_fine


if __name__ == '__main__':
    model = stage_one(patchsize=64)
    model2 = stage_two(patchsize=64)
    c0 = torch.randn([1, 1, 64, 64])
    c1 = torch.randn([1, 1, 64, 64])
    f0 = torch.randn([1, 1, 64, 64])

    down_feature, up_feature, out = model(c0, f0, c1)
    output=model2(c0, f0, c1, down_feature, up_feature)

    print(output.shape)
