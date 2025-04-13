import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)

def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5,
                     stride=stride, padding=2, bias=True)

def BN(in_channels):
    return nn.BatchNorm2d(in_channels, eps=1e-06, momentum=0.1,
                          affine=True, track_running_stats=True)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        # self.BN=BN(in_channels)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out

class Texture_trans(nn.Module):
    def __init__(self, in_dim, num_res_blocks, resolution):
        super(Texture_trans, self).__init__()

        self.search_trans = search_Trans()

        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        self.conv_head = conv5x5(in_dim, in_dim)
        self.ResBlocks = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.ResBlocks.append(ResBlock(in_channels=in_dim, out_channels=in_dim))
        self.conv_tail = conv5x5(in_dim, in_dim)
        self.conv_squeeze = conv5x5(in_dim * 2, in_dim)
        # self.BN=BN(in_dim)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, c0, f0, c1):
        B, L, C = c1.shape
        H, W = (self.resolution, self.resolution)

        c1 = c1.transpose(1, 2).view(B, C, H, W)
        c0 = c0.transpose(1, 2).view(B, C, H, W)
        f0 = f0.transpose(1, 2).view(B, C, H, W)
        Delta_c = c1 - c0  # 纹理变化计算

        T, S = self.search_trans(c0, f0, c1)

        x1 = self.conv_head(Delta_c)
        for i in range(self.num_res_blocks):
            x = self.ResBlocks[i](x1)
        x = self.conv_tail(x)
        Delta_c_fea = x + x1

        # Soft Attention
        x = torch.cat((Delta_c_fea, T), dim=1)
        x = self.conv_squeeze(x)
        x = x * S

        out = Delta_c_fea + x
        out = out.flatten(2).transpose(1, 2)
        # 返回细分辨率纹理变化

        return out

class search_Trans(nn.Module):
    def __init__(self):
        super(search_Trans, self).__init__()

    def bis(self, input, dim, index):
        # hard attention:
        views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)


    def forward(self, c0, f0, c1):
        c1_unfold = F.unfold(c1, kernel_size=(3, 3), padding=1)
        c0_unfold = F.unfold(c0, kernel_size=(3, 3), padding=1)
        c0_unfold = c0_unfold.permute(0, 2, 1)

        c0_unfold = F.normalize(c0_unfold, dim=2)
        c1_unfold = F.normalize(c1_unfold, dim=1)

        R = torch.bmm(c0_unfold, c1_unfold)  # [N, Hr*Wr, H*W]
        R_star, R_star_arg = torch.max(R, dim=1)  # [N, H*W]

        f0_unfold = F.unfold(f0, kernel_size=(3, 3), padding=1)
        T_unfold = self.bis(f0_unfold, 2, R_star_arg)
        T = F.fold(T_unfold, output_size=c1.size()[-2:], kernel_size=(3, 3), padding=1) / (3. * 3.)

        S = R_star.view(R_star.size(0), 1, c1.size(2), c1.size(3))

        return T, S

if __name__ == "__main__":
    # 测试输出维度是否正确
    c0 = torch.randn([4, 128 * 128, 32])
    c1 = torch.randn([4, 128 * 128, 32])
    f0 = torch.randn([4, 128 * 128, 32])
    c0 = c0.cuda()
    c1 = c1.cuda()
    f0 = f0.cuda()
    model = Texture_trans(32, 4, 128)
    model.cuda()
    out = model(c0, f0, c1)
    out = out.cpu().detach().numpy()
    print(out.shape)
