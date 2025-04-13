import torch
import torch.nn as nn
from .pytorch_ssim import msssim

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.pixel_cri = CharbonnierLoss()

    def forward(self, pred, target):
        loss = self.pixel_cri(pred, target) + \
               (1 - msssim(pred, target, val_range=1, normalize='relu'))
        return loss

class CharbonnierLoss(nn.Module):
    """Charbonnier损失函数的深度网络可以更好地处理异常值，比L2损失函数高能提高超分辨率(SR)性能 """
    def __init__(self):
        super(CharbonnierLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, pred, target):
        diff = torch.add(target, -pred)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return  loss
