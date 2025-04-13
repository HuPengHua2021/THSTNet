from .index_cal import cal_patch_index, test_fill_index
from .loss import ReconstructionLoss
from .pytorch_ssim import ssim_numpy

class Average():
    """Compute the average of values"""

    def __init__(self):
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value):
        self.sum += value
        self.count += 1
        self.avg = self.sum / self.count
