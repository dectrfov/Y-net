import pytorch_ssim
import torch.nn as nn
from dwt import dwt

ssim_loss = pytorch_ssim.SSIM()


class WSloss(nn.Module):
    def __init__(self):
        super(MSloss, self).__init__()

    def forward(self, x, y, r=0.7):
        loss = 0
        loss -= ssim_loss(x, y)
        #loss+=L1(x,y)
        l, m, h = 1, 1, 1
        for i in range(2):
            l, m, h = l * r * r, l * r * (1 - r), l * (1 - r) * (1 - r)
            x0, x1, x2 = dwt(x)
            y0, y1, y2 = dwt(y)
            loss = loss - ssim_loss(x1, y1) * 2 * m - ssim_loss(x2, y2) * h
            x, y = x0, y0
        loss -= ssim_loss(x0, y0) * l
        return loss
