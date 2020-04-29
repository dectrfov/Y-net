import torch
import torch.nn as nn


class vertical_net(nn.Module):
    def __init__(self):
        super(vertical_net, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 16, 3, 2, 1, bias=True)
        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1, bias=True)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1, bias=True)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1, bias=True)
        self.conv5 = nn.Conv2d(358, 3, 1, 1, 0, bias=False)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.dconv1 = nn.Conv2d(128, 64, 3, 1, 1, bias=True)
        self.dconv2 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)
        self.dconv3 = nn.Conv2d(32, 16, 3, 1, 1, bias=True)
        self.dconv4 = nn.Conv2d(16, 3, 3, 1, 1, bias=True)

        self.cconv1 = nn.Conv2d(128, 3, 1, 1, 0, bias=False)
        self.cconv2 = nn.Conv2d(64, 3, 1, 1, 0, bias=False)
        self.cconv3 = nn.Conv2d(32, 3, 1, 1, 0, bias=False)
        self.cconv4 = nn.Conv2d(6, 3, 1, 1, 0, bias=False)
        self.cconv5 = nn.Conv2d(15, 3, 1, 1, 0, bias=False)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = nn.Upsample(scale_factor=2, mode='bilinear')(x4)
        x5 = self.relu(self.dconv1(x5))
        x6 = nn.Upsample(scale_factor=2, mode='bilinear')(x5)
        x6 = self.relu(self.dconv2(x6))
        x7 = nn.Upsample(scale_factor=2, mode='bilinear')(x6)
        x7 = self.relu(self.dconv3(x7))
        x8 = nn.Upsample(scale_factor=2, mode='bilinear')(x7)
        x8 = self.relu(self.dconv4(x8))

        # merge different feature
        x9 = torch.cat([x3, x5], 1)
        x9 = self.relu(self.cconv1(x9))
        x9 = nn.Upsample(scale_factor=8, mode='bilinear')(x9)

        x10 = torch.cat([x2, x6], 1)
        x10 = nn.Upsample(scale_factor=4, mode='bilinear')(x10)
        x10 = self.relu(self.cconv2(x10))

        x11 = torch.cat([x1, x7], 1)
        x11 = nn.Upsample(scale_factor=2, mode='bilinear')(x11)
        x11 = self.relu(self.cconv3(x11))

        x12 = torch.cat([x9, x10, x11, x, x8], 1)
        y = self.relu(self.cconv5(x12))

        return y
