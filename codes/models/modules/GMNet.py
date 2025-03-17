import functools
from warnings import filters
import torch.nn as nn
import torch
import models.modules.arch_util as arch_util
import torch.nn.functional as F
from models.modules.arch_util import initialize_weights
from utils.gpu_memory_log import gpu_memory_log


class Down_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels , out_channels, 3, 2, 1, bias=True), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True), nn.ReLU()
        )

        initialize_weights([self.conv], 0.1)
        
    def forward(self, x):
        return self.conv(x)


class ResidualBlock_noBN(nn.Module):
    def __init__(self, nf=64):
        super().__init__()

        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class SQ_Module(nn.Module):
    def __init__(self, size, out_nc, nf=64):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d([size,size])
        self.conv = nn.Sequential(
            nn.Conv2d(nf, nf*2, 1, 1, 0, bias=True), nn.ReLU(),
            nn.Conv2d(nf*2, nf, 1, 1, 0, bias=True), nn.ReLU(),
            nn.Conv2d(nf, out_nc, 1, 1, 0, bias=True)
        )

        initialize_weights([self.conv], 0.1)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x



class GMNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, act_type='relu', opt=None):
        super(GMNet, self).__init__()

        
        res_block = functools.partial(ResidualBlock_noBN, nf=nf)

        # global branch
        self.down1 = Down_conv(in_nc, nf//4)
        self.down2 = Down_conv(nf//4, nf)
        self.res_y = arch_util.make_layer(res_block, 5)

        self.sq_ker = SQ_Module(3, nf)
        self.sq_chn = SQ_Module(1, nf)
        self.sq_qmax = SQ_Module(1, 1)

        self.mask_est = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True), nn.ReLU(),
            nn.Conv2d(nf, 1, 3, 1, 1, bias=True), nn.Sigmoid(),
        )

        self.att_est = nn.Sigmoid()


        # local branch
        self.down_x = Down_conv(in_nc, nf)
        self.res1 = arch_util.make_layer(res_block, 5)
        self.res2 = arch_util.make_layer(res_block, 5)
        self.res3 = arch_util.make_layer(res_block, 5)

        # tail
        self.upconv = nn.Conv2d(nf, 4*nf, 3, 1, 1, bias=True)
        self.upsampler = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        # initialization
        initialize_weights([self.mask_est, self.upconv, self.HRconv, self.conv_last], 0.1)

    def forward(self, x):
        
        # global
        y = x[1]
        y = self.down1(y)
        y = self.down2(y)
        y = self.res_y(y)

        kernel = self.sq_ker(y)
        cse = self.sq_chn(y)
        qmax = self.sq_qmax(y)

        # local
        x = x[0]
        x = self.down_x(x)
        B, C, H, W = x.shape

        x = self.res1(x)
        mask = F.conv2d(x.view(1,B*C, H, W), kernel.view(B*C, 1, 3,3), stride=1, padding=1, dilation=1, groups=B*C).view(B, C, H, W)
        mask = self.mask_est(mask)
        x = x*mask

        x = self.res2(x)
        x = x*self.att_est(cse)
        
        x = self.res3(x)
        out = self.act(self.upsampler(self.upconv(x)))
        out = self.conv_last(self.act(self.HRconv(out)))

        return out, qmax*out