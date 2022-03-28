# Fast End-to-End Trainable Guided Filter, https://github.com/wuhuikai/DeepGuidedFilter
import torch
import torch.nn as nn

from torch.nn import init
import torch.nn.functional as F

from guided_filter_pytorch.guided_filter import FastGuidedFilter

def weights_init_identity(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n_out, n_in, h, w = m.weight.data.size()
        # Last Layer
        if n_out < n_in:
            init.xavier_uniform_(m.weight.data)
            return

        # Except Last Layer
        m.weight.data.zero_()
        ch, cw = h // 2, w // 2
        for i in range(n_in):
            m.weight.data[i, i, ch, cw] = 1.0

    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data,   0.0)

class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.bn  = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)


class ConvGuidedFilter(nn.Module):
    def __init__(self, radius=1, norm=nn.BatchNorm2d, chans=3):
        super(ConvGuidedFilter, self).__init__()
        self.chans = chans
        self.box_filter = nn.Conv2d(self.chans, self.chans, kernel_size=3, padding=radius, dilation=radius, bias=False, groups=self.chans)
        self.conv_a = nn.Sequential(nn.Conv2d(self.chans*2, 32, kernel_size=1, bias=False),
                                    norm(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, kernel_size=1, bias=False),
                                    norm(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, self.chans, kernel_size=1, bias=False))
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr, y_lr, x_hr):
        _, _, h_lrx, w_lrx = x_lr.size()
        _, _, h_hrx, w_hrx = x_hr.size()

        N = self.box_filter(x_lr.data.new().resize_((1, self.chans, h_lrx, w_lrx)).fill_(1.0))
        ## mean_x
        mean_x = self.box_filter(x_lr)/N
        ## mean_y
        mean_y = self.box_filter(y_lr)/N
        ## cov_xy
        cov_xy = self.box_filter(x_lr * y_lr)/N - mean_x * mean_y
        ## var_x
        var_x  = self.box_filter(x_lr * x_lr)/N - mean_x * mean_x

        ## A
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return mean_A * x_hr + mean_b

def build_lr_net(norm=AdaptiveNorm, layer=5, input_chans=3):
    layers = [
        nn.Conv2d(input_chans, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(24),
        nn.LeakyReLU(0.2, inplace=True),
    ]

    for l in range(1, layer):
        layers += [nn.Conv2d(24,  24, kernel_size=3, stride=1, padding=2**l,  dilation=2**l,  bias=False),
                   norm(24),
                   nn.LeakyReLU(0.2, inplace=True)]

    layers += [
        nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(24),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(24, input_chans, kernel_size=1, stride=1, padding=0, dilation=1)
    ]

    net = nn.Sequential(*layers)

    net.apply(weights_init_identity)

    return net

class DeepGuidedFilter(nn.Module):
    def __init__(self, radius=1, eps=1e-8):
        super(DeepGuidedFilter, self).__init__()
        self.lr = build_lr_net()
        self.gf = FastGuidedFilter(radius, eps)

    def forward(self, x_lr, x_hr):
        return self.gf(x_lr, self.lr(x_lr), x_hr).clamp(0, 1)

    def init_lr(self, path):
        self.lr.load_state_dict(torch.load(path))

class DeepGuidedFilterAdvanced(DeepGuidedFilter):
    def __init__(self, radius=1, eps=1e-4):
        super(DeepGuidedFilterAdvanced, self).__init__(radius, eps)

        self.guided_map = nn.Sequential(
            nn.Conv2d(3, 15, 1, bias=False),
            AdaptiveNorm(15),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(15, 3, 1)
        )
        self.guided_map.apply(weights_init_identity)

    def forward(self, x_lr, x_hr):
        return self.gf(self.guided_map(x_lr), self.lr(x_lr), self.guided_map(x_hr))

class DeepGuidedFilterConvGF(nn.Module):
    def __init__(self, radius=1, layer=5, input_chans=3, reversedParams=False):
        super(DeepGuidedFilterConvGF, self).__init__()
        self.lr = build_lr_net(layer=layer, input_chans=input_chans)
        self.gf = ConvGuidedFilter(radius, norm=AdaptiveNorm, chans=input_chans)
        self.reversedParams = reversedParams

    def forward(self, x_lr, x_hr):
        if self.reversedParams:
            x_lr, x_hr = x_hr, x_lr
        transformed_lr = self.lr(x_lr)
        return self.gf(x_lr, transformed_lr, x_hr).clamp(0, 1)

    def init_lr(self, path):
        self.lr.load_state_dict(torch.load(path))

class DeepGuidedFilterGuidedMapConvGF(DeepGuidedFilterConvGF):
    def __init__(self, radius=1, dilation=0, c=32, layer=5, input_chans=3, reversedParams=False):
        super(DeepGuidedFilterGuidedMapConvGF, self).__init__(radius, layer, input_chans)

        self.guided_map = nn.Sequential(
            nn.Conv2d(input_chans, c, 1, bias=False) if dilation==0 else \
                nn.Conv2d(input_chans, c, input_chans, padding=dilation, dilation=dilation, bias=False),
            AdaptiveNorm(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, input_chans, 1)
        )

    def forward(self, x_lr, x_hr):
        if self.reversedParams:
            x_lr, x_hr = x_hr, x_lr
        return self.gf(self.guided_map(x_lr), self.lr(x_lr), self.guided_map(x_hr)).clamp(0, 1)