import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        self.reflection_padding = int(np.floor(kernel_size / 2))
        if self.reflection_padding != 0:
            self.reflection_pad = nn.ReflectionPad2d(1)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.upsample, mode='nearest')
        if self.reflection_padding != 0:
            x = self.reflection_pad(x)
        out = self.conv2d(x)
        return out

###### CIN Layers ##############

class ConditionalInstanceNorm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, track_running_stats=False, printoutputs=False):
        super(ConditionalInstanceNorm, self).__init__()
        self.momentum=momentum
        self.eps=eps
        self.num_features = num_features
        self.printoutputs = printoutputs

    def forward(self, input, beta, gamma):
        out = nn.functional.instance_norm(
            input, running_mean=None, running_var=None, weight=gamma, 
            bias=beta, use_input_stats=True, momentum=self.momentum, eps=self.eps)
        return out


class ResidualBlockCIN(torch.nn.Module):
    """ResidualBlock with Instance Normalization """
    def __init__(self, channels, printoutputs=False, printinputs=False):
        super(ResidualBlockCIN, self).__init__()
        self.conv1 = ConvLayerCIN(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayerCIN(channels, channels, kernel_size=3, stride=1,printoutputs=printoutputs)
        self.relu = torch.nn.ReLU()
        self.num_cin_params = self.conv1.num_cin_params + self.conv2.num_cin_params
        self.printoutputs = printoutputs
        self.printinputs = printinputs

    def forward(self, x, beta=None, gamma=None):
        if self.printinputs:
            print("before_first_res shape:", x.size())
            np.savetxt("before_first_res.txt", x.cpu().view(-1).contiguous().numpy())

        residual = x
        r = int(self.conv1.num_cin_params / 2)
        if beta is None:
            beta, gamma = torch.zeros(r*2), torch.zeros(r*2)
        out = self.relu(self.conv1(x, beta[:r], gamma[:r]))
        out = self.conv2(out, beta[r:len(beta)+1], gamma[r:len(gamma)+1])
        out = out + residual

        if self.printoutputs:
            print("after_last_res shape:", out.size())
            np.savetxt("after_last_res.txt", out.cpu().view(-1).contiguous().numpy())

        return out

class ConvLayerCIN(torch.nn.Module):
    # convolution with reflection padding, 
    # followed by conditional instance normalization 
    def __init__(self, in_channels, out_channels, kernel_size, stride, printoutputs=False):
        super(ConvLayerCIN, self).__init__()
        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride)
        self.in1 = ConditionalInstanceNorm(out_channels, printoutputs=printoutputs)
        self.num_cin_params = out_channels * 2
        self.printoutputs = printoutputs

    def forward(self, x, beta=None, gamma=None):
        out = self.conv(x)
        if beta is None:
            beta, gamma = torch.zeros(int(self.num_cin_params/2)), torch.ones(int(self.num_cin_params/2))
        out = self.in1(out, beta, gamma)
        return out


class UpsampleConvLayerCIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayerCIN, self).__init__()
        self.upsampling = UpsampleConvLayer(in_channels, out_channels, kernel_size, stride, upsample)
        self.in1 = ConditionalInstanceNorm(out_channels)
        self.num_cin_params = out_channels * 2

    def forward(self, x, beta=None, gamma=None):
        out = self.upsampling(x)
        if beta is None:
           beta, gamma = torch.zeros(self.num_cin_params/2), torch.ones(self.num_cin_params/2)
        out = self.in1(out, beta, gamma)
        return out
