import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init
from .general import ConvBN, DiverseBranchBlock, autopad, RepVGGBlock
import Vision.utils.globalVars as glv
import numpy as np

model_urls = {
    'vgg': 'https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth',
}


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


#
# # borrowed from https://github.com/amdegroot/ssd.pytorch/blob/master/ssd.py
# def add_vgg(cfg, batch_norm=False):
#     layers = []
#     in_channels = 3
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         elif v == 'C':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#     conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
#     conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
#     layers += [pool5, conv6,
#                nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
#     return layers
#
#
# def add_extras(cfg, i, size=300):
#     # Extra layers added to VGG for feature scaling
#     layers = []
#     in_channels = i
#     flag = False
#     for k, v in enumerate(cfg):
#         if in_channels != 'S':
#             if v == 'S':
#                 layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
#             else:
#                 layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
#             flag = not flag
#         in_channels = v
#     if size == 512:
#         layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
#         layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))
#     return layers
#
#
# vgg_base = {
#     '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
#             512, 512, 512],
#     '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
#             512, 512, 512],
# }
# extras_base = {
#     '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
#     '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
# }
#
#
# class VGG(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         # size = cfg.INPUT.IMAGE_SIZE
#         size = cfg
#         vgg_config = vgg_base[str(size)]
#         extras_config = extras_base[str(size)]
#
#         self.vgg = nn.ModuleList(add_vgg(vgg_config, batch_norm=True))
#         self.extras = nn.ModuleList(add_extras(extras_config, i=1024, size=size))
#         self.l2_norm = L2Norm(512, scale=20)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         for m in self.extras.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.zeros_(m.bias)
#
#     def init_from_pretrain(self, state_dict):
#         self.vgg.load_state_dict(state_dict)
#
#     def forward(self, x):
#         features = []
#         for i in range(23):
#             x = self.vgg[i](x)
#         s = self.l2_norm(x)  # Conv4_3 L2 normalization
#         features.append(s)
#
#         # apply vgg up to fc7
#         for i in range(23, len(self.vgg)):
#             x = self.vgg[i](x)
#         features.append(x)
#
#         for k, v in enumerate(self.extras):
#             x = F.relu(v(x), inplace=True)
#             if k % 2 == 1:
#                 features.append(x)
#
#         return tuple(features)
#
#
# def vgg(cfg, pretrained=True):
#     model = VGG(cfg)
#     # if pretrained:
#     #     model.init_from_pretrain(load_state_dict_from_url(model_urls['vgg']))
#     return model
#


class VGGStage(nn.Module):
    def __init__(self, c1, c2, n, k=3, s=1, p=None, act=True):
        super(VGGStage, self).__init__()
        if glv.get_value('USE_DBB_FLAG', default_value=False):
            self.conv = nn.Sequential(*[
                DiverseBranchBlock(c1, c2, k, s, autopad(k, p), nonlinear=nn.ReLU() if act else None) if i == 0 else
                DiverseBranchBlock(c2, c2, k, s, autopad(k, p), nonlinear=nn.ReLU()) for i in range(n)
            ])
        else:
            self.conv = nn.Sequential(*[
                ConvBN(c1, c2, k, s, p, act=act) if i == 0 else
                ConvBN(c2, c2, k, s, p, act=act) for i in range(n)
            ])

    def forward(self, x):
        return self.conv(x)


class MaxPool2d(nn.Module):
    def __init__(self, k, s, p, ceil_mode=False):
        super(MaxPool2d, self).__init__()
        self.maxp = nn.MaxPool2d(k, s, p, ceil_mode=ceil_mode)

    def forward(self, x):
        return self.maxp(x)


class RepVGGStage(nn.Module):
    def __init__(self, c1, c2, n, k, s, p=1, g=1, d=1, width_multiplier=None, use_se=False):
        super(RepVGGStage, self).__init__()
        # assert len(width_multiplier) == 4
        # self.use_se = use_se
        # self.in_planes = min(c2, int(c2 * width_multiplier))

        self.conv = nn.Sequential(*nn.ModuleList([RepVGGBlock(c1, c2, kernel_size=k, stride=s, padding=p,
                                                              groups=g, dilation=d, use_se=use_se) if i == 0 else
                                                  RepVGGBlock(c2, c2, kernel_size=k, stride=1, padding=1,
                                                              groups=1, dilation=1, use_se=use_se) for i in range(n)]))

    def forward(self, x):
        out = self.conv(x)
        return out
