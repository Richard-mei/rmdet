import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init
from .general import ConvBN, DiverseBranchBlock, autopad
import Vision.utils.globalVars as glv
from Vision.utils import multi_apply


class SSDHead(nn.Module):
    def __init__(self, c1, nc=20, na=9):
        super(SSDHead, self).__init__()
        assert isinstance(c1, list)

        self.nc = nc
        cls_branch = nn.ModuleList()
        reg_branch = nn.ModuleList()
        for c in c1:
            cls_branch.append(nn.Sequential(*nn.ModuleList([
                nn.Conv2d(c, c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, na * nc, kernel_size=(1, 1))])))
            reg_branch.append(nn.Sequential(*nn.ModuleList([
                nn.Conv2d(c, c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, na * 4, kernel_size=(1, 1))])))

        self.cls_conv = cls_branch
        self.reg_conv = reg_branch

    def forward(self, x):
        assert len(x) == len(self.cls_conv)
        # cls, reg = [], []
        # meg = zip(x, self.cls_conv, self.reg_conv)
        # for m in meg:
        #     f, c, r = m
        #     fc = c(f).permute(0, 2, 3, 1).contiguous().view(f.size(0), -1, self.nc)
        #     fr = r(f).permute(0, 2, 3, 1).contiguous().view(f.size(0), -1, 4)
        #     cls.append(fc)
        #     reg.append(fr)
        # return torch.cat(cls, 1), torch.cat(reg, 1)
        cls, reg = multi_apply(self.forward_once, zip(x, self.cls_conv, self.reg_conv))
        return torch.cat(cls, 1), torch.cat(reg, 1)

    def forward_once(self, m):
        f, c, r = m
        fc = c(f).permute(0, 2, 3, 1).contiguous().view(f.size(0), -1, self.nc)
        fr = r(f).permute(0, 2, 3, 1).contiguous().view(f.size(0), -1, 4)
        return fc, fr

