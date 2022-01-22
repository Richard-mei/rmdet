from .general import *
import Vision.utils.globalVars as glv


class TinyStage(nn.Module):
    def __init__(self, c1, c2, n, k=3, s=2, p=1, groups=1, act=True, use_se=False):
        super(TinyStage, self).__init__()

        # block = nn.ModuleList([
        #     RepVGGBlock(c1, c2, k, s, padding=p, groups=groups, use_se=use_se) if i == 0 else
        #     DiverseBranchBlock(c2, c2, k, 1, padding=1, groups=groups, nonlinear=nn.ReLU() if act else None)
        #     if glv.get_value('USE_DBB_FLAG', default_value=False)
        #     else RepVGGBlock(c2, c2, k, 1, padding=p, groups=groups, use_se=use_se) for i in range(n)
        # ])

        block = nn.ModuleList([
            DiverseBranchBlock(c1, c2, k, s, p, groups=groups, nonlinear=nn.ReLU(inplace=True) if act else None) if i == 0 else
            DiverseBranchBlock(c2, c2, k, 1, padding=p, groups=groups, nonlinear=nn.ReLU(inplace=True) if act else None)
            if glv.get_value('USE_DBB_FLAG', default_value=False)
            else RepVGGBlock(c2, c2, k, 1, padding=p, groups=groups, use_se=use_se) for i in range(n)
        ])
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)
