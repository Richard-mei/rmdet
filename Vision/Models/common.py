import collections
from pathlib import Path
from copy import deepcopy
from .Modules import *
from ..utils import DETECTORS, fuse_conv_and_bn, feature_visualization, initialize_weights, intersect_dicts
import Vision.utils.globalVars as glv


def parse_model(d, ch, logger=None, deploy=False, use_dbb=False, use_cot=False):
    glv.init_global_dict()
    glv.set_value('DEPLOY_FLAG', deploy, logger)
    glv.set_value('USE_DBB_FLAG', use_dbb, logger)
    glv.set_value('USE_CoT_FLAG', use_cot, logger)
    assert not (glv.get_value('USE_DBB_FLAG') and glv.get_value('USE_CoT_FLAG'))
    if logger is not None:
        logger.info('%3s%20s%3s%10s  %-45s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    layers, save, c2 = [], [], ch[-1]
    # cfg = []
    # for k, v in d.items():
    #     cfg += d[k]
    # for i, (f, m, n, args) in enumerate(cfg):
    for i, (f, m, n, args) in enumerate(d['backbone'] + d['neck'] + d['head']):
        m = eval(m) if isinstance(m, str) else m
        n_ = n
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except:
                pass
        if m in [Res_Stage, RepVGGStage, VGGStage, TinyStage, C3, BiFPNBlocks]:
            # c1, c2 = ch[f], args[1]
            # args = [c1, c2, *args[2:]]
            args.insert(2, n)
            n = 1
        # elif m is Concat:
        #     c2 = sum([ch[x] for x in f])

        # if n != 1:
        #     args.insert(2, n)
        #     n = 1
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        np = sum([x.numel() for x in m_.parameters()])
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        if logger is not None:
            logger.info('%3s%20s%3s%10.0f  %-45s%-30s' % (i, f, n_, np, t, args))
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


@DETECTORS.register_module()
class Model(nn.Module):
    def __init__(self, c=None, ch=3, nc=None, anchors=None, logger=None, deploy=False, use_dbb=False, use_cot=False):
        super(Model, self).__init__()
        if isinstance(c, dict):
            self.yaml = c  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(c).name
            with open(c) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch], logger=logger, deploy=deploy, use_dbb=use_dbb,
                                            use_cot=use_cot)

        # initialize_weights(self)

    def forward(self, x, visualize=False):
        y, dt = [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
            if visualize:
                # feature_visualization(x, m.type, m.i, save_dir=visualize)
                feature_visualization(x, m.type, m.i, n=2)

        return x

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
            elif isinstance(m, ConvBN) and hasattr(m, 'bn'):
                m.switch_to_deploy()
        # self.info()
        return self

    def load_pretrained(self, pretrained, exclude=None):
        if exclude is None:
            exclude = []
        try:
            self.model.load_state_dict(pretrained)
            print('Load pretrained successfully.')
        except Exception:
            new_dict = collections.OrderedDict()
            for (k, v), k2 in zip(pretrained.items(), self.state_dict().keys()):
                new_dict[k2] = v
                # if any(x in k2 for x in exclude):
                #     exclude = []
                #     nn.init.constant_(new_dict[k2], -math.log((1 - 0.01) / 0.01))
            csd = intersect_dicts(new_dict, self.state_dict(), exclude=exclude)
            self.load_state_dict(csd, strict=False)
            print(f'------------------------------\nLoad fixed pretrained successfully.Layers:\n{csd.keys()}.')
        finally:
            return self


if __name__ == '__main__':
    cfg = '/home/richard/Projects/XXX/models/configs/retinanet.yaml'

    in_put = torch.randn(1, 3, 416, 416).cuda()
    net = Model(cfg).cuda()

    # coco_pretrained = torch.load('/home/richard/Downloads/voc_81.6.pth')
    # new_dict = collections.OrderedDict()
    #
    # for (k, v), k2 in zip(coco_pretrained.items(), net.state_dict().keys()):
    #     new_dict[k2] = v
    # for _ in range(12):
    #     new_dict.popitem(last=True)
    # net.load_state_dict(new_dict, strict=False)
    net.load_state_dict(torch.load('retinanet.pth'))
    # torch.save(net.state_dict(), 'retinanet.pth')
    out = net(in_put)
