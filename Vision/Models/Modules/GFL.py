from .general import *
import Vision.utils.globalVars as glv
from Vision.utils import multi_apply, build_loss


def featuremapTOmask(featuremap):
    B, C, H, W = featuremap.shape
    heat_map = torch.clamp_min(featuremap, 0)
    heat_map = torch.mean(heat_map, dim=1)
    max = heat_map.reshape(B, H * W).max(dim=1).values
    heat_map /= max.view(B, 1, 1)

    # # heatmap vis
    # heatmap = heat_map.data.cpu().numpy()
    # heatmap = heatmap.squeeze(0)
    # heatmap = np.uint8(255 * heatmap)
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # # heatmap = cv2.resize(heatmap, (640, 480))
    # cv2.imshow('hm', heatmap)
    # cv2.waitKey(0)

    a = torch.ones_like(heat_map)
    b = torch.ones_like(heat_map) * -1
    mask = torch.where(heat_map > heat_map.mean(), a, b)

    # # mask vis
    # heat_map = mask.data.cpu().numpy()
    # heat_map = heat_map.squeeze(0)
    # heatmap = np.uint8(255 * heat_map)
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # cv2.imshow('mask',heatmap)
    # cv2.waitKey(0)

    # mask = mask.unsqueeze(1)
    mask = mask.unsqueeze(1).repeat(1, 9, 1, 1)
    # mask = mask.permute(0, 2, 3, 1).contiguous()
    mask = mask.permute(0, 2, 3, 1).contiguous().view(B, H * W * 9, -1)

    # end_t=time.time()
    # cost_t=1000*(end_t-start_t)
    # print("===>success processing mask, cost time %.2f ms"%cost_t)

    return mask


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x


class GFocalHead(nn.Module):
    def __init__(self, c1, c2,
                 stacked_convs=4,
                 nc=20,
                 reg_max=16,
                 reg_topk=4,
                 reg_channels=64,
                 loss_dfl=dict(cfg='DistributionFocalLoss', loss_weight=0.25),
                 add_mean=True,
                 **kwargs):
        super(GFocalHead, self).__init__()
        self.stacked_convs = stacked_convs
        self.strides = [8, 16, 32, 64, 128]
        self.reg_max = reg_max
        self.integral = Integral(self.reg_max)
        self.reg_topk = reg_topk
        self.reg_channels = reg_channels
        self.add_mean = add_mean
        self.total_dim = reg_topk
        self.use_gs = False
        self.nc = nc
        if add_mean:
            self.total_dim += 1
        print('total dim = ', self.total_dim * 4)

        self.loss_dfl = build_loss(loss_dfl)

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = c1 if i == 0 else c2
            self.cls_convs.append(ConvBN(chn, c2, 3, s=1, p=1))
            self.reg_convs.append(ConvBN(chn, c2, 3, s=1, p=1))
        self.gfl_cls = nn.Conv2d(c2, nc, 3, padding=1)
        self.gfl_reg = nn.Conv2d(
            c2, 4 * (self.reg_max + 1), 3, padding=1)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.strides])

        conf_vector = [nn.Conv2d(4 * self.total_dim, self.reg_channels, 1)]
        conf_vector += [nn.ReLU(inplace=True)]
        conf_vector += [nn.Conv2d(self.reg_channels, 1, 1), nn.Sigmoid()]

        self.reg_conf = nn.Sequential(*conf_vector)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification and quality (IoU)
                    joint scores for all scale levels, each is a 4D-tensor,
                    the channel number is num_classes.
                bbox_preds (list[Tensor]): Box distribution logits for all
                    scale levels, each is a 4D-tensor, the channel number is
                    4*(n+1), n is max value of integral set.
        """
        clsc, regp, mask = multi_apply(self.forward_single, feats, self.scales)
        clsc = torch.cat(clsc, dim=1)
        regp = torch.cat(regp, dim=1)
        mask = torch.cat(mask, dim=1) if self.use_gs else None

        return {'cls_score': clsc, 'reg_pred': regp, 'gs_mask': mask}

    def forward_single(self, x, scale):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls and quality joint scores for a single
                    scale level the channel number is num_classes.
                bbox_pred (Tensor): Box distribution logits for a single scale
                    level, the channel number is 4*(n+1), n is max value of
                    integral set.
        """

        cls_feat = x
        reg_feat = x
        if self.use_gs:
            mask = featuremapTOmask(x)
        else:
            mask = None
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        bbox_pred = scale(self.gfl_reg(reg_feat)).float()
        N, C, H, W = bbox_pred.size()
        prob = F.softmax(bbox_pred.reshape(N, 4, self.reg_max + 1, H, W), dim=2)
        prob_topk, _ = prob.topk(self.reg_topk, dim=2)

        if self.add_mean:
            stat = torch.cat([prob_topk, prob_topk.mean(dim=2, keepdim=True)],
                             dim=2)
        else:
            stat = prob_topk

        quality_score = self.reg_conf(stat.reshape(N, -1, H, W))
        cls_score = self.gfl_cls(cls_feat).sigmoid() * quality_score
        bbox_pred = self.integral(bbox_pred.permute(0, 2, 3, 1))
        return cls_score.permute(0, 2, 3, 1).contiguous().view(N, -1, self.nc), \
               bbox_pred.contiguous().view(N, -1, 4), mask
