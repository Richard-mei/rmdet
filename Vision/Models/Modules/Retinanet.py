from .general import *
import Vision.utils.globalVars as glv
from Vision.utils import multi_apply


class BottleNeck(nn.Module):
    """
    Basic residual block for resnet 50.
    """

    def __init__(self, c1, c2, s=1, expansion=4, act=True, groups=1):
        super(BottleNeck, self).__init__()
        c_ = c2 // expansion
        if glv.get_value('USE_DBB_FLAG', default_value=False):
            self.conv = nn.Sequential(*[
                DiverseBranchBlock(c1, c_, kernel_size=1, nonlinear=nn.ReLU(inplace=True), groups=1),
                DiverseBranchBlock(c_, c_, kernel_size=3, stride=s, padding=1, groups=groups, nonlinear=nn.ReLU(inplace=True)),
                DiverseBranchBlock(c_, c2, kernel_size=1, groups=1)
            ])
            self.ds = DiverseBranchBlock(c1, c2, kernel_size=1, stride=s) if c1 != c2 or s != 1 else nn.Identity()
        elif glv.get_value('USE_CoT_FLAG', default_value=False):
            self.conv = nn.Sequential(*[
                ConvBN(c1, c_, k=1, s=1, act=act),
                CotLayer(dim=c_, kernel_size=3) if s == 1 else ConvBN(c_, c_, k=3, s=s, act=act),
                ConvBN(c_, c2, k=1, s=1, act=False)
            ])
            self.ds = ConvBN(c1, c2, k=1, s=s, act=False) if c1 != c2 or s != 1 else nn.Identity()
        else:
            self.conv = nn.Sequential(*[
                ConvBN(c1, c_, k=1, s=1, act=act, g=1),
                ConvBN(c_, c_, k=3, s=s, act=act, g=groups),
                ConvBN(c_, c2, k=1, s=1, act=False, g=1)
            ])
            self.ds = ConvBN(c1, c2, k=1, s=s, act=False, g=1) if c1 != c2 or s != 1 else nn.Identity()
        self.act = nn.ReLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        x = self.act(torch.add(self.conv(x), self.ds(x)))
        return x


class BasicBlock(nn.Module):
    """
    Basic residual block for resnet 18\34.
    """

    def __init__(self, c1, c2, s=1, expansion=1, act=True):
        super(BasicBlock, self).__init__()
        c_ = c2 // expansion
        if glv.get_value('USE_DBB_FLAG', default_value=False):
            self.conv = nn.Sequential(*[
                DiverseBranchBlock(c1, c_, kernel_size=3, stride=s, padding=1, nonlinear=nn.ReLU(inplace=True) if act else None),
                DiverseBranchBlock(c_, c2, kernel_size=3, stride=1, padding=1)
            ])
            self.ds = DiverseBranchBlock(c1, c2, kernel_size=1, stride=s) if c1 != c2 or s != 1 else nn.Identity()
        elif glv.get_value('USE_CoT_FLAG', default_value=False):
            self.conv = nn.Sequential(*[
                CotLayer(dim=c_, kernel_size=3) if s == 1 else ConvBN(c_, c_, k=3, s=s, act=act),
                CotLayer(dim=c_, kernel_size=3),
            ])
            self.ds = ConvBN(c1, c2, k=1, s=s, act=False) if c1 != c2 or s != 1 else nn.Identity()
        else:
            self.conv = nn.Sequential(*[
                ConvBN(c1, c_, k=3, s=s, act=act),
                ConvBN(c_, c2, k=3, s=1, act=False)
            ])
            self.ds = ConvBN(c1, c2, k=1, s=s, act=False) if c1 != c2 or s != 1 else nn.Identity()
        self.act = nn.ReLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        x = self.act(torch.add(self.conv(x), self.ds(x)))
        return x


# class BasicBlock(nn.Module):
#     """
#     Basic residual block for resnet 18\34.
#     """
#     def __init__(self, c1, c2, s=1, expansion=1, act=True):
#         super(BasicBlock, self).__init__()
#         c_ = c2 // expansion
#         self.conv = nn.Sequential(*[
#             Conv(c1, c_, k=3, s=s, act=act),
#             Conv(c_, c2, k=3, s=1, act=False)
#         ])
#         self.ds = Conv(c1, c2, k=1, s=s, act=False) if c1 != c2 or s != 1 else nn.Identity()
#         self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
#
#     def forward(self, x):
#         x = self.act(torch.add(self.conv(x), self.ds(x)))
#         return x


class Res_Stage(nn.Module):
    def __init__(self, c1, c2, n, s, block, groups=1, act=True):
        super(Res_Stage, self).__init__()
        self.block = nn.ModuleList([
            block(c1, c2, s, act=act, groups=1) if i == 0 else block(c2, c2, 1, act=act, groups=groups) for i in range(n)
            # BottleNeck(c1, c2, s, act=act) if i == 0 else BottleNeck(c2, c2, 1, act=act) for i in range(n)
        ])
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)


class RetinaHead(nn.Module):
    def __init__(self, c1, c2, use_dbb, stacked_convs=2, nc=20, na=9):
        super(RetinaHead, self).__init__()
        self.na = na
        # if glv.get_value('USE_DBB_FLAG', default_value=False) and use_dbb:
        #     self.cls_conv = nn.ModuleList([
        #         DiverseBranchBlock(c1, c2, 3, 1, 1, nonlinear=nn.ReLU()) if i == 0 else
        #         DiverseBranchBlock(c2, c2, 3, 1, 1, nonlinear=nn.ReLU()) for i in range(stacked_convs)
        #     ])
        #     self.reg_conv = nn.ModuleList([
        #         DiverseBranchBlock(c1, c2, 3, 1, 1, nonlinear=nn.ReLU()) if i == 0 else
        #         DiverseBranchBlock(c2, c2, 3, 1, 1, nonlinear=nn.ReLU()) for i in range(stacked_convs)
        #     ])
        # else:
        #     cls_branch = []
        #     reg_branch = []
        #     for i in range(4):
        #         cls_branch.append(nn.Conv2d(c2, c2,
        #                                     kernel_size=3, stride=1, padding=1, bias=True))
        #         reg_branch.append(nn.Conv2d(c2, c2,
        #                                     kernel_size=3, stride=1, padding=1, bias=True))
        #         cls_branch.append(nn.ReLU(inplace=True))
        #         reg_branch.append(nn.ReLU(inplace=True))
        cls_branch = []
        reg_branch = []
        for i in range(stacked_convs):
            cls_branch.append(nn.Conv2d(c1 if i == 0 else c2, c2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True))
            reg_branch.append(nn.Conv2d(c1 if i == 0 else c2, c2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True))
            cls_branch.append(nn.GroupNorm(32, c2))
            reg_branch.append(nn.GroupNorm(32, c2))
            cls_branch.append(nn.ReLU(inplace=True))
            reg_branch.append(nn.ReLU(inplace=True))
        self.cls_conv = nn.Sequential(*cls_branch)
        self.reg_conv = nn.Sequential(*reg_branch)
        self.cls = nn.Conv2d(c2, nc * na, (3, 3), (1, 1), (1, 1), bias=True)
        self.reg = nn.Conv2d(c2, na * 4, (3, 3), (1, 1), (1, 1), bias=True)
        nn.init.constant_(self.cls.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, x):
        clss, bboxs = multi_apply(self.forward_once, x)  # F3, F4, F5
        clss = torch.cat(clss, dim=1)
        bboxs = torch.cat(bboxs, dim=1)
        preds = torch.cat([clss, bboxs], dim=-1)
        return preds

    def forward_once(self, x):
        bs, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        cls = self.cls(self.cls_conv(x)).permute(0, 2, 3, 1).contiguous().view(bs, ny * nx * self.na, -1)
        reg = self.reg(self.reg_conv(x)).permute(0, 2, 3, 1).contiguous().view(bs, ny * nx * self.na, -1)

        return cls, reg


class Detections(nn.Module):
    def __init__(self, c1, c2, stacked_convs=4, nc=80, anchors=None):
        super(Detections, self).__init__()
        # 输入检测头输入输出通道、卷积层数量、类别数、anchor配置参数

        # 1、检测头卷积模块构建
        self.na = len(anchors)
        self.cls_conv = nn.ModuleList([
            Conv(c1, c2, 3, 1, 1, norm=nn.GroupNorm) if i == 0 else Conv(c2, c2, 3, 1, 1, norm=nn.GroupNorm) for i in
            range(stacked_convs)
        ])
        self.reg_conv = nn.ModuleList([
            Conv(c1, c2, 3, 1, 1, norm=nn.GroupNorm) if i == 0 else Conv(c2, c2, 3, 1, 1, norm=nn.GroupNorm) for i in
            range(stacked_convs)
        ])
        self.cls_conv = nn.Sequential(*self.cls_conv)
        self.reg_conv = nn.Sequential(*self.reg_conv)

        # 2、预测层构建及初始化
        self.cls = nn.Conv2d(c2, nc * self.na, (3, 3), (1, 1), (1, 1))
        self.reg = nn.Conv2d(c2, self.na * 4, (3, 3), (1, 1), (1, 1))
        nn.init.constant_(self.cls.bias, -math.log((1 - 0.01) / 0.01))

        # 3、anchor构建（根据输入特征图尺寸确定）
        # self.generator = build_generator(anchors)

        # 4、构建损失
        # self.loss = build_loss()

    # # 4、损失函数计算
    # def loss(self,
    #          cls_scores,
    #          bbox_preds,
    #          gt_bboxes,
    #          gt_labels,
    #          gt_bboxes_ignore=None):
    #     """Compute losses of the head.
    #
    #     Args:
    #         cls_scores (list[Tensor]): Box scores for each scale level
    #             Has shape (N, num_anchors * num_classes, H, W)
    #         bbox_preds (list[Tensor]): Box energies / deltas for each scale
    #             level with shape (N, num_anchors * 4, H, W)
    #         gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
    #             shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
    #         gt_labels (list[Tensor]): class indices corresponding to each box
    #         gt_bboxes_ignore (None | list[Tensor]): specify which bounding
    #             boxes can be ignored when computing the loss. Default: None
    #
    #     Returns:
    #         dict[str, Tensor]: A dictionary of loss components.
    #     """
    #     featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    #     device = cls_scores[0].device
    #     # 1. get anchors for each image.
    #     anchor_list = self.get_anchors(featmap_sizes)
    #
    #     # 2. get loss compute targets according to anchors and ground truth.
    #     cls_reg_targets = self.get_targets(anchor_list, gt_bboxes, gt_labels)
    #     if cls_reg_targets is None:
    #         return None
    #     (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
    #      num_total_pos, num_total_neg) = cls_reg_targets
    #     num_total_samples = (
    #         num_total_pos + num_total_neg if self.sampling else num_total_pos)
    #
    #
    #     num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    #
    #     # concat all level anchors to a single tensor
    #     concat_anchor_list = []
    #     for i in range(len(anchor_list)):
    #         concat_anchor_list.append(torch.cat(anchor_list[i]))
    #     all_anchor_list = images_to_levels(concat_anchor_list,
    #                                        num_level_anchors)
    #
    #     losses_cls, losses_bbox = multi_apply(
    #         self.loss_single,
    #         cls_scores,
    #         bbox_preds,
    #         all_anchor_list,
    #         labels_list,
    #         label_weights_list,
    #         bbox_targets_list,
    #         bbox_weights_list,
    #         num_total_samples=num_total_samples)
    #     return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
    #
    #
    #
    #     pass
    #
    # def get_anchors(self, featmap_sizes):
    #     """Get anchors according to feature map sizes.
    #
    #     Arg:
    #         featmap_sizes (list[tuple]): Multi-level feature map sizes.
    #
    #     Return:
    #         tuple:
    #             anchor_list (list[Tensor]): Anchors of each image.
    #     """
    #     bs = len(featmap_sizes)
    #     multi_level_anchors = self.generator(featmap_sizes)
    #     anchor_list = [multi_level_anchors for _ in range(bs)]
    #     return anchor_list
    #
    # def get_targets(self, anchor_list, gt_boxes, gt_labels):
    #     """Compute regression and classification targets for anchors in
    #     multiple images.
    #
    #     Args:
    #         anchor_list (list[list[Tensor]]): Multi level anchors of each
    #             image. The outer list indicates images, and the inner list
    #             corresponds to feature levels of the image. Each element of
    #             the inner list is a tensor of shape (num_anchors, 4).
    #         valid_flag_list (list[list[Tensor]]): Multi level valid flags of
    #             each image. The outer list indicates images, and the inner list
    #             corresponds to feature levels of the image. Each element of
    #             the inner list is a tensor of shape (num_anchors, )
    #         gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
    #         img_metas (list[dict]): Meta info of each image.
    #         gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
    #             ignored.
    #         gt_labels_list (list[Tensor]): Ground truth labels of each box.
    #         label_channels (int): Channel of label.
    #         unmap_outputs (bool): Whether to map outputs back to the original
    #             set of anchors.
    #
    #     Returns:
    #         tuple: Usually returns a tuple containing learning targets.
    #
    #             - labels_list (list[Tensor]): Labels of each level.
    #             - label_weights_list (list[Tensor]): Label weights of each \
    #                 level.
    #             - bbox_targets_list (list[Tensor]): BBox targets of each level.
    #             - bbox_weights_list (list[Tensor]): BBox weights of each level.
    #             - num_total_pos (int): Number of positive samples in all \
    #                 images.
    #             - num_total_neg (int): Number of negative samples in all \
    #                 images.
    #         additional_returns: This function enables user-defined returns from
    #             `self._get_targets_single`. These returns are currently refined
    #             to properties at each feature map (i.e. having HxW dimension).
    #             The results will be concatenated after the end
    #     """
    #     num_imgs = len(anchor_list)
    #
    #     pass

    def forward(self, x):
        clss, bboxs = multi_apply(self.forward_once, x)
        return torch.cat(clss, dim=1), torch.cat(bboxs, dim=1)

    def forward_once(self, x):
        bs, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        cls = self.cls(self.cls_conv(x)).permute(0, 2, 3, 1).contiguous().view(bs, ny * nx * self.na, -1)
        reg = self.reg(self.reg_conv(x)).permute(0, 2, 3, 1).contiguous().view(bs, ny * nx * self.na, -1)
        return cls, reg


class Decoder(nn.Module):
    def __init__(self, score_thres, nms_iou_thres, max_detection, **kwargs):
        super(Decoder, self).__init__()
        self.score_thres = score_thres
        self.nms_iou_thres = nms_iou_thres
        self.max_detections = max_detection

    def forward(self, inputs_):
        """
        inputs: cls_logits, reg_preds, anchors
        (N, sum(H*W)*A, class_num)
        (N, sum(H*W)*A, 4)
        (sum(H*W)*A, 4)
        """
        cls_logits, reg_preds, anchors = inputs_
        batch_size = cls_logits.shape[0]
        cls_logits = cls_logits.sigmoid_()

        boxes = self.ped2ori_box(reg_preds, anchors)  # (N, sum(H*W)*A, 4)

        cls_scores, cls_ind = torch.max(cls_logits, dim=2)  # (N, sum(H*W)*A)
        cls_ind = cls_ind + 1
        # select topK
        max_det = min(self.max_detections, cls_logits.shape[1])  # topK
        topk_ind = torch.topk(cls_scores, max_det, dim=-1, largest=True, sorted=True)[1]  # (N, topK)
        cls_topk = []
        idx_topk = []
        box_topk = []
        for i in range(batch_size):
            cls_topk.append(cls_scores[i][topk_ind[i]])  # (topK,)
            idx_topk.append(cls_ind[i][topk_ind[i]])  # (topK,)
            box_topk.append(boxes[i][topk_ind[i]])  # (topK,4)
        cls_topk = torch.stack(cls_topk, dim=0)  # (N,topK)
        idx_topk = torch.stack(idx_topk, dim=0)  # (N,topK)
        box_topk = torch.stack(box_topk, dim=0)  # (N,topK,4)

        return self._post_process(cls_topk, idx_topk, box_topk)

    def _post_process(self, topk_scores, topk_inds, topk_boxes):
        """
        topk_scores:(N,topk)
        """
        batch_size = topk_scores.shape[0]
        # mask = topk_scores >= self.score_thres #(N,topK)
        _cls_scores = []
        _cls_idxs = []
        _reg_preds = []
        for i in range(batch_size):
            mask = topk_scores[i] >= self.score_thres
            per_cls_scores = topk_scores[i][mask]  # (?,)
            per_cls_idxs = topk_inds[i][mask]  # (?,)
            per_boxes = topk_boxes[i][mask]  # (?,4)
            nms_ind = self._batch_nms(per_cls_scores, per_cls_idxs, per_boxes)

            _cls_scores.append(per_cls_scores[nms_ind])
            _cls_idxs.append(per_cls_idxs[nms_ind])
            _reg_preds.append(per_boxes[nms_ind])

        return torch.stack(_cls_scores, dim=0), torch.stack(_cls_idxs, dim=0), torch.stack(_reg_preds, dim=0)

    def _batch_nms(self, scores, idxs, boxes):
        """
        scores:(?,)
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        max_coords = boxes.max()

        offsets = idxs.to(boxes) * (max_coords + 1)  # (?,)
        post_boxes = boxes + offsets[:, None]  # (?,4)

        keep = self.box_nms(scores, post_boxes, self.nms_iou_thres)
        return keep

    def box_nms(self, scores, boxes, iou_thres):
        if boxes.shape[0] == 0:
            return torch.zeros(0, device=boxes.device).long()
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = torch.sort(scores, descending=True)[1]  # (?,)
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
            else:
                i = order[0].item()
                keep.append(i)

                xmin = torch.clamp(x1[order[1:]], min=float(x1[i]))
                ymin = torch.clamp(y1[order[1:]], min=float(y1[i]))
                xmax = torch.clamp(x2[order[1:]], max=float(x2[i]))
                ymax = torch.clamp(y2[order[1:]], max=float(y2[i]))

                inter_area = torch.clamp(xmax - xmin, min=0.0) * torch.clamp(ymax - ymin, min=0.0)

                iou = inter_area / (areas[i] + areas[order[1:]] - inter_area + 1e-16)

                mask_ind = (iou <= iou_thres).nonzero().squeeze()

                if mask_ind.numel() == 0:
                    break
                order = order[mask_ind + 1]
        return torch.LongTensor(keep)

    def ped2ori_box(self, reg_preds, anchors):
        """
        reg_preds: (N, sum(H*W)*A, 4)
        anchors:(sum(H*W)*A, 4)
        return (N, sum(H*W)*A, 4) 4:(x1,y1,x2,y2)
        """
        if torch.cuda.is_available():
            mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
            std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
        else:
            mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
            std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        dx, dy, dw, dh = reg_preds[..., 0], reg_preds[..., 1], reg_preds[..., 2], reg_preds[..., 3]  # (N,sum(H*W)*A)
        dx = dx * std[0] + mean[0]
        dy = dy * std[1] + mean[1]
        dw = dw * std[2] + mean[2]
        dh = dh * std[3] + mean[3]

        anchor_w = (anchors[:, 2] - anchors[:, 0]).unsqueeze(0)  # (1,sum(H*W)*A)
        anchor_h = (anchors[:, 3] - anchors[:, 1]).unsqueeze(0)  # (1,sum(H*W)*A)
        anchor_ctr_x = anchors[:, 0].unsqueeze(0) + anchor_w * 0.5  # (1,sum(H*W)*A)
        anchor_ctr_y = anchors[:, 1].unsqueeze(0) + anchor_h * 0.5  # (1,sum(H*W)*A)

        pred_ctr_x = dx * anchor_w + anchor_ctr_x  # (N,sum(H*W)*A)
        pred_ctr_y = dy * anchor_h + anchor_ctr_y  # (N,sum(H*W)*A)
        pred_w = torch.exp(dw) * anchor_w  # (N,sum(H*W)*A)
        pred_h = torch.exp(dh) * anchor_h  # (N,sum(H*W)*A)

        x1 = pred_ctr_x - pred_w * 0.5  # (N,sum(H*W)*A)
        y1 = pred_ctr_y - pred_h * 0.5  # (N,sum(H*W)*A)
        x2 = pred_ctr_x + pred_w * 0.5  # (N,sum(H*W)*A)
        y2 = pred_ctr_y + pred_h * 0.5  # (N,sum(H*W)*A)
        return torch.stack([x1, y1, x2, y2], dim=2)  # (N,sum(H*W)*A,4)


class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_imgs, batch_boxes):
        h, w = batch_imgs.shape[2:]
        batch_boxes[..., [0, 2]] = batch_boxes[..., [0, 2]].clamp_(min=0, max=w - 1)
        batch_boxes[..., [1, 3]] = batch_boxes[..., [1, 3]].clamp_(min=0, max=h - 1)
        return batch_boxes


