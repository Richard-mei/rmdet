import torch.nn as nn
import torch.nn.functional as F
import torch
from ..utils import LOSSES, weighted_loss, build_loss, multi_apply, box_iou


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


@weighted_loss
def quality_focal_loss(pred, target, beta=2.0, use_sigmoid=True):
    r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert len(target) == 2, """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    label, score = target
    if use_sigmoid:
        func = F.binary_cross_entropy_with_logits
    else:
        func = F.binary_cross_entropy

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid() if use_sigmoid else pred
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = func(pred, zerolabel, reduction='none') * scale_factor.pow(beta)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()
    # positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = func(pred[pos, pos_label], score[pos],
        reduction='none') * scale_factor.abs().pow(beta)

    loss = loss.sum(dim=1, keepdim=False)
    return loss


@weighted_loss
def distribution_focal_loss(pred, label):
    r"""Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted general distribution of bounding boxes
            (before softmax) with shape (N, n+1), n is the max value of the
            integral set `{0, ..., n}` in paper.
        label (torch.Tensor): Target distance label for bounding boxes with
            shape (N,).

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    dis_left = label.long()
    dis_right = dis_left + 1
    weight_left = dis_right.float() - label
    weight_right = label - dis_left.float()
    loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left \
        + F.cross_entropy(pred, dis_right, reduction='none') * weight_right
    return loss


@LOSSES.register_module()
class QualityFocalLoss(nn.Module):
    r"""Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.

    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 use_sigmoid=True,
                 beta=2.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(QualityFocalLoss, self).__init__()
        # assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * quality_focal_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            use_sigmoid=self.use_sigmoid,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_cls


@LOSSES.register_module()
class DistributionFocalLoss(nn.Module):
    r"""Distribution Focal Loss (DFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(DistributionFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted general distribution of bounding
                boxes (before softmax) with shape (N, n+1), n is the max value
                of the integral set `{0, ..., n}` in paper.
            target (torch.Tensor): Target distance label for bounding boxes
                with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * distribution_focal_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_cls

#
# class GFocalLoss(nn.Module):
#     def __init__(self, cls_loss, reg_loss, reg_max=None, **kwargs):
#         super(GFocalLoss, self).__init__()
#         self.cls_loss = build_loss(cls_loss)
#         self.reg_loss = build_loss(reg_loss)
#         self.reg_max = 16 if reg_max is None else reg_max
#
#         self.integral = Integral(self.reg_max)
#
#     def forward(self, inputs):
#         preds, annots, anchors = inputs
#         cls_logits, reg_preds = preds[..., 0:-4], preds[..., -4:]
#         boxes = annots[..., :4]
#         classes = annots[..., -1]
#         bacth_size = cls_logits.shape[0]
#
#         cls_logit_list = [cls_logits[i, :, :] for i in range(bacth_size)]
#         reg_pred_list = [reg_preds[i, :, :] for i in range(bacth_size)]
#         gt_box_list = [boxes[i, :, :] for i in range(bacth_size)]
#         gt_classes_list = [classes[i, :] for i in range(bacth_size)]
#         anchors_list = [anchors for _ in range(bacth_size)]  # [x1, y1, x2, y2]
#
#         targets_list, pos_anchors_ind_list, assigned_boxes_list, assigned_classes_list, num_pos_list = multi_apply(
#             self.get_targets,
#             anchors_list,
#             gt_box_list,
#             gt_classes_list,
#             cls_logit_list)
#         cls_loss_list, reg_loss_list = multi_apply(
#             self.loss_single,
#             cls_logit_list,
#             targets_list,
#             num_pos_list,
#             pos_anchors_ind_list,
#             assigned_boxes_list,
#             reg_pred_list,
#             anchors_list)
#
#         cls_loss = torch.stack(cls_loss_list).mean()
#         reg_loss = torch.stack(reg_loss_list).mean()
#         total_loss = cls_loss + reg_loss
#         return {'cls_loss': cls_loss, 'reg_loss': reg_loss, 'total_loss': total_loss}
#
#     # @staticmethod
#     # def get_targets(anchors, gt_boxes, gt_classes, cls_logit):
#     #     """
#     #     # 将一个batch的预测结果按照每个level进行损失计算。loss <== loss_single
#     #     计算所有Anchor和真值框的IoU，小于0.4为背景，大于0.5时，选择与Anchor的IoU最大的真值框对应的类别作为此Anchor分类结果。
#     #     Args:
#     #         anchors: All anchors for one image.
#     #         gt_boxes: Ground truth boxes for one image.
#     #         cls_logit: Classes predictions for one image.
#     #         gt_classes: Ground truth classes for one image.
#     #
#     #     Returns:
#     #         targets: 根据IoU进行正负样本筛选，1为正样本(IoU>=0.5，并进行0ne-hot编码)，0为背景（IoU<0.4为背景），-1丢弃？
#     #         pos_anchors_ind: 正样本（IoU>=0.5）对应的anchor位置
#     #         assigned_boxes: 对于每个Anchor，取与其IoU最大的真值框作为标记框(回归目标框)
#     #         assigned_classes: 对于每个Anchor，取与其IoU最大的真值框对应类别作为预测类别(分类目标类别)
#     #         num_pos: 所有最大IoU>=0.5的Anchor总数//正样本数
#     #     """
#     #     mask1 = gt_boxes[:, 0] == -1
#     #     mask = ~mask1
#     #     # mask = gt_boxes[:, 0] not in [-1, 0]
#     #     gt_boxes = gt_boxes[mask]
#     #     gt_classes = gt_classes[mask]
#     #     IoU = box_iou(anchors, gt_boxes)  # [num_anchors, num_gt_boxes]
#     #
#     #     iou_max, max_ind = torch.max(IoU, dim=1)  # (sum(H*W)*A,)  与每个anchor对应IoU最大的真值框索引
#     #     targets = torch.ones_like(cls_logit) * -1  # (sum(H*W)*A, class_num)
#     #
#     #     targets[iou_max < 0.4, :] = 0  # bg
#     #
#     #     pos_anchors_ind = iou_max >= 0.5  # (?,)
#     #     num_pos = torch.clamp(pos_anchors_ind.sum().float(), min=1.0)
#     #
#     #     assigned_classes = gt_classes[max_ind]  # (sum(H*W)*A, )  # 类别编号 1-20  每个anchor对应一个真值类别
#     #     assigned_boxes = gt_boxes[max_ind, :]  # (sum(H*W)*A, 4)  # 每个anchor对应一个与其IoU最大的真值框
#     #
#     #     targets[pos_anchors_ind, :] = 0
#     #     targets[pos_anchors_ind, (assigned_classes[pos_anchors_ind]).long() - 1] = 1
#     #
#     #     return targets, pos_anchors_ind, assigned_boxes, assigned_classes, num_pos
#     #
#     # def loss_single(self, cls_logit, targets, num_pos, pos_anchor_ind, assigned_boxes, per_reg_pred, anchors):
#     #     """
#     #     Args:
#     #         cls_logit: 一张图片的分类预测结果 [sum(W*H*A), num_classes]
#     #         targets: 一张图片的分类预测结果的One-hot编码 [sum(W*H*A), num_classes]  取值为-1，0，1
#     #         num_pos:  一张图片上的正样本(Anchor)数
#     #         pos_anchor_ind:  正样本indices [sum(W*H*A),] 值为 True, False
#     #         assigned_boxes:  每个Anchor对应的与其IoU最大的真值框
#     #         per_reg_pred:  一张图片的回归预测结果  [sum(W*H*A), 4]
#     #         anchors:   生成的Anchor  [sum(W*H*A), 4]  [x1, y1, x2, y2]
#     #
#     #     Returns:
#     #         cls_loss: 分类损失 FocalLoss
#     #         reg_loss: 回归损失 SmoothL1 // GIoU // DIoU // CIoU
#     #     """
#     #     anchor_widths = anchors[:, 2] - anchors[:, 0]
#     #     anchor_heights = anchors[:, 3] - anchors[:, 1]
#     #     anchor_ctr_x = anchors[:, 0] + anchor_widths * 0.5
#     #     anchor_ctr_y = anchors[:, 1] + anchor_heights * 0.5
#     #
#     #     cls_loss = self.cls_loss(cls_logit, targets).view(1) / num_pos
#     #     reg_loss = self.reg_loss(
#     #         pos_anchor_ind,
#     #         [anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y],
#     #         assigned_boxes,
#     #         per_reg_pred
#     #     )
#     #     return cls_loss, reg_loss
#     #
#
#     def anchor_center(self, anchors):
#         """Get anchor centers from anchors.
#
#         Args:
#             anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.
#
#         Returns:
#             Tensor: Anchor centers with shape (N, 2), "xy" format.
#         """
#         anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
#         anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
#         return torch.stack([anchors_cx, anchors_cy], dim=-1)
#
#     def loss_single(self, anchors, cls_score, bbox_pred, labels, label_weights,
#                     bbox_targets, stride, num_total_samples):
#         """Compute loss of a single scale level.
#
#         Args:
#             anchors (Tensor): Box reference for each scale level with shape
#                 (N, num_total_anchors, 4).
#             cls_score (Tensor): Cls and quality joint scores for each scale
#                 level has shape (N, num_classes, H, W).
#             bbox_pred (Tensor): Box distribution logits for each scale
#                 level with shape (N, 4*(n+1), H, W), n is max value of integral
#                 set.
#             labels (Tensor): Labels of each anchors with shape
#                 (N, num_total_anchors).
#             label_weights (Tensor): Label weights of each anchor with shape
#                 (N, num_total_anchors)
#             bbox_targets (Tensor): BBox regression targets of each anchor wight
#                 shape (N, num_total_anchors, 4).
#             stride (tuple): Stride in this scale level.
#             num_total_samples (int): Number of positive samples that is
#                 reduced over all GPUs.
#
#         Returns:
#             dict[str, Tensor]: A dictionary of loss components.
#         """
#         assert stride[0] == stride[1], 'h stride is not equal to w stride!'
#         anchors = anchors.reshape(-1, 4)
#         cls_score = cls_score.permute(0, 2, 3,
#                                       1).reshape(-1, self.cls_out_channels)
#         bbox_pred = bbox_pred.permute(0, 2, 3,
#                                       1).reshape(-1, 4 * (self.reg_max + 1))
#         bbox_targets = bbox_targets.reshape(-1, 4)
#         labels = labels.reshape(-1)
#         label_weights = label_weights.reshape(-1)
#
#         # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
#         bg_class_ind = self.num_classes
#         pos_inds = ((labels >= 0)
#                     & (labels < bg_class_ind)).nonzero().squeeze(1)
#         score = label_weights.new_zeros(labels.shape)
#
#         if len(pos_inds) > 0:
#             pos_bbox_targets = bbox_targets[pos_inds]
#             pos_bbox_pred = bbox_pred[pos_inds]
#             pos_anchors = anchors[pos_inds]
#             pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]
#
#             weight_targets = cls_score.detach()
#             weight_targets = weight_targets.max(dim=1)[0][pos_inds]
#             pos_bbox_pred_corners = self.integral(pos_bbox_pred)
#             pos_decode_bbox_pred = distance2bbox(pos_anchor_centers,
#                                                  pos_bbox_pred_corners)
#             pos_decode_bbox_targets = pos_bbox_targets / stride[0]
#             score[pos_inds] = bbox_overlaps(
#                 pos_decode_bbox_pred.detach(),
#                 pos_decode_bbox_targets,
#                 is_aligned=True)
#             pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
#             target_corners = bbox2distance(pos_anchor_centers,
#                                            pos_decode_bbox_targets,
#                                            self.reg_max).reshape(-1)
#
#             # regression loss
#             loss_bbox = self.loss_bbox(
#                 pos_decode_bbox_pred,
#                 pos_decode_bbox_targets,
#                 weight=weight_targets,
#                 avg_factor=1.0)
#
#             # dfl loss
#             loss_dfl = self.loss_dfl(
#                 pred_corners,
#                 target_corners,
#                 weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
#                 avg_factor=4.0)
#         else:
#             loss_bbox = bbox_pred.sum() * 0
#             loss_dfl = bbox_pred.sum() * 0
#             weight_targets = torch.tensor(0).cuda()
#
#         # cls (qfl) loss
#         loss_cls = self.loss_cls(
#             cls_score, (labels, score),
#             weight=label_weights,
#             avg_factor=num_total_samples)
#
#         return loss_cls, loss_bbox, loss_dfl, weight_targets.sum()
#
#     @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
#     def loss(self,
#              cls_scores,
#              bbox_preds,
#              gt_bboxes,
#              gt_labels,
#              img_metas,
#              gt_bboxes_ignore=None):
#         """Compute losses of the head.
#
#         Args:
#             cls_scores (list[Tensor]): Cls and quality scores for each scale
#                 level has shape (N, num_classes, H, W).
#             bbox_preds (list[Tensor]): Box distribution logits for each scale
#                 level with shape (N, 4*(n+1), H, W), n is max value of integral
#                 set.
#             gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
#                 shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
#             gt_labels (list[Tensor]): class indices corresponding to each box
#             img_metas (list[dict]): Meta information of each image, e.g.,
#                 image size, scaling factor, etc.
#             gt_bboxes_ignore (list[Tensor] | None): specify which bounding
#                 boxes can be ignored when computing the loss.
#
#         Returns:
#             dict[str, Tensor]: A dictionary of loss components.
#         """
#
#         featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
#         assert len(featmap_sizes) == self.anchor_generator.num_levels
#
#         device = cls_scores[0].device
#         anchor_list, valid_flag_list = self.get_anchors(
#             featmap_sizes, img_metas, device=device)
#         label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
#
#         cls_reg_targets = self.get_targets(
#             anchor_list,
#             valid_flag_list,
#             gt_bboxes,
#             img_metas,
#             gt_bboxes_ignore_list=gt_bboxes_ignore,
#             gt_labels_list=gt_labels,
#             label_channels=label_channels)
#         if cls_reg_targets is None:
#             return None
#
#         (anchor_list, labels_list, label_weights_list, bbox_targets_list,
#          bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets
#
#         num_total_samples = reduce_mean(
#             torch.tensor(num_total_pos).cuda()).item()
#         num_total_samples = max(num_total_samples, 1.0)
#
#         losses_cls, losses_bbox, losses_dfl,\
#             avg_factor = multi_apply(
#                 self.loss_single,
#                 anchor_list,
#                 cls_scores,
#                 bbox_preds,
#                 labels_list,
#                 label_weights_list,
#                 bbox_targets_list,
#                 self.anchor_generator.strides,
#                 num_total_samples=num_total_samples)
#
#         avg_factor = sum(avg_factor)
#         avg_factor = reduce_mean(avg_factor).item()
#         losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
#         losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))
#         return dict(
#             loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl)
#
#
#     def get_targets(self,
#                     anchor_list,
#                     valid_flag_list,
#                     gt_bboxes_list,
#                     img_metas,
#                     gt_bboxes_ignore_list=None,
#                     gt_labels_list=None,
#                     label_channels=1,
#                     unmap_outputs=True):
#         """Get targets for GFL head.
#
#         This method is almost the same as `AnchorHead.get_targets()`. Besides
#         returning the targets as the parent method does, it also returns the
#         anchors as the first element of the returned tuple.
#         """
#         num_imgs = len(img_metas)
#         assert len(anchor_list) == len(valid_flag_list) == num_imgs
#
#         # anchor number of multi levels
#         num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
#         num_level_anchors_list = [num_level_anchors] * num_imgs
#
#         # concat all level anchors and flags to a single tensor
#         for i in range(num_imgs):
#             assert len(anchor_list[i]) == len(valid_flag_list[i])
#             anchor_list[i] = torch.cat(anchor_list[i])
#             valid_flag_list[i] = torch.cat(valid_flag_list[i])
#
#         # compute targets for each image
#         if gt_bboxes_ignore_list is None:
#             gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
#         if gt_labels_list is None:
#             gt_labels_list = [None for _ in range(num_imgs)]
#         (all_anchors, all_labels, all_label_weights, all_bbox_targets,
#          all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
#              self._get_target_single,
#              anchor_list,
#              valid_flag_list,
#              num_level_anchors_list,
#              gt_bboxes_list,
#              gt_bboxes_ignore_list,
#              gt_labels_list,
#              img_metas,
#              label_channels=label_channels,
#              unmap_outputs=unmap_outputs)
#         # no valid anchors
#         if any([labels is None for labels in all_labels]):
#             return None
#         # sampled anchors of all images
#         num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
#         num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
#         # split targets to a list w.r.t. multiple levels
#         anchors_list = images_to_levels(all_anchors, num_level_anchors)
#         labels_list = images_to_levels(all_labels, num_level_anchors)
#         label_weights_list = images_to_levels(all_label_weights,
#                                               num_level_anchors)
#         bbox_targets_list = images_to_levels(all_bbox_targets,
#                                              num_level_anchors)
#         bbox_weights_list = images_to_levels(all_bbox_weights,
#                                              num_level_anchors)
#         return (anchors_list, labels_list, label_weights_list,
#                 bbox_targets_list, bbox_weights_list, num_total_pos,
#                 num_total_neg)
#
#     def _get_target_single(self,
#                            flat_anchors,
#                            valid_flags,
#                            num_level_anchors,
#                            gt_bboxes,
#                            gt_bboxes_ignore,
#                            gt_labels,
#                            img_meta,
#                            label_channels=1,
#                            unmap_outputs=True):
#         """Compute regression, classification targets for anchors in a single
#         image.
#
#         Args:
#             flat_anchors (Tensor): Multi-level anchors of the image, which are
#                 concatenated into a single tensor of shape (num_anchors, 4)
#             valid_flags (Tensor): Multi level valid flags of the image,
#                 which are concatenated into a single tensor of
#                     shape (num_anchors,).
#             num_level_anchors Tensor): Number of anchors of each scale level.
#             gt_bboxes (Tensor): Ground truth bboxes of the image,
#                 shape (num_gts, 4).
#             gt_bboxes_ignore (Tensor): Ground truth bboxes to be
#                 ignored, shape (num_ignored_gts, 4).
#             gt_labels (Tensor): Ground truth labels of each box,
#                 shape (num_gts,).
#             img_meta (dict): Meta info of the image.
#             label_channels (int): Channel of label.
#             unmap_outputs (bool): Whether to map outputs back to the original
#                 set of anchors.
#
#         Returns:
#             tuple: N is the number of total anchors in the image.
#                 anchors (Tensor): All anchors in the image with shape (N, 4).
#                 labels (Tensor): Labels of all anchors in the image with shape
#                     (N,).
#                 label_weights (Tensor): Label weights of all anchor in the
#                     image with shape (N,).
#                 bbox_targets (Tensor): BBox targets of all anchors in the
#                     image with shape (N, 4).
#                 bbox_weights (Tensor): BBox weights of all anchors in the
#                     image with shape (N, 4).
#                 pos_inds (Tensor): Indices of postive anchor with shape
#                     (num_pos,).
#                 neg_inds (Tensor): Indices of negative anchor with shape
#                     (num_neg,).
#         """
#         inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
#                                            img_meta['img_shape'][:2],
#                                            self.train_cfg.allowed_border)
#         if not inside_flags.any():
#             return (None, ) * 7
#         # assign gt and sample anchors
#         anchors = flat_anchors[inside_flags, :]
#
#         num_level_anchors_inside = self.get_num_level_anchors_inside(
#             num_level_anchors, inside_flags)
#         assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
#                                              gt_bboxes, gt_bboxes_ignore,
#                                              gt_labels)
#
#         sampling_result = self.sampler.sample(assign_result, anchors,
#                                               gt_bboxes)
#
#         num_valid_anchors = anchors.shape[0]
#         bbox_targets = torch.zeros_like(anchors)
#         bbox_weights = torch.zeros_like(anchors)
#         labels = anchors.new_full((num_valid_anchors, ),
#                                   self.num_classes,
#                                   dtype=torch.long)
#         label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
#
#         pos_inds = sampling_result.pos_inds
#         neg_inds = sampling_result.neg_inds
#         if len(pos_inds) > 0:
#             pos_bbox_targets = sampling_result.pos_gt_bboxes
#             bbox_targets[pos_inds, :] = pos_bbox_targets
#             bbox_weights[pos_inds, :] = 1.0
#             if gt_labels is None:
#                 # Only rpn gives gt_labels as None
#                 # Foreground is the first class
#                 labels[pos_inds] = 0
#             else:
#                 labels[pos_inds] = gt_labels[
#                     sampling_result.pos_assigned_gt_inds]
#             if self.train_cfg.pos_weight <= 0:
#                 label_weights[pos_inds] = 1.0
#             else:
#                 label_weights[pos_inds] = self.train_cfg.pos_weight
#         if len(neg_inds) > 0:
#             label_weights[neg_inds] = 1.0
#
#         # map up to original set of anchors
#         if unmap_outputs:
#             num_total_anchors = flat_anchors.size(0)
#             anchors = unmap(anchors, num_total_anchors, inside_flags)
#             labels = unmap(
#                 labels, num_total_anchors, inside_flags, fill=self.num_classes)
#             label_weights = unmap(label_weights, num_total_anchors,
#                                   inside_flags)
#             bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
#             bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
#
#         return (anchors, labels, label_weights, bbox_targets, bbox_weights,
#                 pos_inds, neg_inds)
#
#     def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
#         split_inside_flags = torch.split(inside_flags, num_level_anchors)
#         num_level_anchors_inside = [
#             int(flags.sum()) for flags in split_inside_flags
#         ]
#         return num_level_anchors_inside
#
