from ..utils import LOSSES, build_loss
import torch
import torch.nn.functional as F
import numpy as np
import math


@LOSSES.register_module()
class focal_loss(object):
    def __init__(self, alpha=0.25, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, preds, targets):
        """
        Args:
            preds [sum(H*W)*A, num_classes]: classification predictions.
            targets [sum(W*H*A), num_classes]:  one-hot encoded target classes for each anchor.
        Returns:
            total focal loss for every anchor in a image.
        """
        preds = preds.sigmoid()
        preds = torch.clamp(preds, min=1e-4, max=1. - 1e-4)
        if torch.cuda.is_available():
            alpha_factor = torch.ones(targets.shape).cuda() * self.alpha
        else:
            alpha_factor = torch.ones(targets.shape) * self.alpha

        alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor,
                                   (1. - alpha_factor))  # [sum(W*H*A), num_classes]
        focal_weights = torch.where(torch.eq(targets, 1.), 1 - preds, preds)  # [sum(W*H*A), num_classes]
        focal_weights = alpha_factor * torch.pow(focal_weights, self.gamma)  # [sum(W*H*A), num_classes]

        bce = - (targets * torch.log(preds) + (1. - targets) * torch.log(1. - preds))  # [sum(W*H*A), num_classes]
        cls_loss = focal_weights * bce  # [sum(W*H*A), num_classes]

        if torch.cuda.is_available():
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros_like(cls_loss).cuda())
        else:
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros_like(cls_loss))
        return cls_loss.sum()


@LOSSES.register_module()
class smooth_l1(object):

    def __call__(self, pos_inds, anchor_infos, boxes, reg_pred):
        """
        Args:
            pos_inds [sum(H*W*A),]:  计算Anchor与真值框的IoU，根据阈值(0.4,0.5)筛选出正负样本(Anchor)
            anchor_infos: [anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y]
            boxes [sum(H*W)*A, 4]:  每个anchor对应的一个与其IoU最大的真值框
            reg_pred [sum(W*H*A), 4]:  回归分支预测输出，[dx,dy,dw,dh]
        """

        anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y = anchor_infos  # (sum(H*W)*A,)
        if pos_inds.sum() > 0:

            pos_reg_pred = reg_pred[pos_inds, :]  # (num_pos, 4)

            gt_widths = boxes[pos_inds][:, 2] - boxes[pos_inds][:, 0]
            gt_heights = boxes[pos_inds][:, 3] - boxes[pos_inds][:, 1]
            gt_ctr_x = boxes[pos_inds][:, 0] + gt_widths * 0.5
            gt_ctr_y = boxes[pos_inds][:, 1] + gt_heights * 0.5

            pos_anchor_widths = anchor_widths[pos_inds]
            pos_anchor_heights = anchor_heights[pos_inds]
            pos_anchor_ctr_x = anchor_ctr_x[pos_inds]
            pos_anchor_ctr_y = anchor_ctr_y[pos_inds]

            gt_widths = torch.clamp(gt_widths, min=1.0)
            gt_heights = torch.clamp(gt_heights, min=1.0)

            target_dx = (gt_ctr_x - pos_anchor_ctr_x) / pos_anchor_widths
            target_dy = (gt_ctr_y - pos_anchor_ctr_y) / pos_anchor_heights
            target_dw = torch.log(gt_widths / pos_anchor_widths)
            target_dh = torch.log(gt_heights / pos_anchor_heights)

            targets = torch.stack([target_dx, target_dy, target_dw, target_dh], dim=0).t()  # (num_pos,4)
            if torch.cuda.is_available():
                targets = targets / torch.FloatTensor([0.1, 0.1, 0.2, 0.2]).cuda()
            else:
                targets = targets / torch.FloatTensor([0.1, 0.1, 0.2, 0.2])

            reg_diff = torch.abs(targets - pos_reg_pred)  # (num_pos,4)  torch.abs() 取绝对值
            reg_loss = torch.where(
                torch.le(reg_diff, 1.0 / 9.0),
                0.5 * 9.0 * torch.pow(reg_diff, 2),
                reg_diff - 0.5 / 9.0
            )
            return reg_loss.mean()
        else:
            if torch.cuda.is_available():
                reg_loss = torch.tensor(0).float().cuda()
            else:
                reg_loss = torch.tensor(0).float()

            return reg_loss


@LOSSES.register_module()
class GIoU(object):
    def __call__(self, pos_inds, anchor_infos, boxes, reg_pred):
        """
        Args:
            pos_inds [sum(H*W*A),]:  计算Anchor与真值框的IoU，根据阈值(0.4,0.5)筛选出正负样本(Anchor)
            anchor_infos: [anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y]
            boxes [sum(H*W)*A, 4]:  每个anchor对应的一个与其IoU最大的真值框
            reg_pred [sum(W*H*A), 4]:  回归分支预测输出，[dx,dy,dw,dh]
        Returns:
            GIoU Loss for one image.
        """
        anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y = anchor_infos  # (sum(H*W)*A,)
        if pos_inds.sum() > 0:

            pos_reg_pred = reg_pred[pos_inds, :]  # (num_pos, 4)

            gt_boxes = boxes[pos_inds, :]  # (num_pos, 4)

            pos_anchor_widths = anchor_widths[pos_inds]  # (num_pos,)
            pos_anchor_heights = anchor_heights[pos_inds]  # (num_pos,)
            pos_anchor_ctr_x = anchor_ctr_x[pos_inds]  # (num_pos,)
            pos_anchor_ctr_y = anchor_ctr_y[pos_inds]  # (num_pos,)

            dx = pos_reg_pred[:, 0] * 0.1  # (num_pos,)
            dy = pos_reg_pred[:, 1] * 0.1  # (num_pos,)
            dw = pos_reg_pred[:, 2] * 0.2  # (num_pos,)
            dh = pos_reg_pred[:, 3] * 0.2  # (num_pos,)

            pred_ctr_x = dx * pos_anchor_widths + pos_anchor_ctr_x  # (num_pos,)
            pred_ctr_y = dy * pos_anchor_heights + pos_anchor_ctr_y  # (num_pos,)
            pred_w = torch.exp(dw) * pos_anchor_widths  # (num_pos,)
            pred_h = torch.exp(dh) * pos_anchor_heights  # (num_pos,)

            pred_x1 = pred_ctr_x - pred_w * 0.5  # (num_pos,)
            pred_y1 = pred_ctr_y - pred_h * 0.5  # (num_pos,)
            pred_x2 = pred_ctr_x + pred_w * 0.5  # (num_pos,)
            pred_y2 = pred_ctr_y + pred_h * 0.5  # (num_pos,)

            preds_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=0).t()  # (num_pos,4)
            reg_loss = self.compute_giou_loss(gt_boxes, preds_boxes)
        else:
            if torch.cuda.is_available():
                reg_loss = torch.tensor(0).float().cuda()
            else:
                reg_loss = torch.tensor(0).float()

        return reg_loss

    @staticmethod
    def compute_giou_loss(boxes1, boxes2):
        """
        boxes1 :(N,4)  (x1,y1,x2,y2)
        boxes2: (N,4)  (x1,y1,x2,y2)
        """
        x1y1 = torch.max(boxes1[:, :2], boxes2[:, :2])
        x2y2 = torch.min(boxes1[:, 2:], boxes2[:, 2:])
        wh = torch.clamp(x2y2 - x1y1, min=0.)
        area_inter = wh[:, 0] * wh[:, 1]
        area_b1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area_b2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area_b1 + area_b2 - area_inter
        iou = area_inter / (union + 1e-16)

        x1y1_max = torch.min(boxes1[:, :2], boxes2[:, :2])
        x2y2_max = torch.max(boxes1[:, 2:], boxes2[:, 2:])
        g_wh = torch.clamp(x2y2_max - x1y1_max, min=0.)
        g_area = g_wh[:, 0] * g_wh[:, 1]

        giou = iou - (g_area - union) / g_area.clamp(1e-10)
        loss = 1. - giou
        return loss.mean()


@LOSSES.register_module()
class DIoU(object):
    def __call__(self, pos_inds, anchor_infos, boxes, reg_pred):
        """
        Args:
            pos_inds [sum(H*W*A),]:  计算Anchor与真值框的IoU，根据阈值(0.4,0.5)筛选出正负样本(Anchor)
            anchor_infos: [anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y]
            boxes [sum(H*W)*A, 4]:  每个anchor对应的一个与其IoU最大的真值框
            reg_pred [sum(W*H*A), 4]:  回归分支预测输出，[dx,dy,dw,dh]
        Returns:
            DIoU Loss for one image.
        """
        anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y = anchor_infos
        if pos_inds.sum() > 0:
            pos_reg_pred = reg_pred[pos_inds, :]  # (num_pos, 4)

            gt_boxes = boxes[pos_inds, :]  # (num_pos, 4)

            pos_anchor_widths = anchor_widths[pos_inds]  # (num_pos,)
            pos_anchor_heights = anchor_heights[pos_inds]  # (num_pos,)
            pos_anchor_ctr_x = anchor_ctr_x[pos_inds]  # (num_pos,)
            pos_anchor_ctr_y = anchor_ctr_y[pos_inds]  # (num_pos,)

            dx = pos_reg_pred[:, 0] * 0.1  # (num_pos,)
            dy = pos_reg_pred[:, 1] * 0.1  # (num_pos,)
            dw = pos_reg_pred[:, 2] * 0.2  # (num_pos,)
            dh = pos_reg_pred[:, 3] * 0.2  # (num_pos,)

            pred_ctr_x = dx * pos_anchor_widths + pos_anchor_ctr_x  # (num_pos,)
            pred_ctr_y = dy * pos_anchor_heights + pos_anchor_ctr_y  # (num_pos,)
            pred_w = torch.exp(dw) * pos_anchor_widths  # (num_pos,)
            pred_h = torch.exp(dh) * pos_anchor_heights  # (num_pos,)

            pred_x1 = pred_ctr_x - pred_w * 0.5  # (num_pos,)
            pred_y1 = pred_ctr_y - pred_h * 0.5  # (num_pos,)
            pred_x2 = pred_ctr_x + pred_w * 0.5  # (num_pos,)
            pred_y2 = pred_ctr_y + pred_h * 0.5  # (num_pos,)

            preds_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=0).t()  # (num_pos,4)
            reg_loss = self.compute_diou_loss(gt_boxes, preds_boxes)
        else:
            if torch.cuda.is_available():
                reg_loss = torch.tensor(0).float().cuda()
            else:
                reg_loss = torch.tensor(0).float()

        return reg_loss

    @staticmethod
    def compute_diou_loss(boxes1, boxes2):
        """
        boxes1 :(N,4)  (x1,y1,x2,y2)
        boxes2: (N,4)  (x1,y1,x2,y2)
        """
        x1y1 = torch.max(boxes1[:, :2], boxes2[:, :2])
        x2y2 = torch.min(boxes1[:, 2:], boxes2[:, 2:])

        top_left = torch.min(boxes1[:, :2], boxes2[:, :2])
        bottom_right = torch.max(boxes1[:, 2:], boxes2[:, 2:])

        wh = torch.clamp(x2y2 - x1y1, min=0.)
        area_inter = wh[:, 0] * wh[:, 1]
        area_b1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area_b2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area_b1 + area_b2 - area_inter
        iou = area_inter / (union + 1e-16)

        # ctrx1, ctry1 = boxes1[:, 0] + (boxes1[:, 2] - boxes1[:, 0])/2, boxes1[:, 1] + (boxes1[:, 3] - boxes1[:, 1])/2
        # ctrx2, ctry2 = boxes2[:, 0] + (boxes2[:, 2] - boxes2[:, 0])/2, boxes2[:, 1] + (boxes2[:, 3] - boxes2[:, 1])/2

        # [N, 1]
        distance = torch.add(
            torch.pow(torch.div(torch.sum((boxes2[:, [0, 2]] - boxes1[:, [0, 2]]), dim=1, keepdim=True), 2), 2),
            torch.pow(torch.div(torch.sum((boxes2[:, [1, 3]] - boxes1[:, [1, 3]]), dim=1, keepdim=True), 2), 2))

        loss = 1 - iou + distance / torch.sum(torch.pow((bottom_right - top_left), 2), 1)

        return loss.mean()


@LOSSES.register_module()
class CIoU(object):
    def __call__(self, pos_inds, anchor_infos, boxes, reg_pred):
        """
        Args:
            pos_inds [sum(H*W*A),]:  计算Anchor与真值框的IoU，根据阈值(0.4,0.5)筛选出正负样本(Anchor)
            anchor_infos: [anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y]
            boxes [sum(H*W)*A, 4]:  每个anchor对应的一个与其IoU最大的真值框
            reg_pred [sum(W*H*A), 4]:  回归分支预测输出，[dx,dy,dw,dh]
        Returns:
            CIoU Loss for one image.
        """
        anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y = anchor_infos
        if pos_inds.sum() > 0:
            pos_reg_pred = reg_pred[pos_inds, :]  # (num_pos, 4)

            gt_boxes = boxes[pos_inds, :]  # (num_pos, 4)

            pos_anchor_widths = anchor_widths[pos_inds]  # (num_pos,)
            pos_anchor_heights = anchor_heights[pos_inds]  # (num_pos,)
            pos_anchor_ctr_x = anchor_ctr_x[pos_inds]  # (num_pos,)
            pos_anchor_ctr_y = anchor_ctr_y[pos_inds]  # (num_pos,)

            dx = pos_reg_pred[:, 0] * 0.1  # (num_pos,)
            dy = pos_reg_pred[:, 1] * 0.1  # (num_pos,)
            dw = pos_reg_pred[:, 2] * 0.2  # (num_pos,)
            dh = pos_reg_pred[:, 3] * 0.2  # (num_pos,)

            pred_ctr_x = dx * pos_anchor_widths + pos_anchor_ctr_x  # (num_pos,)
            pred_ctr_y = dy * pos_anchor_heights + pos_anchor_ctr_y  # (num_pos,)
            pred_w = torch.exp(dw) * pos_anchor_widths  # (num_pos,)
            pred_h = torch.exp(dh) * pos_anchor_heights  # (num_pos,)

            pred_x1 = pred_ctr_x - pred_w * 0.5  # (num_pos,)
            pred_y1 = pred_ctr_y - pred_h * 0.5  # (num_pos,)
            pred_x2 = pred_ctr_x + pred_w * 0.5  # (num_pos,)
            pred_y2 = pred_ctr_y + pred_h * 0.5  # (num_pos,)

            preds_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=0).t()  # (num_pos,4)
            reg_loss = self.compute_ciou_loss(gt_boxes, preds_boxes)
        else:
            if torch.cuda.is_available():
                reg_loss = torch.tensor(0).float().cuda()
            else:
                reg_loss = torch.tensor(0).float()

        return reg_loss

    @staticmethod
    def compute_ciou_loss(boxes1, boxes2):
        """
        boxes1 :(N,4)  (x1,y1,x2,y2)
        boxes2: (N,4)  (x1,y1,x2,y2)
        """
        x1y1 = torch.max(boxes1[:, :2], boxes2[:, :2])
        x2y2 = torch.min(boxes1[:, 2:], boxes2[:, 2:])

        top_left = torch.min(boxes1[:, :2], boxes2[:, :2])
        bottom_right = torch.max(boxes1[:, 2:], boxes2[:, 2:])

        wh = torch.clamp(x2y2 - x1y1, min=0.)
        area_inter = wh[:, 0] * wh[:, 1]
        area_b1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area_b2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area_b1 + area_b2 - area_inter
        iou = area_inter / (union + 1e-16)

        # ctrx1, ctry1 = boxes1[:, 0] + (boxes1[:, 2] - boxes1[:, 0])/2, boxes1[:, 1] + (boxes1[:, 3] - boxes1[:, 1])/2
        # ctrx2, ctry2 = boxes2[:, 0] + (boxes2[:, 2] - boxes2[:, 0])/2, boxes2[:, 1] + (boxes2[:, 3] - boxes2[:, 1])/2

        # [N, 1]
        distance = torch.add(
            torch.pow(torch.div(torch.sum((boxes2[:, [0, 2]] - boxes1[:, [0, 2]]), dim=1, keepdim=True), 2), 2),
            torch.pow(torch.div(torch.sum((boxes2[:, [1, 3]] - boxes1[:, [1, 3]]), dim=1, keepdim=True), 2), 2))

        v = 4 / torch.pow(torch.tensor(np.pi), 2) * torch.pow(
            torch.arctan((boxes1[:, 2] - boxes1[:, 0]) / (boxes1[:, 3] - boxes1[:, 1])) -
            torch.arctan((boxes2[:, 2] - boxes2[:, 0]) / (boxes2[:, 3] - boxes2[:, 1])), 2)
        alpha = v/(1-iou+v)
        loss = 1 - iou + distance / torch.sum(torch.pow((bottom_right - top_left), 2), 1) + alpha*v
        return loss.mean()


@LOSSES.register_module()
class CrossEntropyLoss(object):
    def __init__(self, reduction=None):
        self.reduction = reduction

    def __call__(self, a, b):
        loss = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        return loss(a, b)


@LOSSES.register_module()
class SmoothL1Loss(object):
    def __init__(self, reduction=None):
        self.reduction = reduction

    def __call__(self, a, b):
        loss = torch.nn.SmoothL1Loss(reduction=self.reduction)
        return loss(a, b)


@LOSSES.register_module()
class BCELoss(object):
    def __init__(self, reduction=None):
        self.reduction = reduction

    def __call__(self, a, b):
        loss = torch.nn.BCELoss(reduction=self.reduction)
        return loss(a, b)


class ComputeLoss:
    def __init__(self, loss_cfg, anchor_cfg):
        self.loss = build_loss(loss_cfg)
        # self.generater = build_generator(anchor_cfg)

    def __call__(self, preds, annots, anchors):
        return self.loss([preds, annots, anchors])
