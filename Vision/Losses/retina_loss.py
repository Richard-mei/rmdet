from ..utils import build_loss, LOSSES
from ..utils import box_iou
import torch.nn as nn
import torch
from Vision.utils import multi_apply


@LOSSES.register_module()
class retina_loss(nn.Module):
    def __init__(self, cls_loss, reg_loss, **kwargs):
        super(retina_loss, self).__init__()
        self.cls_loss = build_loss(cls_loss)
        self.reg_loss = build_loss(reg_loss)

    def forward(self, inputs):
        """
        cls_logits :(n, sum(H*W)*A, class_num)
        reg_preds:(n, sum(H*W)*A, 4)
        anchors:(sum(H*W)*A, 4)
        boxes:(n, max_num, 4)
        classes:(n, max_num)
        """
        preds, annots, anchors = inputs
        # cls_logits, reg_preds = preds[..., 0:-4], preds[..., -4:]
        cls_logits, reg_preds = preds['cls_score'], preds['reg_pred']
        boxes = annots[..., :4]
        classes = annots[..., -1]

        bacth_size = cls_logits.shape[0]
        cls_logit_list = [cls_logits[i, :, :] for i in range(bacth_size)]
        reg_pred_list = [reg_preds[i, :, :] for i in range(bacth_size)]
        anchors_list = [anchors[:, :] for _ in range(bacth_size)]  # [x1, y1, x2, y2]
        gt_box_list = [boxes[i, :, :] for i in range(bacth_size)]
        gt_classes_list = [classes[i, :] for i in range(bacth_size)]

        targets_list, pos_anchors_ind_list, assigned_boxes_list, assigned_classes_list, num_pos_list = multi_apply(
            self.get_targets,
            anchors_list,
            gt_box_list,
            gt_classes_list,
            cls_logit_list)
        cls_loss_list, reg_loss_list = multi_apply(
            self.loss_single,
            cls_logit_list,
            targets_list,
            num_pos_list,
            pos_anchors_ind_list,
            assigned_boxes_list,
            reg_pred_list,
            anchors_list)

        cls_loss = torch.stack(cls_loss_list).mean()
        reg_loss = torch.stack(reg_loss_list).mean()
        total_loss = cls_loss + reg_loss
        return {'cls_loss': cls_loss, 'reg_loss': reg_loss, 'total_loss': total_loss}

        # for i in range(bacth_size):
        #     per_cls_logit = cls_logits[i, :, :]  # (sum(H*W)*A, class_num)
        #     per_reg_pred = reg_preds[i, :, :]
        #     per_boxes = boxes[i, :, :]
        #     per_classes = classes[i, :]
        #
        #     mask = per_boxes[:, 0] != -1
        #     per_boxes = per_boxes[mask]  # (?, 4)
        #     per_classes = per_classes[mask]  # (?,)
        #
        #     if per_classes.shape[0] == 0:
        #         alpha_factor = torch.ones(
        #             per_cls_logit.shape).cuda() * 0.25 if torch.cuda.is_available() else torch.ones(
        #             per_cls_logit.shape) * 0.25
        #         alpha_factor = 1. - alpha_factor
        #         focal_weights = per_cls_logit
        #         focal_weights = alpha_factor * torch.pow(focal_weights, 2.0)
        #         bce = -(torch.log(1.0 - per_cls_logit))
        #         cls_loss = focal_weights * bce
        #         class_loss.append(cls_loss.sum())
        #         reg_loss.append(torch.tensor(0).float())
        #         continue
        #     targets, pos_anchors_ind, assigned_boxes, assigned_classes, num_pos = self.get_targets(anchors, per_boxes,
        #                                                                                            per_classes,
        #                                                                                            per_cls_logit)
        #
        #     class_loss.append(self.cls_loss(per_cls_logit, targets).view(1) / num_pos)
        #     reg_loss.append(self.reg_loss(pos_anchors_ind, [anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y],
        #                                   assigned_boxes, per_reg_pred))
        # cls_loss = torch.stack(class_loss).mean()
        # reg_loss = torch.stack(reg_loss).mean()
        # total_loss = cls_loss + reg_loss
        # return cls_loss, reg_loss, total_loss

        # for i in range(bacth_size):
        #     per_cls_logit = cls_logits[i, :, :]  # (sum(H*W)*A, class_num)
        #     per_reg_pred = reg_preds[i, :, :]
        #     print(per_reg_pred)
        #     per_boxes = boxes[i, :, :]
        #     per_classes = classes[i, :]
        #     mask = per_boxes[:, 0] != -1
        #     per_boxes = per_boxes[mask]  # (?, 4)
        #     per_classes = per_classes[mask]  # (?,)
        #     if per_classes.shape[0] == 0:
        #         alpha_factor = torch.ones(
        #             per_cls_logit.shape).cuda() * 0.25 if torch.cuda.is_available() else torch.ones(
        #             per_cls_logit.shape) * 0.25
        #         alpha_factor = 1. - alpha_factor
        #         focal_weights = per_cls_logit
        #         focal_weights = alpha_factor * torch.pow(focal_weights, 2.0)
        #         bce = -(torch.log(1.0 - per_cls_logit))
        #         cls_loss = focal_weights * bce
        #         class_loss.append(cls_loss.sum())
        #         reg_loss.append(torch.tensor(0).float())
        #         continue
        #     IoU = box_iou(anchors, per_boxes)  # (sum(H*W)*A, ?)
        #
        #     iou_max, max_ind = torch.max(IoU, dim=1)  # (sum(H*W)*A,)
        #
        #     targets = torch.ones_like(per_cls_logit) * -1  # (sum(H*W)*A, class_num)
        #
        #     targets[iou_max < 0.4, :] = 0  # bg
        #
        #     pos_anchors_ind = iou_max >= 0.5  # (?,)
        #     num_pos = torch.clamp(pos_anchors_ind.sum().float(), min=1.0)
        #
        #     assigned_classes = per_classes[max_ind]  # (sum(H*W)*A, )
        #     assigned_boxes = per_boxes[max_ind, :]  # (sum(H*W)*A, 4)
        #
        #     targets[pos_anchors_ind, :] = 0
        #     targets[pos_anchors_ind, (assigned_classes[pos_anchors_ind]).long() - 1] = 1
        #
        #     class_loss.append(self.cls_loss(per_cls_logit, targets).view(1) / num_pos)
        #     reg_loss.append(self.reg_loss(pos_anchors_ind, [anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y],
        #                                   assigned_boxes, per_reg_pred))
        # cls_loss = torch.stack(class_loss).mean()
        # reg_loss = torch.stack(reg_loss).mean()
        # total_loss = cls_loss + reg_loss
        # return cls_loss, reg_loss, total_loss

    @staticmethod
    def get_targets(anchors, gt_boxes, gt_classes, cls_logit):
        """
        # ?????????batch???????????????????????????level?????????????????????loss <== loss_single
        ????????????Anchor???????????????IoU?????????0.4??????????????????0.5???????????????Anchor???IoU??????????????????????????????????????????Anchor???????????????
        Args:
            anchors: All anchors for one image.
            gt_boxes: Ground truth boxes for one image.
            cls_logit: Classes predictions for one image.
            gt_classes: Ground truth classes for one image.

        Returns:
            targets: ??????IoU???????????????????????????1????????????(IoU>=0.5????????????0ne-hot??????)???0????????????IoU<0.4???????????????-1?????????
            pos_anchors_ind: ????????????IoU>=0.5????????????anchor??????
            assigned_boxes: ????????????Anchor????????????IoU?????????????????????????????????(???????????????)
            assigned_classes: ????????????Anchor????????????IoU????????????????????????????????????????????????(??????????????????)
            num_pos: ????????????IoU>=0.5???Anchor??????//????????????
        """
        mask1 = gt_boxes[:, 0] == -1
        mask = ~mask1
        # mask = gt_boxes[:, 0] not in [-1, 0]
        gt_boxes = gt_boxes[mask]
        gt_classes = gt_classes[mask]
        IoU = box_iou(anchors, gt_boxes)  # [num_anchors, num_gt_boxes]

        iou_max, max_ind = torch.max(IoU, dim=1)  # (sum(H*W)*A,)  ?????????anchor??????IoU????????????????????????
        targets = torch.ones_like(cls_logit) * -1  # (sum(H*W)*A, class_num)

        targets[iou_max < 0.4, :] = 0  # bg

        pos_anchors_ind = iou_max >= 0.5  # (?,)
        num_pos = torch.clamp(pos_anchors_ind.sum().float(), min=1.0)

        assigned_classes = gt_classes[max_ind]  # (sum(H*W)*A, )  # ???????????? 1-20  ??????anchor????????????????????????
        assigned_boxes = gt_boxes[max_ind, :]  # (sum(H*W)*A, 4)  # ??????anchor??????????????????IoU??????????????????

        targets[pos_anchors_ind, :] = 0
        targets[pos_anchors_ind, (assigned_classes[pos_anchors_ind]).long() - 1] = 1

        return targets, pos_anchors_ind, assigned_boxes, assigned_classes, num_pos

    def loss_single(self, cls_logit, targets, num_pos, pos_anchor_ind, assigned_boxes, per_reg_pred, anchors):
        """
        Args:
            cls_logit: ????????????????????????????????? [sum(W*H*A), num_classes]
            targets: ????????????????????????????????????One-hot?????? [sum(W*H*A), num_classes]  ?????????-1???0???1
            num_pos:  ???????????????????????????(Anchor)???
            pos_anchor_ind:  ?????????indices [sum(W*H*A),] ?????? True, False
            assigned_boxes:  ??????Anchor???????????????IoU??????????????????
            per_reg_pred:  ?????????????????????????????????  [sum(W*H*A), 4]
            anchors:   ?????????Anchor  [sum(W*H*A), 4]  [x1, y1, x2, y2]

        Returns:
            cls_loss: ???????????? FocalLoss
            reg_loss: ???????????? SmoothL1 // GIoU // DIoU // CIoU
        """
        anchor_widths = anchors[:, 2] - anchors[:, 0]
        anchor_heights = anchors[:, 3] - anchors[:, 1]
        anchor_ctr_x = anchors[:, 0] + anchor_widths * 0.5
        anchor_ctr_y = anchors[:, 1] + anchor_heights * 0.5

        cls_loss = self.cls_loss(cls_logit, targets).view(1) / num_pos
        reg_loss = self.reg_loss(
            pos_anchor_ind,
            [anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y],
            assigned_boxes,
            per_reg_pred
        )
        return cls_loss, reg_loss


@LOSSES.register_module()
class gs_retina_loss(nn.Module):
    def __init__(self, cls_loss, reg_loss, **kwargs):
        super(gs_retina_loss, self).__init__()
        self.cls_loss = build_loss(cls_loss)
        self.reg_loss = build_loss(reg_loss)

    def forward(self, inputs):
        """
        cls_logits :(n, sum(H*W)*A, class_num)
        reg_preds:(n, sum(H*W)*A, 4)
        anchors:(sum(H*W)*A, 4)
        boxes:(n, max_num, 4)
        classes:(n, max_num)
        """
        preds, annots, anchors = inputs
        # cls_logits, reg_preds = preds[..., 0:-4], preds[..., -4:]
        cls_logits, reg_preds, gs_mask = preds['cls_score'], preds['reg_pred'], preds['gs_mask']
        boxes = annots[..., :4]
        classes = annots[..., -1]

        bacth_size = cls_logits.shape[0]
        cls_logit_list = [cls_logits[i, gs_mask[i, :, 0] != -1, :] for i in range(bacth_size)]
        reg_pred_list = [reg_preds[i, gs_mask[i, :, 0] != -1, :] for i in range(bacth_size)]
        anchors_list = [anchors[gs_mask[i, :, 0] != -1, :] for i in range(bacth_size)]  # [x1, y1, x2, y2]
        gt_box_list = [boxes[i, :, :] for i in range(bacth_size)]
        gt_classes_list = [classes[i, :] for i in range(bacth_size)]

        targets_list, pos_anchors_ind_list, assigned_boxes_list, assigned_classes_list, num_pos_list = multi_apply(
            self.get_targets,
            anchors_list,
            gt_box_list,
            gt_classes_list,
            cls_logit_list)
        cls_loss_list, reg_loss_list = multi_apply(
            self.loss_single,
            cls_logit_list,
            targets_list,
            num_pos_list,
            pos_anchors_ind_list,
            assigned_boxes_list,
            reg_pred_list,
            anchors_list)

        cls_loss = torch.stack(cls_loss_list).mean()
        reg_loss = torch.stack(reg_loss_list).mean()
        total_loss = cls_loss + reg_loss
        return {'cls_loss': cls_loss, 'reg_loss': reg_loss, 'total_loss': total_loss}

        # for i in range(bacth_size):
        #     per_cls_logit = cls_logits[i, :, :]  # (sum(H*W)*A, class_num)
        #     per_reg_pred = reg_preds[i, :, :]
        #     per_boxes = boxes[i, :, :]
        #     per_classes = classes[i, :]
        #
        #     mask = per_boxes[:, 0] != -1
        #     per_boxes = per_boxes[mask]  # (?, 4)
        #     per_classes = per_classes[mask]  # (?,)
        #
        #     if per_classes.shape[0] == 0:
        #         alpha_factor = torch.ones(
        #             per_cls_logit.shape).cuda() * 0.25 if torch.cuda.is_available() else torch.ones(
        #             per_cls_logit.shape) * 0.25
        #         alpha_factor = 1. - alpha_factor
        #         focal_weights = per_cls_logit
        #         focal_weights = alpha_factor * torch.pow(focal_weights, 2.0)
        #         bce = -(torch.log(1.0 - per_cls_logit))
        #         cls_loss = focal_weights * bce
        #         class_loss.append(cls_loss.sum())
        #         reg_loss.append(torch.tensor(0).float())
        #         continue
        #     targets, pos_anchors_ind, assigned_boxes, assigned_classes, num_pos = self.get_targets(anchors, per_boxes,
        #                                                                                            per_classes,
        #                                                                                            per_cls_logit)
        #
        #     class_loss.append(self.cls_loss(per_cls_logit, targets).view(1) / num_pos)
        #     reg_loss.append(self.reg_loss(pos_anchors_ind, [anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y],
        #                                   assigned_boxes, per_reg_pred))
        # cls_loss = torch.stack(class_loss).mean()
        # reg_loss = torch.stack(reg_loss).mean()
        # total_loss = cls_loss + reg_loss
        # return cls_loss, reg_loss, total_loss

        # for i in range(bacth_size):
        #     per_cls_logit = cls_logits[i, :, :]  # (sum(H*W)*A, class_num)
        #     per_reg_pred = reg_preds[i, :, :]
        #     print(per_reg_pred)
        #     per_boxes = boxes[i, :, :]
        #     per_classes = classes[i, :]
        #     mask = per_boxes[:, 0] != -1
        #     per_boxes = per_boxes[mask]  # (?, 4)
        #     per_classes = per_classes[mask]  # (?,)
        #     if per_classes.shape[0] == 0:
        #         alpha_factor = torch.ones(
        #             per_cls_logit.shape).cuda() * 0.25 if torch.cuda.is_available() else torch.ones(
        #             per_cls_logit.shape) * 0.25
        #         alpha_factor = 1. - alpha_factor
        #         focal_weights = per_cls_logit
        #         focal_weights = alpha_factor * torch.pow(focal_weights, 2.0)
        #         bce = -(torch.log(1.0 - per_cls_logit))
        #         cls_loss = focal_weights * bce
        #         class_loss.append(cls_loss.sum())
        #         reg_loss.append(torch.tensor(0).float())
        #         continue
        #     IoU = box_iou(anchors, per_boxes)  # (sum(H*W)*A, ?)
        #
        #     iou_max, max_ind = torch.max(IoU, dim=1)  # (sum(H*W)*A,)
        #
        #     targets = torch.ones_like(per_cls_logit) * -1  # (sum(H*W)*A, class_num)
        #
        #     targets[iou_max < 0.4, :] = 0  # bg
        #
        #     pos_anchors_ind = iou_max >= 0.5  # (?,)
        #     num_pos = torch.clamp(pos_anchors_ind.sum().float(), min=1.0)
        #
        #     assigned_classes = per_classes[max_ind]  # (sum(H*W)*A, )
        #     assigned_boxes = per_boxes[max_ind, :]  # (sum(H*W)*A, 4)
        #
        #     targets[pos_anchors_ind, :] = 0
        #     targets[pos_anchors_ind, (assigned_classes[pos_anchors_ind]).long() - 1] = 1
        #
        #     class_loss.append(self.cls_loss(per_cls_logit, targets).view(1) / num_pos)
        #     reg_loss.append(self.reg_loss(pos_anchors_ind, [anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y],
        #                                   assigned_boxes, per_reg_pred))
        # cls_loss = torch.stack(class_loss).mean()
        # reg_loss = torch.stack(reg_loss).mean()
        # total_loss = cls_loss + reg_loss
        # return cls_loss, reg_loss, total_loss

    @staticmethod
    def get_targets(anchors, gt_boxes, gt_classes, cls_logit):
        """
        # ?????????batch???????????????????????????level?????????????????????loss <== loss_single
        ????????????Anchor???????????????IoU?????????0.4??????????????????0.5???????????????Anchor???IoU??????????????????????????????????????????Anchor???????????????
        Args:
            anchors: All anchors for one image.
            gt_boxes: Ground truth boxes for one image.
            cls_logit: Classes predictions for one image.
            gt_classes: Ground truth classes for one image.

        Returns:
            targets: ??????IoU???????????????????????????1????????????(IoU>=0.5????????????0ne-hot??????)???0????????????IoU<0.4???????????????-1?????????
            pos_anchors_ind: ????????????IoU>=0.5????????????anchor??????
            assigned_boxes: ????????????Anchor????????????IoU?????????????????????????????????(???????????????)
            assigned_classes: ????????????Anchor????????????IoU????????????????????????????????????????????????(??????????????????)
            num_pos: ????????????IoU>=0.5???Anchor??????//????????????
        """
        mask1 = gt_boxes[:, 0] == -1
        mask = ~mask1
        # mask = gt_boxes[:, 0] not in [-1, 0]
        gt_boxes = gt_boxes[mask]
        gt_classes = gt_classes[mask]
        IoU = box_iou(anchors, gt_boxes)  # [num_anchors, num_gt_boxes]

        iou_max, max_ind = torch.max(IoU, dim=1)  # (sum(H*W)*A,)  ?????????anchor??????IoU????????????????????????
        targets = torch.ones_like(cls_logit) * -1  # (sum(H*W)*A, class_num)

        targets[iou_max < 0.4, :] = 0  # bg

        pos_anchors_ind = iou_max >= 0.5  # (?,)
        num_pos = torch.clamp(pos_anchors_ind.sum().float(), min=1.0)

        assigned_classes = gt_classes[max_ind]  # (sum(H*W)*A, )  # ???????????? 1-20  ??????anchor????????????????????????
        assigned_boxes = gt_boxes[max_ind, :]  # (sum(H*W)*A, 4)  # ??????anchor??????????????????IoU??????????????????

        targets[pos_anchors_ind, :] = 0
        targets[pos_anchors_ind, (assigned_classes[pos_anchors_ind]).long() - 1] = 1

        return targets, pos_anchors_ind, assigned_boxes, assigned_classes, num_pos

    def loss_single(self, cls_logit, targets, num_pos, pos_anchor_ind, assigned_boxes, per_reg_pred, anchors):
        """
        Args:
            cls_logit: ????????????????????????????????? [sum(W*H*A), num_classes]
            targets: ????????????????????????????????????One-hot?????? [sum(W*H*A), num_classes]  ?????????-1???0???1
            num_pos:  ???????????????????????????(Anchor)???
            pos_anchor_ind:  ?????????indices [sum(W*H*A),] ?????? True, False
            assigned_boxes:  ??????Anchor???????????????IoU??????????????????
            per_reg_pred:  ?????????????????????????????????  [sum(W*H*A), 4]
            anchors:   ?????????Anchor  [sum(W*H*A), 4]  [x1, y1, x2, y2]

        Returns:
            cls_loss: ???????????? FocalLoss
            reg_loss: ???????????? SmoothL1 // GIoU // DIoU // CIoU
        """
        anchor_widths = anchors[:, 2] - anchors[:, 0]
        anchor_heights = anchors[:, 3] - anchors[:, 1]
        anchor_ctr_x = anchors[:, 0] + anchor_widths * 0.5
        anchor_ctr_y = anchors[:, 1] + anchor_heights * 0.5

        cls_loss = self.cls_loss(cls_logit, targets).view(1) / num_pos
        reg_loss = self.reg_loss(
            pos_anchor_ind,
            [anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y],
            assigned_boxes,
            per_reg_pred
        )
        return cls_loss, reg_loss
