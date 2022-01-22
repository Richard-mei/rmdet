import torch
from collections import Counter
from tqdm import tqdm
from Vision.utils import ConfusionMatrix, retina_nms, ap_per_class, process_batch, time_sync, xyxy2xywh, xywh2xyxy
import numpy as np
import torch.nn as nn
import cv2
import pandas as pd


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


def mean_average_precision(pred_bboxes, true_boxes, iou_threshold, classes):
    # pred_bboxes(list): [[train_idx,class_pred,prob_score,x1,y1,x2,y2], ...]
    # true_boxes(tensor) (num_boxes, 6(train_idx,x,y,w,h,class))
    # pred_boxes(tensor) (num_preds, 7(train_idx,x,y,w,h,prob_score,class_pred))

    gt_boxes = xywh2xyxy(true_boxes[..., 1:5])
    pred_box = xywh2xyxy(pred_bboxes[..., 1:5])
    true_boxes[..., 1:5] = gt_boxes
    pred_bboxes[..., 1:5] = pred_box
    average_precisions = []  # 存储每一个类别的AP
    all_ap = []
    epsilon = 1e-6  # 防止分母为0

    # 对于每一个类别
    for c in range(len(classes)):
        detections = []  # 存储预测为该类别的bbox
        ground_truths = []  # 存储本身就是该类别的bbox(GT)

        for detection in pred_bboxes:
            if detection[6] == c:
                detections.append(detection)
        for true_box in true_boxes:
            if true_box[5] == c:
                ground_truths.append(true_box)

        amount_bboxes = Counter(int(gt[0]) for gt in ground_truths)

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)  # 置0，表示这些真实框初始时都没有与任何预测框匹配
        # 此时，amount_bboxes={0:torch.tensor([0,0,0]),1:torch.tensor([0,0,0,0,0])}

        # 将预测框按照置信度从大到小排序
        detections.sort(key=lambda x: x[5], reverse=True)

        # 初始化TP,FP
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))

        # TP+FN就是当前类别GT框的总数，是固定的
        total_true_bboxes = len(ground_truths)

        # 如果当前类别一个GT框都没有，那么直接跳过即可
        if total_true_bboxes == 0:
            continue

        # 对于每个预测框，先找到它所在图片中的所有真实框，然后计算预测框与每一个真实框之间的IoU，大于IoU阈值且该真实框没有与其他预测框匹配，则置该预测框的预测结果为TP，否则为FP
        for detection_idx, detection in enumerate(detections):
            # 在计算IoU时，只能是同一张图片内的框做，不同图片之间不能做
            # 图片的编号存在第0个维度
            # 于是下面这句代码的作用是：找到当前预测框detection所在图片中的所有真实框，用于计算IoU
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            num_gts = len(ground_truth_img)

            best_iou = 0
            best_gt_idx = 0
            for idx, gt in enumerate(ground_truth_img):
                # 计算当前预测框detection与它所在图片内的每一个真实框的IoU
                iou = insert_over_union(detection[1:5].clone().detach(), gt[1:5].clone().detach())
                # iou = xywh_iou(torch.tensor(detection[1:5]).unsqueeze(0), torch.tensor(gt[1:5]).unsqueeze(0)).squeeze(0)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou > iou_threshold:
                # 这里的detection[0]是amount_bboxes的一个key，表示图片的编号，best_gt_idx是该key对应的value中真实框的下标
                if amount_bboxes[int(detection[0])][best_gt_idx] == 0:  # 只有没被占用的真实框才能用，0表示未被占用（占用：该真实框与某预测框匹配【两者IoU
                    # 大于设定的IoU阈值】）
                    TP[detection_idx] = 1  # 该预测框为TP
                    amount_bboxes[int(detection[0])][
                        best_gt_idx] = 1  # 将该真实框标记为已经用过了，不能再用于其他预测框。因为一个预测框最多只能对应一个真实框（最多：IoU小于IoU阈值时，预测框没有对应的真实框)
                else:
                    FP[detection_idx] = 1  # 虽然该预测框与真实框中的一个框之间的IoU大于IoU阈值，但是这个真实框已经与其他预测框匹配，因此该预测框为FP
            else:
                FP[detection_idx] = 1  # 该预测框与真实框中的每一个框之间的IoU都小于IoU阈值，因此该预测框直接为FP

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        # 套公式
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))

        # 把[0,1]这个点加入其中
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # 使用trapz计算AP
        ap = float(torch.trapz(precisions, recalls))
        average_precisions.append(ap)
        all_ap.append((classes[c].capitalize(), ap))
    mAP = sum(average_precisions) / len(average_precisions)
    all_ap.append(('', ''))
    all_ap.append(('mAP', mAP))
    return all_ap, mAP


def insert_over_union(boxes_preds, boxes_labels):
    box1_x1 = boxes_preds[..., 0:1]
    box1_y1 = boxes_preds[..., 1:2]
    box1_x2 = boxes_preds[..., 2:3]
    box1_y2 = boxes_preds[..., 3:4]  # shape:[N,1]

    box2_x1 = boxes_labels[..., 0:1]
    box2_y1 = boxes_labels[..., 1:2]
    box2_x2 = boxes_labels[..., 2:3]
    box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # 计算交集区域面积
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def mAP_compute(model, data_loader, iou_thresh, classes, generator):
    model = torch.nn.DataParallel(model)
    model.cuda().eval()
    decoder = Decoder(score_thres=0.01, nms_iou_thres=0.5, max_detection=300)
    clipbox = ClipBoxes()
    gt_boxes = []
    pred_boxes = []
    pred_cls = []
    for iter_num, data in enumerate(tqdm(data_loader)):
        torch.cuda.empty_cache()
        img = data['img']
        annot = data['annot'].squeeze(0)
        anchors = generator.grid_anchors(img)
        anchors = xywh2xyxy(anchors)
        img_idx = torch.tensor(iter_num).expand(annot.shape[0], 1)
        gt_boxes.append(torch.cat((img_idx, annot), dim=1).cuda())  # 适用于batchsize为1
        with torch.no_grad():
            cls, reg = model(img.cuda())  # cls[bs, w*h*a, nc], reg[bs, w*h*a, 4]  reg:[dx dy dw dh]
            cls_, idx, boxes = decoder([cls, reg, anchors])
            boxes = clipbox(img, boxes)
            if len(cls) == 0:
                continue
            #
            # img = np.asarray(img[0].detach().cpu().permute(1, 2, 0), dtype='uint8')
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # for i, box in enumerate(boxes[0]):
            #     pt1 = (int(box[0]), int(box[1]))
            #     pt2 = (int(box[2]), int(box[3]))
            #     cv2.rectangle(img, pt1, pt2, (255, 0, 0))
            #     clss = "%s" % (classes[int(idx[0][i])])
            #     cv2.putText(img, f'{clss}', (int(box[0]), int(box[1]) + 20), 0, 1, 2)
            #     # cv2.putText(img, f'{cls}_{scores[i]}', (int(box[0]), int(box[1]) + 20), 0, 1, 2)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)

            pred_idx = torch.tensor(iter_num, device=cls.device).expand(cls_.shape[1], 1)
            pred_boxes.append(
                torch.cat((pred_idx, boxes.squeeze(0), cls_.squeeze(0)[:, None], idx.squeeze(0)[:, None]), dim=1))
    pred_boxes = torch.cat(pred_boxes, dim=0).cuda()
    gt_boxes = torch.cat(gt_boxes, dim=0).cuda()

    all_ap, MAP = mean_average_precision(pred_boxes, gt_boxes, iou_threshold=iou_thresh, classes=classes)
    print(MAP)
    return all_ap, MAP


def pred2ori_box(reg_preds, anchors):
    """
    reg_preds: (N, sum(H*W)*A, 4)  format: [dx, dy, dw, dh]
    anchors:(sum(H*W)*A, 4)   format: [x1, y1, x2, y2]
    return (N, sum(H*W)*A, 4) format:[x1,y1,x2,y2]
    """

    mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
    std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
    # if torch.cuda.is_available():
    #     mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
    #     std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
    # else:
    #     mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
    #     std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
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


def map_compute(model,
                data_loader,
                logger=None,
                iou_thresh=0.6,
                conf_thresh=0.01,
                nc=20,
                plots=False,
                save_dir=None,
                generater=None,
                class_name=None,
                multi_label=False,
                agnostic=False,
                retur=False):
    with torch.no_grad():
        model.eval()
        device = next(model.parameters()).device
        iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        seen = 0
        s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        if logger:
            logger.info(s)
        dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        jdict, stats, ap, ap_class = [], [], [], []

        confusion_matrix = ConfusionMatrix(nc=nc)
        # names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}

        for batch_step, data in enumerate(tqdm(data_loader)):
            t1 = time_sync()
            images, annots = data['img'].to(device), data['annot'].to(device)

            anchors = generater.grid_anchors(images)
            # anchors = xywh2xyxy(anchors)
            nb, _, height, width = images.shape  # batch size, channels, height, width
            t2 = time_sync()
            dt[0] += t2 - t1
            pred_ = model(images, visualize=False)
            cls_preds, reg_preds = pred_[..., 0:-4], pred_[..., -4:]
            # for i in range(nb):
            #    clspred = cls_preds[i]
            #    print(max(torch.max(clspred, dim=1).values))

            reg_preds = pred2ori_box(reg_preds, anchors)  # regpred format: [x1, x2, y1, y2]
            reg_preds[..., [0, 2]] = reg_preds[..., [0, 2]].clamp_(min=0, max=width - 1)
            reg_preds[..., [1, 3]] = reg_preds[..., [1, 3]].clamp_(min=0, max=height - 1)
            # for i in range(nb):
            #     print(max(torch.max(cls_preds[i, ...], dim=1).values))
            cls_preds = cls_preds.sigmoid_()
            preds = torch.cat((reg_preds, cls_preds), dim=-1)
            dt[1] += time_sync() - t2
            lb = []  # for autolabelling
            # lb = [annots[annots[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling

            t3 = time_sync()
            #  batch preds ==> list[preds]
            out = retina_nms(preds, conf_thresh, iou_thresh, labels=lb, multi_label=multi_label,
                             agnostic=agnostic)  # list[[n1, 6], [n2, 6]]
            # out = non_max_suppression(preds, conf_thresh, iou_thresh, labels=lb, multi_label=True, agnostic=False)
            dt[2] += time_sync() - t3
            for si, pred in enumerate(out):
                labels = annots[si]  # 取出batch中一张图片的标签
                mask = labels[:, -1] != -1
                labels = labels[mask]
                nl = len(labels)
                tcls = labels[:, -1].tolist() if nl else []  # 类别真值
                seen += 1
                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                predn = pred.clone()  # [xyxy]

                if nl:
                    tbox = labels[:, 0:4]  # 目标框真值 [x1, y1, x2, y2]
                    labelsn = torch.cat((labels[:, 4:5], tbox), 1)
                    correct = process_batch(predn, labelsn, iouv)  # 计算IoU并匹配
                    if plots:
                        confusion_matrix.process_batch(predn, labelsn)
                else:
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

            # if plots and batch_step < 3:
            #     f = f'./val_batch{batch_step}_labels.jpg'  # labels
            #     Thread(target=plot_images, args=(images, annots, None, f, names[1:]), daemon=True).start()
            #     f = f'./val_batch{batch_step}_pred.jpg'  # predictions
            #     Thread(target=plot_images, args=(images, output_to_target(out), None, f, names[1:]), daemon=True).start()

    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    pd_data = []
    if len(stats) and stats[0].any():

        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=class_name[1:])
        for i in range(ap_class.shape[0]):
            pd_data.append({'Names': f'{class_name[ap_class[i]]}',
                            'Precision': '%.3f' % p[i],
                            'Recall': '%.3f' % r[i],
                            'AP.5': '%.3f' % ap[i][0],
                            'AP.5:.95': '%.3f' % ap.mean(1)[i]})
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results

    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    if logger:
        logger.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
        logger.info(pd.DataFrame(pd_data))
    if retur:
        return map50
