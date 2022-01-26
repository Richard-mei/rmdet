import argparse
import collections
import copy
import glob
import os
import os.path as osp
import time
from collections import ChainMap
import numpy as np
import pandas as pd
import torch.utils.data
from torch import optim
from torchvision import transforms
from tqdm import tqdm

from Vision import Model, file, build_dataset, Normalizer, collate_fn, Resizer, build_optimizer, \
    collect_env, get_root_logger, ComputeLoss, build_generator, get_by_key, model_info, \
    initialize_weights, time_sync, ConfusionMatrix, retina_nms, process_batch, ap_per_class, pred2ori_box

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default='Vision/configs/retinanet_fpn_r50.yaml')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--epochs', type=int, default=50, help='number of total epochs')
parser.add_argument('--best_map', type=int, default=0.5)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--n_gpu', type=str, default='0', help='GPU device ids')
parser.add_argument('--log_iter', type=int, default=20)
parser.add_argument('--eval_epoch', type=int, default=1)
parser.add_argument('--freeze', type=int or list, default=None, help='Number of layers to freeze(list(int) or int). '
                                                                     'backbone=6, all=15')
parser.add_argument('--deploy', type=bool, default=False, help='switch the model mode(False for train,True to fuse '
                                                               'model).')
parser.add_argument('--use_dbb', type=bool, default=False, help='whether to use diverse branch block.')
parser.add_argument('--use_cot', type=bool, default=False, help='whether to use CoTLayer')
parser.add_argument('--save_path', type=str, default=None, help='path to save onnx model.')
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.n_gpu

config = file.load(opt.config)
weights = torch.load(opt.weights)

data_cfg = config.get('dataset')
train_cfg = ChainMap(data_cfg.get('base'), data_cfg.get('train'))
eval_cfg = ChainMap(data_cfg.get('base'), data_cfg.get('eval'))

# logger
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
logpath = f'runs/val/{timestamp}'.lower()
save_path = logpath if opt.save_path is None else opt.save_path

if not osp.exists(logpath):
    os.makedirs(logpath)
file.save(config, osp.join(logpath, f'{timestamp}.json'))
log_file = osp.join(logpath, f'{timestamp}.log')
logger = get_root_logger(name=f'{train_cfg["cfg"]}'.lower(), log_file=log_file, log_level='INFO',
                         format_='color_format')
env_info_dict = collect_env()
env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
dash_line = '-' * 60 + '\n'
logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

# model
model = Model(config['model'], logger=logger, deploy=opt.deploy, use_dbb=opt.use_dbb, use_cot=opt.use_cot)
initialize_weights(model)

model.load_pretrained(weights)

if torch.cuda.is_available():
    # model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()
else:
    model = torch.nn.DataParallel(model)

device = next(model.parameters()).device

model_info(model, verbose=True, logger=logger)

# dataset
eval_dataset = build_dataset(cfg=eval_cfg,
                             transform=transforms.Compose([Resizer(img_sizes=eval_cfg.get('img_sizes')), Normalizer()]))
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_cfg.get('batch_size'), collate_fn=collate_fn)

# optimizer
optimizer = build_optimizer(config.get('optimizer'), params=model.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
loss_hist = collections.deque(maxlen=100)
compute_loss = ComputeLoss(config['loss'], config['generator'])
generater = build_generator(config['generator'])
names = get_by_key(config, 'CLASSES_NAME')

# map_compute(model, eval_loader, logger, iou_thresh=0.5, conf_thresh=0.01, nc=80, plots=True, save_dir=logpath,
#             generater=generater,
#             class_name=names[1:], multi_label=False, agnostic=False)


s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
if logger:
    logger.info(s)
dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
jdict, stats, ap, ap_class = [], [], [], []
device = next(model.parameters()).device
iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
niou = iouv.numel()
seen = 0
confusion_matrix = ConfusionMatrix(nc=20)
iou_thresh = 0.5
conf_thresh = 0.01
plots = True
save_dir = logpath
nc = 20
with torch.no_grad():
    model.training = False
    model.eval()

    model_fused = copy.deepcopy(model)

    assert model_fused.training is False
    for module in model_fused.modules():
        if hasattr(module, 'switch_to_deploy'):
            try:
                module.switch_to_deploy()
            except:
                continue
    torch.save(model_fused.state_dict(), f'{save_path}/fused_last_epoch.pth')
    # dummy_input = torch.randn(size=(1, 3, 480, 640)).cuda().half()
    # torch.onnx.export(model_fused.module, dummy_input, f'{save_path}/retina.onnx', input_names=['images'], output_names=['output'], opset_version=11)

    epoch_loss = []
    for batch_step, data in enumerate(tqdm(eval_loader)):
        t1 = time_sync()

        # global_steps = len(train_loader) * epoch + batch_step
        images, annots = data['img'].to(device), data['annot'].to(device)
        nb, _, height, width = images.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1
        anchors = generater.grid_anchors(images)  # anchors format: [xyxy]
        preds = model.forward(images, visualize=False)
        cls_preds, reg_preds = preds[..., 0:-4], preds[..., -4:]
        # cls_preds, reg_preds = model.forward(images, visualize=False)
        cls_preds = cls_preds.sigmoid_()

        reg_preds = pred2ori_box(reg_preds, anchors)  # regpred format: [x1, x2, y1, y2]
        reg_preds[..., [0, 2]] = reg_preds[..., [0, 2]].clamp_(min=0, max=width - 1)
        reg_preds[..., [1, 3]] = reg_preds[..., [1, 3]].clamp_(min=0, max=height - 1)

        preds = torch.cat((reg_preds, cls_preds), dim=-1)
        dt[1] += time_sync() - t2
        lb = []  # for autolabelling
        # lb = [annots[annots[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling

        t3 = time_sync()
        #  batch preds ==> list[preds]
        out = retina_nms(preds, conf_thresh, iou_thresh, labels=lb, multi_label=False,
                         agnostic=False)  # list[[n1, 6], [n2, 6], ...]
        # out = non_max_suppression(preds, conf_thresh, iou_thresh, labels=lb, multi_label=False, agnostic=False)
        dt[2] += time_sync() - t3
        # img_idx = torch.tensor(batch_step).expand(annots.shape[0], 1)

        for si, pred in enumerate(out):
            labels = annots[si]
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
        #     f = f'{save_dir}/val_batch{batch_step}_labels.jpg'  # labels
        #     Thread(target=plot_images, args=(images, annots, None, f, names[1:]), daemon=True).start()
        #     f = f'{save_dir}/val_batch{batch_step}_pred.jpg'  # predictions
        #     Thread(target=plot_images, args=(images, output_to_target(out), None, f, names[1:]), daemon=True).start()

    stats = [np.concatenate(x, 0) for x in zip(*stats)]

    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir,
                                              names=names[1:])
        pd_data = []
        for i in range(ap_class.shape[0]):
            pd_data.append({'Names': f'{names[ap_class[i]]}',
                            'Precision': '%.3f' % p[i],
                            'Recall': '%.3f' % r[i],
                            'AP.5': '%.3f' % ap[i][0],
                            'AP.5:.95': '%.3f' % ap.mean(1)[i]})
            # print(f'{names[ap_class[i]]}', [p[i], r[i], ap[i][0], ap.mean(1)[i]])

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
