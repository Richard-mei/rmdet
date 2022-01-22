import argparse
import os
import os.path as osp
import time
from collections import ChainMap
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from tensorboardX import SummaryWriter
from Vision import Model, file, build_dataset, collate_fn, Resizer, get_root_logger, build_generator, get_by_key, \
    model_info, initialize_weights, pred2ori_box, retina_nms
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default='Vision/configs/retinanet_fpn_r50.yaml')
parser.add_argument('--weights', type=str, default='checkpoints/epoch_39_92.8.pth')
parser.add_argument('--num_workers', type=int, default=6)
parser.add_argument('--n_gpu', type=str, default='0', help='GPU device ids')
parser.add_argument('--freeze', type=int or list, default=None, help='Number of layers to freeze(list(int) or int). '
                                                                     'backbone=6, all=15')
parser.add_argument('--deploy', type=bool, default=False, help='switch the model mode(False for train,True to fuse '
                                                               'model).')
parser.add_argument('--use_dbb', type=bool, default=False, help='whether to use diverse branch block.')
parser.add_argument('--use_cot', type=bool, default=False, help='whether to use CoTLayer')
parser.add_argument('--source', '-s', type=str, default='voc',
                    help='Detect images source.cam for camera,voc for voc datasets.')
parser.add_argument('--save', type=bool, default=False,
                    help='Detect images source.cam for camera,voc for voc datasets.')
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.n_gpu

config = file.load(opt.config)
weights = torch.load(opt.weights)

data_cfg = config.get('dataset')
train_cfg = ChainMap(data_cfg.get('base'), data_cfg.get('train'))
eval_cfg = ChainMap(data_cfg.get('base'), data_cfg.get('eval'))

# logger
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
logpath = f'runs/detect/{train_cfg.get("cfg")}/{timestamp}'.lower()
if not osp.exists(logpath):
    os.makedirs(logpath)
writer = SummaryWriter(logdir=logpath, filename_suffix=f'_{timestamp}')
log_file = osp.join(logpath, f'{timestamp}.log')
logger = get_root_logger(name=f'{train_cfg["cfg"]}'.lower(), log_file=log_file, log_level='INFO',
                         format_='color_format')
names = get_by_key(config, 'CLASSES_NAME')

model = Model(config['model'], logger=logger, deploy=opt.deploy, use_dbb=opt.use_dbb, use_cot=opt.use_cot)
initialize_weights(model)
model.load_pretrained(weights)

if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()
else:
    model = torch.nn.DataParallel(model)
device = next(model.parameters()).device

# Freeze
freeze = [f'model.{x}.' for x in range(opt.freeze)] if isinstance(opt.freeze,
                                                                  int) else opt.freeze if opt.freeze else None  # layers to freeze
if freeze is not None:
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            v.requires_grad = False

model_info(model, verbose=True, logger=logger)

# dataset
eval_dataset = build_dataset(cfg=eval_cfg, transform=Resizer(img_sizes=eval_cfg.get('img_sizes')))
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_cfg.get('batch_size'), collate_fn=collate_fn)

generater = build_generator(config['generator'])

model.training = False
model.eval()
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, len(names))]

if opt.source == 'cam':
    cap = cv2.VideoCapture(0)

    while True:
        s, img = cap.read()
        # images = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).to(device)
        img1 = transforms.ToTensor()(img)
        images = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)(img1).unsqueeze(0)
        b, _, height, width = images.shape
        t1 = time.time()
        preds = model(images, visualize=False)
        cls_preds, reg_preds = preds[..., 0:-4], preds[..., -4:]

        t2 = time.time()
        anchors = generater.grid_anchors(images)
        reg_preds = pred2ori_box(reg_preds, anchors)  # regpred format: [x1, y1, x2, y2]
        reg_preds[..., [0, 2]] = reg_preds[..., [0, 2]].clamp_(min=0, max=width - 1)
        reg_preds[..., [1, 3]] = reg_preds[..., [1, 3]].clamp_(min=0, max=height - 1)
        cls_preds = cls_preds.sigmoid_()
        preds = torch.cat((reg_preds, cls_preds), dim=-1)
        out = retina_nms(preds, conf_thresh=0.3, iou_thresh=0.5, labels=None, multi_label=True,
                         agnostic=False)  # list[[n1, 6], [n2, 6]]
        for i in range(b):
            pred = out[i]
            boxes, scores, cls = pred[:, :4], pred[:, 4], pred[:, 5]
            for j, box in enumerate(boxes):
                pt1 = (int(box[0]), int(box[1]))
                pt2 = (int(box[2]), int(box[3]))
                cv2.rectangle(img, pt1, pt2, tuple(255 * k for k in list(colors[int(cls[j])][:3])))
                clss = "%s" % names[int(cls[j])]
                cv2.putText(img, f'{clss}_%.3f' % (scores[j]), (int(box[0]), int(box[1]) + 20), 0, 0.5,
                            tuple(255 * k for k in list(colors[int(cls[j])][:3])), 1)
            cv2.putText(img, f'FPS: %.3f' % (1 / (t2 - t1)), (5, 15), 0, 0.5, (0, 0, 255), 1)
        cv2.imshow('img', img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

elif opt.source == 'voc':
    for idx, data in enumerate(eval_loader):
        images, annots = data['img'].to(device), data['annot'].to(device)
        ids = data['id']
        b, _, height, width = images.shape
        t1 = time.time()
        preds = model(images, visualize=True if idx == 53 else False)
        t2 = time.time()
        cls_preds, reg_preds = preds[..., 0:-4], preds[..., -4:]
        anchors = generater.grid_anchors(images)  # anchors format: [x1, y1, x2, y2]
        reg_preds = pred2ori_box(reg_preds, anchors)  # regpred format: [x1, y1, x2, y2]
        reg_preds[..., [0, 2]] = reg_preds[..., [0, 2]].clamp_(min=0, max=width - 1)
        reg_preds[..., [1, 3]] = reg_preds[..., [1, 3]].clamp_(min=0, max=height - 1)
        cls_preds = cls_preds.sigmoid_()
        preds = torch.cat((reg_preds, cls_preds), dim=-1)
        out = retina_nms(preds, conf_thresh=0.3, iou_thresh=0.2, labels=None, multi_label=False,
                         agnostic=False)  # list[[n1, 6], [n2, 6]]
        for i in range(b):
            img = np.asarray(images[i].detach().cpu().permute(1, 2, 0), dtype='uint8')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            pred = out[i]
            boxes, scores, cls = pred[:, :4], pred[:, 4], pred[:, 5]
            for j, box in enumerate(boxes):
                pt1 = (int(box[0]), int(box[1]))
                pt2 = (int(box[2]), int(box[3]))
                cv2.rectangle(img, pt1, pt2, tuple(255 * k for k in list(colors[int(cls[j])][:3])))
                clss = "%s" % names[int(cls[j])]
                cv2.putText(img, f'{clss}_%.3f' % (scores[j]), (int(box[0]), int(box[1]) + 20), 0, 0.5,
                            tuple(255 * k for k in list(colors[int(cls[j])][:3])), 1)
            cv2.putText(img, f'FPS: %.3f' % (1 / (t2 - t1)), (5, 15), 0, 0.5, (255, 0, 0), 1)
            try:
                assert opt.save is False
                cv2.imshow(f'img_{ids[0][1]}', img)
                key = cv2.waitKey(0)
                if key & 0xFF == ord('q') or key == 113:
                    cv2.destroyAllWindows()
                    break
                break
            except:
                print(f'save detected result to {logpath}/{ids[0][1]}.jpg')
                cv2.imwrite(f'{logpath}/{ids[0][1]}.jpg', img)
        # key = cv2.waitKey(0)
        # if key & 0xFF == ord('w') or key == 119:
        #    cv2.destroyAllWindows()
        #    break
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
