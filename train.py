import argparse
import collections
import copy
import glob
import os
import os.path as osp
import time
from collections import ChainMap
import numpy as np
import torch.utils.data
from tensorboardX import SummaryWriter
from torch import optim
from torchvision import transforms

from Vision import Model, file, build_dataset, Augmenter, Normalizer, collate_fn, Resizer, build_optimizer, \
    collect_env, get_root_logger, ComputeLoss, build_generator, get_by_key, model_info, \
    map_compute, initialize_weights, tensorboard_event_vis

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default='Vision/configs/retinanet_fpn_r50.yaml')
parser.add_argument('--resume_from', type=str, default=None)
parser.add_argument('--pretrained', type=str, default=None)
parser.add_argument('--exclude_layer', type=list, default=['fc.weight', 'fc.bias'],
                    help='layers weights to del,default is repvgg.')
parser.add_argument('--epochs', type=int, default=30, help='number of total epochs')
parser.add_argument('--best_map', type=int, default=0.9)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--n_gpu', type=str, default='0', help='GPU device ids')
parser.add_argument('--log_iter', type=int, default=20)
parser.add_argument('--eval_epoch', type=int, default=1)
parser.add_argument('--freeze', type=int or list, default=0, help='Number of layers to freeze(list(int) or int). '
                                                                  'backbone=6, all=15')
parser.add_argument('--deploy', type=bool, default=False, help='switch the model mode(False for train,True to fuse '
                                                               'model).')
parser.add_argument('--use_dbb', type=bool, default=False, help='whether to use diverse branch block.')
parser.add_argument('--use_cot', type=bool, default=False, help='whether to use CoTLayer')
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.n_gpu

if opt.resume_from is not None:
    config = file.load(glob.glob(f'{opt.resume_from}/*.json')[0])
    weights = torch.load(glob.glob(f'{opt.resume_from}/epoch_23_0.705.pth')[0])
else:
    config = file.load(opt.config)
    weights = None

data_cfg = config.get('dataset')
train_cfg = ChainMap(data_cfg.get('base'), data_cfg.get('train'))
eval_cfg = ChainMap(data_cfg.get('base'), data_cfg.get('eval'))

# logger
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
logpath = f'runs/{config.get("cfg")}/{train_cfg.get("cfg")}/{timestamp}'.lower()
if not osp.exists(logpath):
    os.makedirs(logpath)
file.save(config, osp.join(logpath, f'{timestamp}.json'))
writer = SummaryWriter(logdir=logpath, filename_suffix=f'_{timestamp}')
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
if weights is not None:
    model.load_pretrained(weights)
elif opt.pretrained is not None:
    model.load_pretrained(torch.load(opt.pretrained), exclude=opt.exclude_layer)

if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()
else:
    model = torch.nn.DataParallel(model)
device = next(model.parameters()).device

# Freeze module
freeze = [f'model.{x}.' for x in range(opt.freeze)] if isinstance(opt.freeze,
                                                                  int) else opt.freeze if opt.freeze else None  # layers to freeze
logger.info(f'Frozen module: {freeze}' + '\n' + dash_line)
if freeze is not None:
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            v.requires_grad = False

model_info(model, verbose=True, logger=logger)

# dataset
train_dataset = build_dataset(cfg=train_cfg, transform=transforms.Compose([
    Resizer(img_sizes=train_cfg.get('img_sizes')),
    Normalizer()]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_cfg.get('batch_size'), collate_fn=collate_fn,
                                           shuffle=True, pin_memory=True)
# sampler = WeightedRandomSampler: 根据各样本权重选择数据, RandomSampler：shuffle=True时调用 # https://zhuanlan.zhihu.com/p/117270644


eval_dataset = build_dataset(cfg=eval_cfg, transform=transforms.Compose([Resizer(img_sizes=eval_cfg.get('img_sizes')),
                                                                         Normalizer()]))
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_cfg.get('batch_size'), collate_fn=collate_fn)

# optimizer
optimizer = build_optimizer(config.get('optimizer'), params=model.parameters())
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.001)
# scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[13, 22], gamma=0.1)

compute_loss = ComputeLoss(config['loss'], config['generator'])
generater = build_generator(config['generator'])
names = get_by_key(config, 'CLASSES_NAME')

best_map = opt.best_map
for epoch in range(opt.epochs):
    model.training = True
    model.train()
    # epoch_loss = []
    if epoch == 2:
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
    for batch_step, data in enumerate(train_loader):
        global_steps = len(train_loader) * epoch + batch_step
        images, annots = data['img'].to(device), data['annot'].to(device)
        anchors = generater.grid_anchors(images)
        optimizer.zero_grad()
        # for pram in optimizer.param_groups:
        #     lr = pram['lr']
        lr = scheduler.get_last_lr()[0]
        preds = model.forward(images, visualize=False)
        losses_dict = compute_loss(preds, annots, anchors)
        loss = losses_dict['total_loss']
        if bool(loss == 0):
            continue

        for k, v in losses_dict.items():
            writer.add_scalar(k, v, global_step=global_steps)
        writer.add_scalar('lr', lr, global_step=epoch)
        # flood = (loss - b).abs() + b
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        # epoch_loss.append(float(loss))

        logg_info = f'Epoch: %3s | Iteration: %8s/%4s/%-4s' % (epoch + 1, global_steps, batch_step, len(train_loader))

        for k, v in losses_dict.items():
            logg_info += f' | %-6s: %1.5f' % (k, v)
        if batch_step % opt.log_iter == 0:
            logger.info(logg_info + f' | lr: %1.5f' % lr)
        del loss

    #  eval loss
    with torch.no_grad():
        for batch_step, data in enumerate(eval_loader):
            _steps = len(train_loader) * epoch + batch_step
            images, annots = data['img'].to(device), data['annot'].to(device)
            anchors = generater.grid_anchors(images)
            preds = model.forward(images, visualize=False)
            losses_dict = compute_loss(preds, annots, anchors)
            for k, v in losses_dict.items():
                writer.add_scalar(f'eval_{k}', v, global_step=_steps)
    # torch.save(model.state_dict(), f'{logpath}/epoch_{epoch+1}.pth')
    # model_fused = copy.deepcopy(model)
    # for module in model_fused.modules():
    #     if hasattr(module, 'switch_to_deploy'):
    #         module.switch_to_deploy()
    # torch.save(model_fused.state_dict(), f'{logpath}/fused_last_epoch.pth')
    scheduler.step()
    config['epoch'] = epoch + 1
    config['steps'] = len(train_loader) * (epoch + 1)
    config['optimizer']['lr'] = lr  # 保存当前学习率
    file.save(config, osp.join(logpath, f'{timestamp}.json'))
    if (epoch + 1) % opt.eval_epoch == 0:
        torch.cuda.empty_cache()
        model.training = False
        model.eval()
        map50 = map_compute(model, eval_loader, logger, iou_thresh=0.5, conf_thresh=0.01, nc=len(names[1:]), plots=True,
                            save_dir=logpath,
                            generater=generater, class_name=names, multi_label=False, agnostic=False, retur=True)

        # all_ap, mAP = mAP_compute(model, eval_loader, iou_thresh=0.5,
        #                           classes=get_by_key(config, 'CLASSES_NAME'), generator=generater)
        # logger.info(dash_line)
        # logger.info(pd.DataFrame(all_ap, columns=['Class', 'AP']))
        # logger.info(dash_line)
        if map50 > best_map:
            best_map = map50
            torch.save(model.state_dict(), '{}/epoch_{}_{:.3f}.pth'.format(logpath, epoch + 1, map50))

writer.export_scalars_to_json(osp.join(logpath, f'{timestamp}_scalars.json'))
model.eval()
tensorboard_event_vis()(glob.glob(f'{logpath}/events.out*')[0], 'total_loss', f'{logpath}/total_loss.txt',
                        f'{logpath}/total_loss', 'upper right')
