'''
训练
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import utils

from tensorboardX import SummaryWriter
from modle import modle_1
from dataset import dataset

 def main(args):
    # 设置日志， 和设备
    args.save_dir = utils.get_save_dir(args.save_dir, args.name, training=true)
    log = utils.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    log.info('Building model...')
    modle = modle_1(args)
    #model=torch.nn.parallel.DistributedDataParallel(model)
    #可能要用并行加快速度
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()
    ema = util.EMA(model, args.ema_decay)

    # 保存
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # 优化器
    optimizer = optim.Adadelta(model.parameters(), args.lr,
                               weight_decay=args.l2_wd)
    # 加载数据集
    train_loader = dataset.train_loader()
    dev_loader = dataset.dev_loader()

    # 训练
    log.info('Training...')
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for args in train_loader:
                # Setup for forward
                cw_idxs = cw_idxs.to(device)
                qw_idxs = qw_idxs.to(device)
                batch_size = cw_idxs.size(0)
                optimizer.zero_grad()

                # 前向传播
                log_p1, log_p2 = model(args)
                y1, y2 = y1.to(device), y2.to(device)
                loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
                loss_val = loss.item()

                # 反向传播
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step(step // batch_size)
                ema(model, step // batch_size)
