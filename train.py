


import numpy as np
import torch
import torch.optim as optim
import utils
import argparse

from tensorboardX import SummaryWriter
from modle import RNet
from dataset import dataset

# 设置参数
def args():
    parser = argparse.ArgumentParser('Train a model on SQuAD')
    parser.add_argument('--train_record_file',
                        type=str,
                        default='./data/train')
    parser.add_argument('--dev_record_file',
                        type=str,
                        default='--./data/dev')
    parser.add_argument('--test_record_file',
                        type=str,
                        default='--./data/test')
    parser.add_argument('--save_dir',
                        type=int,
                        default='./save/',
                        help='Base dir')
    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=True,
                        help='Name to identify training or test run.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.5,
                        help='Learning rate')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch_size')
    return parser.parse_args()

# 保存文件
def get_save_dir(base_dir, name, training):
    subdir = 'train' if training else 'test'
    save_dir = os.path.join(base_dir, subdir, f'{name}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

# 保存
def CheckpointSaver(args.save_dir,max_checkpoints,metric_name,log):


# 日志
def get_logger(log_dir, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def main():
    args=args()
    # 设置日志和设备
    args.save_dir = utils.get_save_dir(args.save_dir, args.name, training=true)
    log = get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    log.info('Building model...')
    modle = RNet()
    #model=torch.nn.parallel.DistributedDataParallel(model)
    #可能要用并行加快速度
    
    '''
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()
    # ema = util.EMA(model, args.ema_decay)
    '''

    # 优化器
    optimizer = optim.Adadelta(model.parameters(), args.lr,
                               weight_decay=args.decay)
    # 加载数据集
    log.info('Building dataset...')
    train_dataset = dataset.SQuAD()
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn)
    dev_dataset = dataset.SQuAD(dev)
    dev_loader = data.DataLoader(dev_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn)

    # 训练
    log.info('Training...')
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad():
            for args in train_loader:
                # Setup for forward
                eList[0] = eList[0].to(device)
                cList[0] = cList[0].to(device)
                batch_size = cList[0].size(0)
                optimizer.zero_grad()

                # 前向传播
                log_p1, log_p2 = model(args)
                p[0], p[1] = p[0].to(device), p[1].to(device)
                loss = F.nll_loss(log_p1, p[0]) + F.nll_loss(log_p2, p[1])
                loss_val = loss.item()

                # 反向传播
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step(step // batch_size)
                ema(model, step // batch_size)
