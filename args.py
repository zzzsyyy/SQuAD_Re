import argparse

def get_train_args():
    '''Get args for train.py'''
    parser = argparse.ArgumentParser('Train a model on SQuAD')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--eval_steps',
                        type=int,
                        default=50000,
                        help='Number of steps between successive evaluations.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.5,
                        help='Learning rate')
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help='Decay rate for exponential moving average of parameters.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epoch for which to train.')
    parser.add_argument('--metric_name',
                        type=str,
                        default='F1',
                        choices=('NLL', 'EM', 'F1'),
                        help='Name of dev metric to determine best checkpoint.')
    if args.metric_name == 'NLL':
        # Best checkpoint is the one that minimizes negative log-likelihood
        args.maximize_metric = False
    elif args.metric_name in ('EM', 'F1'):
        # Best checkpoint is the one that maximizes EM or F1
        args.maximize_metric = True
    else:
        raise ValueError(f'Unrecognized metric name: "{args.metric_name}"')


def add_comme_args(parser):
    parser.add_argument('--train_record_file',
                        type=str,
                        default='./data/train')
    parser.add_argument('--dev_record_file',
                        type=str,
                        default='--./data/dev')
    parser.add_argument('--test_record_file',
                        type=str,
                        default='--./data/test')
    

def add_train_test_args(parser):
    parser.argument('--save_dir',
                    type=int,
                    default='./save/',
                    help='Base dir')
    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=True,
                        help='Name to identify training or test run.')
    