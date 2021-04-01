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
    '''parser.add_argument('--l2_wd',
                        type=float,
                        default=0,
                        help='L2 weight decay')'''
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epoch for which to train.')


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
    

# def add_