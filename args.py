import argparse

def get_train_args():
    '''Get args for train.py'''
    parser = argparse.ArgumentParser('Train a model on SQuAD')

    add_common_args(parser)
    add_train_test_args(parser)

    


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
    