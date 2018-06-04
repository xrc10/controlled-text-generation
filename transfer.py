import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from ctextgen.datasets import *
from ctextgen.model import RNN_VAE

import argparse
import random
import time

parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train Discriminator'
)

parser.add_argument('--dataset', default='SST', help='string of the dataset name')
parser.add_argument('--data_path', default=None,
                    help='string of the path of the dataset')
parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--save_path', default='models/',
                    help='string of path to save the model')

args = parser.parse_args()


mb_size = 32
z_dim = 20
h_dim = 64
lr = 1e-3
lr_decay_every = 1000000
n_iter = 20000
log_interval = 1000
z_dim = h_dim
c_dim = 2

if args.dataset == 'SST':
    dataset = SST_Dataset()
elif 'GYAFC' in args.dataset:
    dataset = GYAFC_Dataset(data_path=args.data_path)
else:
    logger.error('unrecognized dataset: {}'.format(args.dataset))
    sys.exit(-1)

torch.manual_seed(int(time.time()))

model = RNN_VAE(
    dataset.n_vocab, h_dim, z_dim, c_dim, p_word_dropout=0.3,
    pretrained_embeddings=dataset.get_vocab_vectors(), freeze_embeddings=True,
    gpu=args.gpu
)


model.load_state_dict(torch.load(os.path.join(args.save_path,
                    'discriminator_{}.bin'.format(args.dataset))))

def main():
    f_pred_out = open(os.path.join(args.save_path, 'pred.txt'), 'w')
    f_compare_out = open(os.path.join(args.save_path, 'pred.txt'), 'w')
    trg_c = model.sample_c_prior(1)
    # Generate target sample from the source z
    trg_c[0, 0], trg_c[0, 1] = 0, 1
    while 1:
        test_batch = dataset.next_test_batch(args.gpu)

        if test_batch is None:
            break

        src_inputs = test_batch[0]
        src_z = model.forward_encoder(src_inputs)[0]

        for i in range(src_z.size()[0]):
            sample_idxs = model.sample_sentence(src_z[i,:], trg_c)
            sample_sent = dataset.idxs2sentence(sample_idxs)
            ori_sent = dataset.idxs2sentence(src_inputs[i].tolist())

            f_pred_out.write("{}\n".format(sample_sent))
            f_compare_out.write("{}\t{}\n".format(ori_sent, sample_sent))

if __name__ == '__main__':
    main()
