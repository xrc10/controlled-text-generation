import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from ctextgen.dataset import SST_Dataset
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

if args.gpu:
    model.load_state_dict(torch.load(os.path.join(args.save_path,
                    'discriminator_{}.bin'.format(args.dataset))))
else:
    model.load_state_dict(torch.load(os.path.join(args.save_path,
                    'discriminator_{}.bin'.format(args.dataset))),
                    map_location=lambda storage, loc: storage))




# Samples latent and conditional codes randomly from prior
z = model.sample_z_prior(1)
c = model.sample_c_prior(1)

# Generate positive sample given z
c[0, 0], c[0, 1] = 1, 0

_, c_idx = torch.max(c, dim=1)
sample_idxs = model.sample_sentence(z, c, temp=0.1)

print('\nSentiment: {}'.format(dataset.idx2label(int(c_idx))))
print('Generated: {}'.format(dataset.idxs2sentence(sample_idxs)))

# Generate negative sample from the same z
c[0, 0], c[0, 1] = 0, 1

_, c_idx = torch.max(c, dim=1)
sample_idxs = model.sample_sentence(z, c, temp=0.8)

print('\nSentiment: {}'.format(dataset.idx2label(int(c_idx))))
print('Generated: {}'.format(dataset.idxs2sentence(sample_idxs)))

print()
