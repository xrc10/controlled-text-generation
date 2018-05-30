
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
import logging

parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train VAE as in Bowman, 2016, with c ~ p(c)'
)

parser.add_argument('--dataset', default='SST', help='string of the dataset name')
parser.add_argument('--data_path', default=None,
                    help='string of the path of the dataset')
parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--save', default=False, action='store_true',
                    help='whether to save model or not')
parser.add_argument('--save_path', default='models/',
                    help='string of path to save the model')

args = parser.parse_args()

logging.basicConfig(filename=args.dataset+'.log',level=logging.DEBUG)

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

model = RNN_VAE(
    dataset.n_vocab, h_dim, z_dim, c_dim, p_word_dropout=0.3,
    pretrained_embeddings=dataset.get_vocab_vectors(), freeze_embeddings=False,
    gpu=args.gpu
)

def main():
    # Annealing for KL term
    kld_start_inc = 3000 # number of iters that we start to include KL divergence
    kld_weight = 0.01 # staring weight
    kld_max = 0.15 # max(final) weight
    kld_inc = (kld_max - kld_weight) / (n_iter - kld_start_inc) # increased KL
                                                                # weight

    trainer = optim.Adam(model.vae_params, lr=lr)

    for it in range(n_iter):
        inputs, labels = dataset.next_batch(args.gpu)

        recon_loss, kl_loss = model.forward(inputs)
        loss = recon_loss + kld_weight * kl_loss

        # Anneal kl_weight
        if it > kld_start_inc and kld_weight < kld_max:
            kld_weight += kld_inc

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm(model.vae_params, 5)
        trainer.step()
        trainer.zero_grad()

        if it % log_interval == 0:
            z = model.sample_z_prior(1)
            c = model.sample_c_prior(1)

            sample_idxs = model.sample_sentence(z, c)
            sample_sent = dataset.idxs2sentence(sample_idxs)

            print('Iter-{}; Loss: {:.4f}; Recon: {:.4f}; KL: {:.4f}; Grad_norm: {:.4f};'
                  .format(it, loss.data[0], recon_loss.data[0], kl_loss.data[0], grad_norm))

            print('Sample: "{}"'.format(sample_sent))
            print()

        # Anneal learning rate
        new_lr = lr * (0.5 ** (it // lr_decay_every))
        for param_group in trainer.param_groups:
            param_group['lr'] = new_lr


def save_model(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    torch.save(model.state_dict(), os.path.join(args.save_path,
                'vae_{}.bin'.format(args.dataset)))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        if args.save:
            save_model(args)
        exit(0)

    if args.save:
        save_model(args)
