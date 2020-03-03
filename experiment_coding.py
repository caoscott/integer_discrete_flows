# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
from torchvision import transforms as T

import timer
from utils.load_data import load_dataset

parser = argparse.ArgumentParser(
    description='PyTorch Discrete Normalizing flows')

parser.add_argument('-d', '--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'imagenet32', 'imagenet64', 'oi_test'],
                    metavar='DATASET',
                    help='Dataset choice.')

parser.add_argument('-s', '--snap-dir', type=str, help="model dir")

parser.add_argument('-nc', '--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--manual_seed', type=int,
                    help='manual seed, if not given resorts to random seed.')

parser.add_argument('-li', '--log_interval', type=int, default=20, metavar='LOG_INTERVAL',
                    help='how many batches to wait before logging training status')

parser.add_argument('--evaluate_interval_epochs', type=int, default=25,
                    help='Evaluate per how many epochs')


# optimization settings
parser.add_argument('-e', '--epochs', type=int, default=2000, metavar='EPOCHS',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('-es', '--early_stopping_epochs', type=int, default=300, metavar='EARLY_STOPPING',
                    help='number of early stopping epochs')

parser.add_argument('-bs', '--batch_size', type=int, default=10, metavar='BATCH_SIZE',
                    help='input batch size for training (default: 100)')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, metavar='LEARNING_RATE',
                    help='learning rate')
parser.add_argument('--warmup', type=int, default=10,
                    help='number of warmup epochs')

parser.add_argument('--data_augmentation_level', type=int, default=2,
                    help='data augmentation level')

parser.add_argument('--no_decode', action='store_true', default=False,
                    help='disables decoding')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}


def encode_images(img, model, decode):
    batchsize, img_c, img_h, img_w = img.size()
    # c, h, w = model.args.input_size
    c, h, w = img_c, img_h, img_w

    # assert img_h == img_w and h == w

    if img_h != h:
        assert img_h % h == 0
        steps = img_h // h

        states = [[] for i in range(batchsize)]
        state_sizes = [0 for i in range(batchsize)]
        bpd = [0 for i in range(batchsize)]
        error = 0

        for j in range(steps):
            for i in range(steps):
                r = encode_patches(
                    img[:, :, j*h:(j+1)*h, i*w:(i+1)*w], model, decode)
                for b in range(batchsize):

                    if r[0][b] is None:
                        states[b].append(None)
                    else:
                        states[b].extend(r[0][b])
                    state_sizes[b] += r[1][b]
                    bpd[b] += r[2][b] / steps**2
                    error += r[3]
        return states, state_sizes, bpd, error
    else:
        return encode_patches(img, model, decode)


def encode_patches(imgs, model, decode):
    batchsize, img_c, img_h, img_w = imgs.size()
    c, h, w = img_c, img_h, img_w
    # c, h, w = model.args.input_size
    # assert img_h == h and img_w == w

    states = model.encode(imgs)

    bpd = model.forward(imgs)[1].cpu().numpy()

    state_sizes = []
    error = 0

    for b in range(batchsize):
        if states[b] is None:
            # Using escape bit ;)
            state_sizes += [8 * img_c * img_h * img_w + 1]

            # Error remains unchanged.
            print('Escaping, not encoding.')

        else:
            if decode:
                x_recon = model.decode([states[b]])

                error += torch.sum(
                    torch.abs(x_recon.int() - imgs[b].int())).item()

            # Append state plus an escape bit
            state_sizes += [32 * len(states[b]) + 1]

    return states, state_sizes, bpd, error


def run(args, kwargs):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # args.snap_dir = snap_dir = \
    #    '/u/scottcao/scratch/idf/discrete_logisticoi_flows_8_levels_4__2020-02-23_12_29_53/'

    # ==================================================================================================================
    # SNAPSHOTS
    # ==================================================================================================================

    # ==================================================================================================================
    # LOAD DATA
    # ==================================================================================================================
    _, _, test_loader, args = load_dataset(args, **kwargs)

    final_model = torch.load(args.snap_dir + 'a.model')
    if hasattr(final_model, 'module'):
        final_model = final_model.module
    final_model = final_model.cuda()

    sizes = []
    errors = []
    bpds = []

    acc = timer.TimeAccumulator()

    t = 0
    with torch.no_grad():
        for data in test_loader:

            with acc.execute():
                if args.cuda:
                    data = data.cuda()
                state, state_sizes, bpd, error = \
                    encode_images(data, final_model, decode=not args.no_decode)

            errors += [error]
            bpds.extend(bpd)
            sizes.extend(state_sizes)

            t += len(data)

            print(
                'Examples: {}/{} bpd compression: {:.3f} error: {},'
                ' analytical bpd {:.3f} time: {:.3f}'.format(
                    t, len(test_loader.dataset),
                    np.mean(sizes) / np.prod(data.size()[1:]),
                    np.sum(errors),
                    np.mean(bpds),
                    acc.mean_time_spent(),
                ))

            # if args.no_decode:
            # print('Not testing decoding.')
            # else:
            # print('Error: {}'.format(np.sum(errors)))
    print('Final bpd: {:.3f} error: {}'.format(
        np.mean(sizes) / np.prod(data.size()[1:]),
        np.sum(errors)))
    print(
        'Took {:.3f} seconds / example'.format(acc.mean_time_spent()))


if __name__ == "__main__":

    run(args, kwargs)
