# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import os
import random
import shutil
import sys
import time
import warnings
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Visualize spectrum')
parser.add_argument('--data', metavar='DIR', default="/datasets01/imagenet_full_size/061417",
                    help='path to dataset')
parser.add_argument('--rep', action="store_true")
parser.add_argument('--projector', action="store_true")
parser.add_argument('--checkpoint', type=str)


def main():
    args = parser.parse_args()

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    val_dataset = datasets.ImageFolder(os.path.join(args.data, 'val'), val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=32, pin_memory=True)

    model, projector = load(args.checkpoint, args.projector)
    embedding_spectrum = singular(args, model, projector, val_loader)
    print(np.log(embedding_spectrum))


def load(filename, projector):
    # create model
    model = models.resnet50()
    model.fc = nn.Identity()

    # create projector
    if projector:
        sizes = [2048, 2048, 128]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        layers.append(nn.BatchNorm1d(sizes[-1]))
        projector = nn.Sequential(*layers)
    else:
        projector = nn.Identity()

    cudnn.benchmark = True

    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location="cpu")

        backbone = checkpoint['backbone']
        msg = model.load_state_dict(backbone, strict=True)
        assert set(msg.missing_keys) == set()

        if projector:
            proj = checkpoint['projector']
            msg = projector.load_state_dict(proj, strict=True)
            assert set(msg.missing_keys) == set()

        print("=> loaded pre-trained model '{}'".format(filename))
    else:
        print("=> no checkpoint found at '{}'".format(filename))
        raise

    model = model.cuda()
    projector = projector.cuda()
    return model, projector


def singular(args, model, projector, val_loader):

    model.eval()
    projector.eval()

    with torch.no_grad():
        latents = []

        for _, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            if args.rep:
                output = model(images)
            else:
                output = projector(model(images))

            latents.append(output)

        latents = torch.cat(latents, dim=0)

        z = torch.nn.functional.normalize(latents, dim=1)

        # calculate covariance
        z = z.cpu().detach().numpy()
        z = np.transpose(z)
        c = np.cov(z)
        _, d, _ = np.linalg.svd(c)

    return d


def exclude_bias_and_norm(p):
    return p.ndim == 1


if __name__ == '__main__':
    main()
