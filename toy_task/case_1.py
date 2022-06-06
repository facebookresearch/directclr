# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='Case 1: Strong Augmentation')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--noise', default=0.01, type=float)


def augmentation(x, noise):
    x1 = x + torch.cat([torch.zeros(128, 8), torch.randn(128, 8) * noise], dim=1).cuda()
    x2 = x + torch.cat([torch.zeros(128, 8), torch.randn(128, 8) * noise], dim=1).cuda()
    return x1, x2


def main():
    np.random.seed(0)

    args = parser.parse_args()

    # data
    x = torch.randn(10000, 16)

    # network (single linear layer)
    net = nn.Linear(16, 16).cuda()

    # optimizer, SGD with fixed learning rate and no weight decay
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=0)


    for i in range(10000):
        # sample data
        idx = np.random.choice(10000, 128, replace=False)
        xi = x[idx].cuda()

        # apply augmentation
        x1, x2 = augmentation(xi, args.noise)

        # apply encoder
        z1 = net(x1)
        z2 = net(x2)

        # train
        optimizer.zero_grad()
        loss = infoNCE(z1, z2)
        loss.backward()
        optimizer.step()

        # print spectrum
        if i % 10 == 9:
            embedding_spectrum = spectrum(net)
            print("\n iteration:", i, embedding_spectrum)



def spectrum(net):
    x = torch.randn(1000, 16).cuda()
    z = net(x)
    z = z.cpu().detach().numpy()
    z = np.transpose(z)
    c = np.cov(z)

    _, d, _ = np.linalg.svd(c)
    return d


def infoNCE(z1, z2, temperature=0.1):
    logits = z1 @ z2.T
    logits /= temperature
    n = z1.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).cuda()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss


if __name__ == '__main__':
    main()

