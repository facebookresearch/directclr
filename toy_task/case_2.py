import torch
import torch.nn as nn
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='Case 2: Implicit Regularization')
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--noise', default=0.001, type=float)


def create_model():
    layers = []
    for _ in range(2):
        layer = torch.nn.Linear(16, 16, bias=False)
        layers.append(layer)
        layer.weight.data = init()
        
    return nn.Sequential(*layers)


def augmentation(x, noise):
    x1 = x + torch.randn_like(x) * noise
    x2 = x + torch.randn_like(x) * noise
    return x1, x2


def main():
    np.random.seed(0)

    args = parser.parse_args()

    # data
    x = torch.randn(10000, 16)

    net = create_model().cuda()

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

        if i % 10 == 0:
            W1_singular, W2_singular, align_matrix = alignment(net)
            embedding_spectrum = spectrum(net)
            print("\n iteration:", i)
            print("W1_singular", np.log(W1_singular))
            print("W2_singular", np.log(W2_singular))
            print("embedding_spectrum", np.log(embedding_spectrum))

    
    import cv2
    cv2.imwrite('./align-matrix.png', np.abs(align_matrix)*256)


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


def alignment(net):
    weights = []
    for layer in net.parameters():
        W = layer
        if W.ndim == 2:
            weights.append(W)

    u1, d1, _ = np.linalg.svd(weights[-2].data.cpu().numpy())
    _, d2, v2 = np.linalg.svd(weights[-1].data.cpu().numpy())
    align_matrix = v2 @ u1

    return d1, d2, align_matrix


def init():
    W = np.random.random((16, 16))
    U, S, V = np.linalg.svd(W)
    S = np.arange(16)
    S = np.exp(-S * 0.35)
    S = np.diag(S)
    W = U @ S @ V
    return torch.Tensor(W)


if __name__ == '__main__':
    main()

