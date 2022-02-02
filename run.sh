#!/bin/bash
#SBATCH --job-name=1-20-directclr
#SBATCH --output=/checkpoint/ljng/latent-noise/cls-log/1-20-directclr.out
#SBATCH --partition=learnlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --constraint=volta32gb
#SBATCH --signal=USR1@60
#SBATCH --mem=480G
#SBATCH --open-mode=append
#SBATCH --time 2880

srun --label python linear_probe.py \
        --data /datasets01/imagenet_full_size/061417 \
        --pretrained /checkpoint/ljng/latent-noise/pretrained/1-16-directclr-resnet50.pth
