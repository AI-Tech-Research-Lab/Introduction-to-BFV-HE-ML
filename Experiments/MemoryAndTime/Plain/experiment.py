import time

from Pyfhel import Pyfhel

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import numpy as np

import logging as log

from memory_profiler import profile

device = 'cpu'

log.basicConfig(filename='experiments.log',
                format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S', encoding='utf-8', level=log.DEBUG)

class Square(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t):
        return torch.pow(t, 2)

transform = transforms.ToTensor()

test_set = torchvision.datasets.MNIST(
    root='../../data',
    train=False,
    download=True,
    transform=transform
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=1,
    shuffle=True
)

def forward_one_image(network, device):
    network.eval()

    with torch.no_grad():
        for batch in test_loader:  # Get Batch
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            preds = network(images)  # Pass Batch
            return


lenet1 = torch.load("../../LeNet1_Approx_single_square.pt")
lenet1.eval()
lenet1.to(device)

lenet1_singletanh = torch.load("../../LeNet1_single_tanh.pt")
lenet1_singletanh.eval()
lenet1_singletanh.to(device)

lenet1_singlesquare = torch.load("../../LeNet1_single_tanh.pt")
lenet1_singlesquare.eval()
lenet1_singlesquare.to(device)


# @profile
def experiment_LeNet1():
    forward_one_image(lenet1, device)


# @profile
def experiment_LeNet1_singletanh():
    forward_one_image(lenet1_singletanh, device)


# @profile
def experiment_LeNet1_singlesquare():
    forward_one_image(lenet1_singlesquare, device)


if __name__ == '__main__':

    log.info("Starting experiment...")
    starting_time = time.time()
    experiment_LeNet1()
    t = time.time() - starting_time
    log.info(f"The processing of one image for LeNet-1 required {t}")
    experiment_LeNet1_singletanh()
    t = time.time() - t - starting_time
    log.info(f"The processing of one image for LeNet-1 (single tanh) required {t}")
    experiment_LeNet1_singlesquare()
    t = time.time() - t - starting_time
    log.info(f"The processing of one image for approx LeNet-1 (single square) required {t}")



