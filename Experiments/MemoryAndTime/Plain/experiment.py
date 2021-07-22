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

n_experiments = 1000


# @profile
def experiment_LeNet1():
	for i in range(0, n_experiments):
		forward_one_image(lenet1, device)


# @profile
def experiment_LeNet1_singletanh():
	for i in range(0, n_experiments):
		forward_one_image(lenet1_singletanh, device)


# @profile
def experiment_LeNet1_singlesquare():
	for i in range(0, n_experiments):
		forward_one_image(lenet1_singlesquare, device)


if __name__ == '__main__':

    log.info("Starting experiment...")
    starting_time = time.time()
    experiment_LeNet1()
    t1 = time.time()
    log.info(f"The processing of one image for LeNet-1 required {(t1-starting_time)/n_experiments} seconds")
    experiment_LeNet1_singletanh()
    t2 = time.time()
    log.info(f"The processing of one image for LeNet-1 (single tanh) required {(t2-t1)/n_experiments} seconds")
    experiment_LeNet1_singlesquare()
    t3 = time.time()
    log.info(f"The processing of one image for approx LeNet-1 (single square) required {(t3-t2)/n_experiments} seconds")



