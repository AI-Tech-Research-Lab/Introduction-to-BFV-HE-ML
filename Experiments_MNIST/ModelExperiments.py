import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import logging as log
log.basicConfig(filename='ModelExperiments.log',
                format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S', encoding='utf-8', level=log.DEBUG)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.ToTensor()

train_set = torchvision.datasets.MNIST(
    root = './data',
    train=True,
    download=True,
    transform=transform
)

test_set = torchvision.datasets.MNIST(
    root = './data',
    train=False,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=50,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=50,
    shuffle=True
)

class Square(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t):
        return torch.pow(t, 2)


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def train_net(network, epochs, device):
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    for epoch in range(epochs):

        total_loss = 0
        total_correct = 0

        for batch in train_loader:  # Get Batch
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            preds = network(images)  # Pass Batch
            loss = F.cross_entropy(preds, labels)  # Calculate Loss

            optimizer.zero_grad()
            loss.backward()  # Calculate Gradients
            optimizer.step()  # Update Weights

            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)


def test_net(network, device):
    network.eval()
    total_loss = 0
    total_correct = 0

    with torch.no_grad():
        for batch in test_loader:  # Get Batch
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            preds = network(images)  # Pass Batch
            loss = F.cross_entropy(preds, labels)  # Calculate Loss

            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)

        accuracy = round(100. * (total_correct / len(test_loader.dataset)), 4)

    return total_correct / len(test_loader.dataset)


experiments = 10

# Initial LeNet-1
accuracies = []
for i in range(0, experiments):
    LeNet1 = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=5),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2),

        nn.Conv2d(4, 12, kernel_size=5),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2),

        nn.Flatten(),

        nn.Linear(192, 10),
    )

    LeNet1.to(device)
    train_net(LeNet1, 15, device)
    acc = test_net(LeNet1, device)
    accuracies.append(acc)

m = np.array(accuracies)
log.info(f"Results for LeNet-1:")
log.info(f"Mean accuracy on test set: {np.mean(m)}")
log.info(f"Var: {np.var(m)}")

# Optional: save the last trained LeNet-1:
torch.save(LeNet1, "LeNet1.pt")


# LeNet-1 with a single tanh
accuracies = []
for i in range(0, experiments):
    LeNet1_singletanh = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=5),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2),

        nn.Conv2d(4, 12, kernel_size=5),
        # nn.Tanh(),
        nn.AvgPool2d(kernel_size=2),

        nn.Flatten(),

        nn.Linear(192, 10),
    )

    LeNet1_singletanh.to(device)
    train_net(LeNet1_singletanh, 15, device)
    acc = test_net(LeNet1_singletanh, device)
    accuracies.append(acc)

m = np.array(accuracies)
log.info(f"Results for LeNet-1 (single tanh):")
log.info(f"Mean accuracy on test set: {np.mean(m)}")
log.info(f"Var: {np.var(m)}")

# Optional: save the last trained LeNet-1 (single tanh):
torch.save(LeNet1_singletanh, "LeNet1_single_tanh.pt")


# Approximated LeNet-1 (single square)
accuracies = []
for i in range(0, experiments):
    Approx_LeNet1 = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=5),
        Square(),
        nn.AvgPool2d(kernel_size=2),

        nn.Conv2d(4, 12, kernel_size=5),
        # nn.Tanh(),
        nn.AvgPool2d(kernel_size=2),

        nn.Flatten(),

        nn.Linear(192, 10),
    )

    Approx_LeNet1.to(device)
    train_net(Approx_LeNet1, 15, device)
    acc = test_net(Approx_LeNet1, device)
    accuracies.append(acc)

m = np.array(accuracies)
log.info(f"Results for approximated LeNet-1 (single square):")
log.info(f"Mean accuracy on test set: {np.mean(m)}")
log.info(f"Var: {np.var(m)}")

# Optional: save the last trained approximated LeNet-1:
torch.save(Approx_LeNet1, "LeNet1_Approx_single_square.pt")

# Approximated LeNet-1 (single square) - the one saved and used by the encrypted processing
model = torch.load("LeNet1_Approx_single_square.pt")
model.eval()
model.to(device)
acc = test_net(model, device)
log.info(f"Results for approximated LeNet-1 (single square) - the one saved to file: {acc}")





