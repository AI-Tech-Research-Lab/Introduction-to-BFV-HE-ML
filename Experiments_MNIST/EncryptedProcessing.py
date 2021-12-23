from functools import reduce

from Pyfhel import Pyfhel

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import numpy as np

import logging as log
log.basicConfig(filename='experiments.log',
                format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S', encoding='utf-8', level=log.DEBUG)

transform = transforms.ToTensor()

test_set = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

model_file = "../LeNet1_Approx_single_square.pt"
log.info(f"Loading model from file {model_file}...")


class Square(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t):
        return torch.pow(t, 2)

model = torch.load(model_file)
model.eval()

log.info(model)


# Code for matrix encoding/encryption
def encode_matrix(HE, matrix):
    try:
        return np.array(list(map(HE.encodeFrac, matrix)))
    except TypeError:
        return np.array([encode_matrix(HE, m) for m in matrix])


def decode_matrix(HE, matrix):
    try:
        return np.array(list(map(HE.decodeFrac, matrix)))
    except TypeError:
        return np.array([decode_matrix(HE, m) for m in matrix])


def encrypt_matrix(HE, matrix):
    try:
        return np.array(list(map(HE.encryptFrac, matrix)))
    except TypeError:
        return np.array([encrypt_matrix(HE, m) for m in matrix])


def decrypt_matrix(HE, matrix):
    try:
        return np.array(list(map(HE.decryptFrac, matrix)))
    except TypeError:
        return np.array([decrypt_matrix(HE, m) for m in matrix])


# Code for encoded CNN
class ConvolutionalLayer:
    def __init__(self, HE, weights, stride=(1, 1), padding=(0, 0), bias=None):
        self.HE = HE
        self.weights = encode_matrix(HE, weights)
        self.stride = stride
        self.padding = padding
        self.bias = bias
        if bias is not None:
            self.bias = encode_matrix(HE, bias)

    def __call__(self, t):
        t = apply_padding(t, self.padding)
        result = np.array([[np.sum([convolute2d(image_layer, filter_layer, self.stride)
                                    for image_layer, filter_layer in zip(image, _filter)], axis=0)
                            for _filter in self.weights]
                           for image in t])

        if self.bias is not None:
            return np.array([[layer + bias for layer, bias in zip(image, self.bias)] for image in result])
        else:
            return result


def convolute2d(image, filter_matrix, stride):
    x_d = len(image[0])
    y_d = len(image)
    x_f = len(filter_matrix[0])
    y_f = len(filter_matrix)

    y_stride = stride[0]
    x_stride = stride[1]

    x_o = ((x_d - x_f) // x_stride) + 1
    y_o = ((y_d - y_f) // y_stride) + 1

    def get_submatrix(matrix, x, y):
        index_row = y * y_stride
        index_column = x * x_stride
        return matrix[index_row: index_row + y_f, index_column: index_column + x_f]

    return np.array(
        [[np.sum(get_submatrix(image, x, y) * filter_matrix) for x in range(0, x_o)] for y in range(0, y_o)])

def apply_padding(t, padding):
    y_p = padding[0]
    x_p = padding[1]
    zero = t[0][0][y_p+1][x_p+1] - t[0][0][y_p+1][x_p+1]
    return [[np.pad(mat, ((y_p, y_p), (x_p, x_p)), 'constant', constant_values=zero) for mat in layer] for layer in t]


class LinearLayer:
    def __init__(self, HE, weights, bias=None):
        self.HE = HE
        self.weights = encode_matrix(HE, weights)
        self.bias = bias
        if bias is not None:
            self.bias = encode_matrix(HE, bias)

    def __call__(self, t):
        result = np.array([[np.sum(image * row) for row in self.weights] for image in t])
        if self.bias is not None:
            result = np.array([row + self.bias for row in result])
        return result


class SquareLayer:
    def __init__(self, HE):
        self.HE = HE

    def __call__(self, image):
        return square(self.HE, image)


def square(HE, image):
    try:
        return np.array(list(map(lambda x: HE.power(x, 2), image)))
    except TypeError:
        return np.array([square(HE, m) for m in image])


class FlattenLayer:
    def __call__(self, image):
        dimension = image.shape
        return image.reshape(dimension[0], dimension[1]*dimension[2]*dimension[3])


class AveragePoolLayer:
    def __init__(self, HE, kernel_size, stride=(1, 1), padding=(0, 0)):
        self.HE = HE
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, t):
        t = apply_padding(t, self.padding)
        return np.array([[_avg(self.HE, layer, self.kernel_size, self.stride) for layer in image] for image in t])


def _avg(HE, image, kernel_size, stride):
    x_s = stride[1]
    y_s = stride[0]

    x_k = kernel_size[1]
    y_k = kernel_size[0]

    x_d = len(image[0])
    y_d = len(image)

    x_o = ((x_d - x_k) // x_s) + 1
    y_o = ((y_d - y_k) // y_s) + 1

    denominator = HE.encodeFrac(1 / (x_k * y_k))

    def get_submatrix(matrix, x, y):
        index_row = y * y_s
        index_column = x * x_s
        return matrix[index_row: index_row + y_k, index_column: index_column + x_k]

    return [[np.sum(get_submatrix(image, x, y)) * denominator for x in range(0, x_o)] for y in range(0, y_o)]


# We can now define a function to "convert" a PyTorch model to a list of sequential HE-ready-to-be-used layers:
def build_from_pytorch(HE, net):
    # Define builders for every possible layer

    def conv_layer(layer):
        if layer.bias is None:
            bias = None
        else:
            bias = layer.bias.detach().numpy()

        return ConvolutionalLayer(HE, weights=layer.weight.detach().numpy(),
                                  stride=layer.stride,
                                  padding=layer.padding,
                                  bias=bias)

    def lin_layer(layer):
        if layer.bias is None:
            bias = None
        else:
            bias = layer.bias.detach().numpy()
        return LinearLayer(HE, layer.weight.detach().numpy(),
                           bias)

    def avg_pool_layer(layer):
        # This proxy is required because in PyTorch an AvgPool2d can have kernel_size, stride and padding either of
        # type (int, int) or int, unlike in Conv2d
        kernel_size = (layer.kernel_size, layer.kernel_size) if isinstance(layer.kernel_size, int) else layer.kernel_size
        stride = (layer.stride, layer.stride) if isinstance(layer.stride, int) else layer.stride
        padding = (layer.padding, layer.padding) if isinstance(layer.padding, int) else layer.padding

        return AveragePoolLayer(HE, kernel_size, stride, padding)

    def flatten_layer(layer):
        return FlattenLayer()

    def square_layer(layer):
        return SquareLayer(HE)

    # Maps every PyTorch layer type to the correct builder
    options = {"Conv": conv_layer,
               "Line": lin_layer,
               "Flat": flatten_layer,
               "AvgP": avg_pool_layer,
               "Squa": square_layer
               }

    encoded_layers = [options[str(layer)[0:4]](layer) for layer in net]
    return encoded_layers


log.info(f"Run the experiments...")

from joblib import Parallel, delayed, parallel_backend
import time
n_threads = 8

log.info(f"I will use {n_threads} threads.")

p = 953983721
m = 4096

log.info(f"Using encryption parameters: m = {m}, p = {p}")

HE = Pyfhel()
HE.contextGen(p=p, m=m)
HE.keyGen()
relinKeySize=3
HE.relinKeyGen(bitCount=5, size=relinKeySize)

model.to("cpu")
model_encoded = build_from_pytorch(HE, model)


experiment_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=n_threads,
    shuffle=True
)


def enc_and_process(image):
    encrypted_image = encrypt_matrix(HE, image.unsqueeze(0).numpy())
    
    for layer in model_encoded:
        encrypted_image = layer(encrypted_image)
    
    result = decrypt_matrix(HE, encrypted_image)
    return result
    

def check_net():
    total_correct = 0
    n_batch = 0
        
    for batch in experiment_loader:
        images, labels = batch
        with parallel_backend('multiprocessing'):
            preds = Parallel(n_jobs=n_threads)(delayed(enc_and_process)(image) for image in images)

        preds = reduce(lambda x, y: np.concatenate((x, y)), preds)
        preds = torch.Tensor(preds)

        for image in preds:
            for value in image:
                if value > 100000:
                    log.warning("WARNING: probably you are running out of NB.")

        total_correct += preds.argmax(dim=1).eq(labels).sum().item()
        n_batch = n_batch + 1
        if n_batch % 5 == 0 or n_batch == 1:
            log.info(f"Done {n_batch} batches.")
            log.info(f"This means we processed {n_threads * n_batch} images.")
            log.info(f"Correct images for now: {total_correct}")
            log.info("---------------------------")

    return total_correct


starting_time = time.time()

log.info(f"Start experiment...")

correct = check_net()

total_time = time.time() - starting_time
log.info(f"Total corrects on the entire test set: {correct}")
log.info("Time: ", total_time)







