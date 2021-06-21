from simple_net import SimpleNet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from torch import utils

def visTensor(tensors):
    for tensor in tensors:
        mean = torch.mean(tensor)
        tensor = tensor.data.clone()
        tensor[tensor >= mean] = 255
        tensor[tensor < mean] = 0
        np_array = tensor.numpy()
        plt.imshow(np_array)
        plt.show()


if __name__ == "__main__":
    model = SimpleNet()
    model.load_state_dict(torch.load("models/cnn.h5"))
    model.eval()
    layer = 1
    filter = model.involution.kernels.data.clone()
    visTensor(filter)