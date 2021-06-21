import torch.nn as nn
import torch


class Involution(nn.Module):
    def __init__(self, output_channel=20, height = 28, width = 28):
        super(Involution, self).__init__()
        self.output_channel = output_channel
        self.height = height
        self.width = width
        self.kernels = torch.rand(output_channel, height, width)
        self.myparameters = nn.Parameter(self.kernels)

    def forward(self, x):
        y = torch.ones([x.shape[0],self.output_channel, self.height, self.width])
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                y[i][j] = x[i] * self.kernels[j]
        return y