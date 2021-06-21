import torch.nn as nn
import torch.nn.functional as F
from involution import Involution
from contamination import Contamination


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # self.net = nn.Sequential(
        #     Involution(),
        #     Contamination(),
        #     nn.LeakyReLU(),
        #     nn.Flatten(),
        #     nn.Linear(3 * 28 * 28, 10)
        # )

        self.involution = Involution()
        self.contamination = Contamination()
        self.linear = nn.Linear(5 * 28 * 28, 10)

        # self.flatten = nn.Flatten()
        # self.linear = nn.Linear(28 * 28, 100)
        # self.linear2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.involution(x)
        x = self.contamination(x)
        x = F.leaky_relu(x)
        x = x.reshape(x.shape[0],-1)
        x = self.linear(x)

        # x = self.flatten(x)
        # x = self.linear(x)
        # x = F.leaky_relu(x)
        # x = self.linear2(x)
        return F.log_softmax(x, dim=1)