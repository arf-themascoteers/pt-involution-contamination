import torch.nn as nn
import torch


class Contamination(nn.Module):
    def __init__(self, kernel_size=4, channel=3, height = 28, width = 28):
        super(Contamination, self).__init__()
        self.kernel_size = kernel_size
        self.channel = channel
        self.height = height
        self.width = width
        self.rows = height // kernel_size
        self.cols = width // kernel_size
        self.dummy = torch.tensor([0.1])
        self.myparameters = nn.Parameter(self.dummy)

    def forward(self, x):
        y = x.clone()
        for i in range(self.rows):
            for j in range(self.cols):
                self.contaminate_from_source(x,y,i,j)
        return y

    def contaminate_from_source(self, x, y, src_row , src_col):
        drows = [-1,-1,0,1,1,1,0,-1]
        dcols = [0,-1,-1,-1,0,1,1,1]

        for i in range(8):
            neighbour_row = src_row + drows[i]
            neighbour_col = src_col + dcols[i]

            if neighbour_row < 0 or neighbour_row >= self.rows:
                continue

            if neighbour_col < 0 or neighbour_col >= self.cols:
                continue

            self.contaminate_from_source_to_dest(x,y,src_row, src_col, neighbour_row, neighbour_col)

    def contaminate_from_source_to_dest(self, x,y, src_row, src_col, dest_row, dest_col):
        y[:,:,dest_row, dest_col] = y[:,:,dest_row, dest_col] + x[:,:,src_row, src_col] * 0.2
