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
                self.get_contaminated(x, y, i, j)
        return y

    def get_contaminated(self, x, y, dest_row, dest_col):
        drows = [-1,-1,0,1,1,1,0,-1]
        dcols = [0,-1,-1,-1,0,1,1,1]
        neighbours = []
        for i in range(8):
            neighbour_row = dest_row + drows[i]
            neighbour_col = dest_col + dcols[i]

            if neighbour_row < 0 or neighbour_row >= self.rows:
                continue

            if neighbour_col < 0 or neighbour_col >= self.cols:
                continue
            neighbours.append((neighbour_row, neighbour_col))

        self.get_contaminate_by_neighbours(x, y, dest_row, dest_col, neighbours)

    def get_contaminate_by_neighbours(self, x, y, dest_row, dest_col, neighbours):
        count = len(neighbours)
        if count == 0:
            return

        summed = torch.zeros([y.shape[0],y.shape[1], self.kernel_size, self.kernel_size])
        for tup in neighbours:
            row = tup[0]
            col = tup[1]
            row_start = row * self.kernel_size
            row_end = row * self.kernel_size + self.kernel_size
            col_start = col * self.kernel_size
            col_end = col * self.kernel_size + self.kernel_size
            summed = summed[:,:] + x[:,:,row_start:row_end, col_start:col_end]
        summed = summed / count

        dest_row_start = dest_row * self.kernel_size
        dest_row_end = dest_row * self.kernel_size + self.kernel_size

        dest_col_start = dest_col * self.kernel_size
        dest_col_end = dest_col * self.kernel_size + self.kernel_size

        y[:,:,dest_row_start:dest_row_end, dest_col_start :dest_col_end] = \
            x[:,:,dest_row_start : dest_row_end, dest_col_start: dest_col_end] * 0.8 + summed[:,:] * 0.2
