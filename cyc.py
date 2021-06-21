from simple_net import SimpleNet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

def visTensor(tensors):
    for tensor in tensors:
        mean = torch.mean(tensor)
        tensor = tensor.data.clone()
        tensor[tensor >= mean] = 255
        tensor[tensor < mean] = 0
        np_array = tensor.numpy()
        plt.imshow(np_array)
        plt.show()

def train(model):
    NUM_EPOCHS = 1000
    BATCH_SIZE = 1

    working_set = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True,
    )

    dataloader = DataLoader(working_set, batch_size=BATCH_SIZE, shuffle=False)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    for epoch  in range(0, NUM_EPOCHS):
        for data, y_true in dataloader:
            optimizer.zero_grad()
            data = data.reshape(data.shape[0],data.shape[2], data.shape[3])
            y_pred = model(data)
            loss = F.nll_loss(y_pred, y_true)
            loss.backward()
            optimizer.step()
            print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
            break
    torch.save(model.state_dict(), 'models/cnn.h5')
    return model

model = SimpleNet()
filter = model.involution.kernels.data.clone()
visTensor(filter)
train(model)
filter = model.involution.kernels.data.clone()
visTensor(filter)



