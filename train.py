from simple_net import SimpleNet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

def train():
    NUM_EPOCHS = 3
    BATCH_SIZE = 100

    working_set = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True,
    )

    dataloader = DataLoader(working_set, batch_size=BATCH_SIZE, shuffle=True)
    model = SimpleNet()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    counter = len(working_set) // BATCH_SIZE
    for epoch  in range(0, NUM_EPOCHS):
        pass_no =0
        for data, y_true in dataloader:
            optimizer.zero_grad()
            data = data.reshape(data.shape[0],data.shape[2], data.shape[3])
            y_pred = model(data)
            loss = F.nll_loss(y_pred, y_true)
            loss.backward()
            optimizer.step()
            pass_no = pass_no + 1
            print(f'Epoch:{epoch + 1} Pass:{pass_no} ({counter}), Loss:{loss.item():.4f}')
    torch.save(model.state_dict(), 'models/cnn.h5')
    return model

train()


