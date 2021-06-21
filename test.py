from simple_net import SimpleNet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

def test():
    BATCH_SIZE = 100

    working_set = datasets.MNIST(
        root='data',
        train=False,
        transform=ToTensor(),
        download=True,
    )

    dataloader = DataLoader(working_set, batch_size=BATCH_SIZE, shuffle=True)
    model = SimpleNet()
    model.load_state_dict(torch.load("models/cnn.h5"))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, y_true in dataloader:
            y_pred = model(data)
            pred = torch.argmax(y_pred, dim=1, keepdim=True)
            correct += pred.eq(y_true.data.view_as(pred)).sum()
            total += 1

    print(f"{correct} correct among {len(working_set)}")

test()
exit(0)