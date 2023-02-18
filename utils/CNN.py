import torch
from torchvision import resnet18
from utils.modules import v2iDownLayer

class CNN(torch.nn.Module):
    def __init__(self, inputDim, outputDim, num_downLayer=3):
        super().__init__()
        self.v2i = v2iDownLayer(inputDim, imgShape, num_downLayer)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 3, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.resnet = resnet18()
        self.resnet.fc = torch.nn.Linear(512, outputDim)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.v2i(x)
        out = self.conv(out)
        out = self.resnet(out)
        out = self.softmax(out)
        return out
