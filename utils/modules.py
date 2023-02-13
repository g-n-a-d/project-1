import torch

class downLayer(torch.nn.Module):
    def __init__(self, inputDim):
        super().__init__()
        self.down = torch.nn.Sequential(
            torch.nn.Linear(inputDim, inputDim//2),
            torch.nn.BatchNorm1d(inputDim//2),
            torch.nn.ReLU()
        )

    def forward(self, x):
        out = self.down(x)
        return out
