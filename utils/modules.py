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
    
class v2iDownLayer(torch.nn.Module):
    def __init__(self, inputDim, outputShape=(224, 224), numLayer=3):
        super().__init__()
        layerList = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Conv1d(1, inputDim//2, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(kernel_size=2)
        )])
        for i in range(1, numLayer):
            layerList.append(torch.nn.Sequential(
                torch.nn.Conv1d(inputDim//(2**i), inputDim//(2**(i + 1)), kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AvgPool1d(kernel_size=2)
            ))
        self.down = torch.nn.Sequential(*layerList)

    def forward(self, x):
        out = x.unsqueeze(dim=1)
        out = self.down(out)
        out = out.unsqueeze(dim=1)
        return out
