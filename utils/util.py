import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, arr_features, arr_label, device):
        self.x = torch.tensor(arr_features).to(device)
        self.y = torch.tensor(arr_label).to(device)

    def __getitem__(self, index):
        return self.x[index, :], self.y[index]

    def __len__(self):
        return len(self.y)