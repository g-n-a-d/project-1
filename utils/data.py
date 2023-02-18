import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

class BODMAS(torch.utils.data.Dataset):
    def __init__(self, x, label):
        self.label = label
        self.ohe = OneHotEncoder(handle_unknown='ignore')
        self.ohe.fit(label)
        self.scaler = MinMaxScaler()
        self.scaler.fit(x)
        self.x = self.scaler.transform(x)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x_ = torch.Tensor(self.x[index])
        label_ = torch.Tensor(self.ohe.transform([[self.label[index]]]).toarray().reshape(-1,))
        return x_, label_
