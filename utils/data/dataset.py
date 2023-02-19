import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

class BODMAS(torch.utils.data.Dataset):
    def __init__(self, x, label):
        self.label = list(map(lambda w: [w], label))
        self.ohe = OneHotEncoder(handle_unknown='ignore')
        self.ohe.fit(self.label)
        self.scaler = MinMaxScaler()
        self.scaler.fit(x)
        self.x = self.scaler.transform(x)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x_ = torch.Tensor(self.x[index])
        label_ = torch.Tensor(self.ohe.transform([self.label[index]]).toarray().reshape(-1,))
        return x_, label_

class APISeq_TFIDF(torch.utils.data.Dataset):
    def __init__(self, x, label, tfidf, size=768):
        self.x = x
        self.label = list(map(lambda w: [w], label))
        self.tfidf = tfidf
        self.size = size
        self.ohe = OneHotEncoder(handle_unknown='ignore')
        self.ohe.fit(self.label)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x_ = []
        for token in self.x[index]:
            x_.append(self.tfidf[token])
        x_ = torch.Tensor(x_)
        if x_.shape[0] > self.size:
            x_ = x_[:self.size]
        elif x_.shape[0] < self.size:
            x_ = torch.cat((x_, torch.zeros(self.size - x_.shape[0])), dim=0)
        label_ = torch.Tensor(self.ohe.transform([self.label[index]]).toarray().reshape(-1,))
        return x_, label_

class APISeq_token(torch.utils.data.Dataset):
    def __init__(self, x, label, size=256):
        self.x = x
        self.label = list(map(lambda w: [w], label))
        self.size = size
        self.ohe = OneHotEncoder(handle_unknown='ignore')
        self.ohe.fit(self.label)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x_ = torch.Tensor(self.x[index]).int()
        if x_.shape[0] > self.size:
            x_ = x_[:self.size]
        elif x_.shape[0] < self.size:
            x_ = torch.cat((x_, torch.zeros(self.size - x_.shape[0]).int()), dim=0)
        label_ = torch.Tensor(self.ohe.transform([self.label[index]]).toarray().reshape(-1,))
        return x_, label_
