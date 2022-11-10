import torch

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        dim_list = [output_dim]
        while 2*dim_list[-1] < input_dim:
            dim_list.append(2*dim_list[-1])
        dim_list.append(input_dim)
        dim_list.reverse()
        module_list = []
        for i in range(len(dim_list) - 1):
            module_list.append(torch.nn.Linear(dim_list[i], dim_list[i + 1]))
            if i < len(dim_list) - 2:
                module_list.append(torch.nn.ReLU())
        self.net = torch.nn.ModuleList(module_list)

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        dim_list = [input_dim]
        while 2*dim_list[-1] < output_dim:
            dim_list.append(2*dim_list[-1])
        dim_list.append(output_dim)
        module_list = []
        for i in range(len(dim_list) - 1):
            module_list.append(torch.nn.Linear(dim_list[i], dim_list[i + 1]))
            if i < len(dim_list) - 2:
                module_list.append(torch.nn.ReLU())
        self.net = torch.nn.ModuleList(module_list)

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x

class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.parameters() , lr=1e-9)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        
    def learn(self, data_loader, learning_rate=1e-9, epochs=7, quiet=False):
        self.optimizer.param_groups[0]['lr'] = learning_rate
        self.train()
        for i in range(epochs):
            for ii, (x_train, y_train) in enumerate(data_loader):
                self.optimizer.zero_grad()
                loss = self.criterion(self(x_train), x_train)
                loss.backward()
                self.optimizer.step()
                if not quiet:
                    print('epoch: {}, batch: {}/{} -------> loss: {}'.format(i, ii + 1, len(data_loader), loss))

    def encode(self, x):
        self.eval()
        return self.encoder(x)