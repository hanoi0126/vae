import torch 
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.lr = nn.Linear(28*28, 300)
        self.lr2 = nn.Linear(300, 100)
        self.lr_ave = nn.Linear(100, z_dim) #mean
        self.lr_dev = nn.Linear(100, z_dim) #log(sigma^2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.lr(x)
        x = self.relu(x)
        x = self.lr2(x)
        x = self.relu(x)
        ave = self.lr_ave(x) #mean
        log_dev = self.lr_dev(x) #log(sigma^2)

        ep = torch.randn_like(ave)
        z = ave + torch.exp(log_dev / 2) * ep
        return z, ave, log_dev

class Decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.lr = nn.Linear(z_dim, 100)
        self.lr2 = nn.Linear(100, 300)
        self.lr3 = nn.Linear(300, 28*28)
        self.relu = nn.ReLU()
    
    def forward(self, z):
        x = self.lr(z)
        x = self.relu(x)
        x = self.lr2(x)
        x = self.relu(x)
        x = self.lr3(x)
        x = torch.sigmoid(x)
        return x

class VAE(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
    
    def forward(self, x):
        z, ave, log_dev = self.encoder(x)
        x = self.decoder(z)
        return x, z, ave, log_dev