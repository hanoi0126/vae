import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.lr = nn.Linear(28*28, 300)
        self.lr2 = nn.Linear(300, 100)
        self.lr3 = nn.Linear(100, z_dim) #mean
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.lr(x)
        x = self.relu(x)
        x = self.lr2(x)
        x = self.relu(x)
        x = self.lr3(x)
        return x

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

class AE(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z
