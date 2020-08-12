import torch
from torch import nn


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size,  num_layers, num_classes, device):
        super(GRU, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers = self.num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.gru(x, h0)
        #out = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        #out = self.fc(out)[:, -1]

        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(1, 10), nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Linear(10, 7), nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Linear(7, 3), nn.LeakyReLU())
        self.layer4 = nn.Sequential(nn.Linear(3, 1), nn.LeakyReLU())

    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out


