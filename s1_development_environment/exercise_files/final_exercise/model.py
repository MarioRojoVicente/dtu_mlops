import torch
import torch.nn.functional as F
from torch import nn


class MyAwesomeModel(nn.Module):
    """My awesome model."""
    def __init__(self, input=784, h1=256, h2=128, h3=64, output=10):
        super().__init__()
        self.fc1 = nn.Linear(input, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, output)
        self.drp = nn.Dropout(p=0.2)
        self.hidden_layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        # make sure input tensor is flattened
        x = torch.flatten(x)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.drp(F.relu(self.fc2(x)))
        x = self.drp(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=0)

        return x
