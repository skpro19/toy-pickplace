import torch
from torch import nn


class MLP(nn.Module):

    def __init__(self, *, 
                obs_dim: int=40, 
                action_dim: int=8, 
                ): 
        super().__init__()

        # layers
        self.input_layer = nn.Linear(obs_dim, 128)
        # hidden layers
        self.h1 = nn.Linear(128, 128)
        self.h2 = nn.Linear(128, 128)
        self.h3 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, action_dim) 
        self.relu = nn.ReLU()
    

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        
        x = self.h1(x)
        x = self.relu(x)

        x = self.h2(x)
        x = self.relu(x)

        x = self.h3(x)
        x = self.relu(x)

        x = self.output_layer(x)
        return x
