import torch
from torch import nn


class MLP(nn.Module):

    def __init__(self, *, 
                obs_dim: int=45, 
                action_dim: int=8, 
                ): 
        super().__init__()

        # print(f"MLP init!")
        # print(f"obs_dim=>{obs_dim} action_dim=>{action_dim}")
        # layers
        self.input_layer = nn.Linear(obs_dim, 128)
        # hidden layers
        self.h1 = nn.Linear(128, 128)
        self.h2 = nn.Linear(128, 128)
        self.h3 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, action_dim) 
        self.relu = nn.ReLU()
    

    def forward(self, x):

        # print(f"FORWARD!")
        x = self.input_layer(x)
        # print(f"M1")
        x = self.relu(x)
        # print(f"M2")
        x = self.h1(x)
        # print(f"M3")
        x = self.relu(x)
        # print(f"M4")
        x = self.h2(x)
        # print(f"M5")
        x = self.relu(x)
        # print(f"M6")
        x = self.h3(x)
        # print(f"M7")
        x = self.relu(x)
        # print(f"M8")

        x = self.output_layer(x)
        return x
