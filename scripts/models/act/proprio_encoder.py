import torch 
import torch.nn as nn

class ACTProprioEncoder(nn.Module):
    """ raw proprio observations (B, 9) => tokens (B, 1, 128) """ 
    
    def __init__(self, * , hidden_dims:int = 128, obs_dims:int = 9): 
        super().__init__()

        self.hidden_dims = hidden_dims
        self.obs_dims = obs_dims

        self.projection = nn.Linear(self.obs_dims, self.hidden_dims)
        
    def forward(self, x: torch.Tensor): 
        if x.ndim != 2: 
            raise ValueError(f"x.ndim != 2 : {x.ndim}")
        if x.shape[1] != self.obs_dims:
            raise ValueError(f"x.shape[1] != {self.obs_dims}: {x.shape[1]}")

        x = self.projection(x)
        x = x.unsqueeze(1)

        if x.ndim != 3: 
            raise ValueError(f"x.ndim != 3 : {x.ndim}")
        if x.shape[1] != 1: 
            raise ValueError(f"x.shape[1] != 1: {x.shape[1]}")
        if x.shape[2] != self.hidden_dims: 
            raise ValueError(f"x.shape[2] != {self.hidden_dims}: {x.shape[2]}")

        return x
        