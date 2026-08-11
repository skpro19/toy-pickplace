import torch.nn as nn
import torch

class ACTVisionEncoder(nn.Module): 

    def __init__(self, *, hidden_dims=128):
        
        super().__init__() 
        self.hiddem_dims = hidden_dims

        self.conv1 = nn.Conv2d(in_channels=3, 
                            out_channels=32, 
                            kernel_size=3, 
                            stride = 2, 
                            padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=32, 
                            out_channels=64, 
                            kernel_size=3, 
                            stride = 2, 
                            padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, 
                            out_channels=hidden_dims, 
                            kernel_size=3, 
                            stride = 2, 
                            padding=1)
        
        self.relu = nn.ReLU()
        

    def forward(self, x: torch.Tensor): 
        if x.ndim != 4:
            raise ValueError(f"x.ndim != 4: {x.ndim}")
        if x.shape[1] != 3:
            raise ValueError(f"x.shape[1] != 3: {x.shape[1]}")
        if x.shape[2] != 64:
            raise ValueError(f"x.shape[2] != 64: {x.shape[2]}")
        if x.shape[3] != 64:
            raise ValueError(f"x.shape[3] != 64: {x.shape[3]}")
        
        x = self.conv1(x)
        x = self.relu(x)
        print(f"x.shape => {x.shape}")
        
        x = self.conv2(x)
        x = self.relu(x)
        print(f"x.shape => {x.shape}")
        
        x = self.conv3(x)
        x = self.relu(x)
        print(f"x.shape => {x.shape}")
        
        B, _, H, W = x.shape
        x = x.permute(0,2,3,1).reshape(B, H * W, self.hiddem_dims)

        if x.shape[-1] != self.hiddem_dims:
            raise ValueError(f"hidden_dims!= {self.hiddem_dims}: {x.shape[-1]}")

        return x

    
