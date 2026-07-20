
import torch.nn as nn
import torch

class VisionEncoder(nn.Module):
    """3-layer CNN encoder: 64×64 RGB → 128-D embedding.

    Shape flow (B = batch, H = W = 64):

        Input          [B,   3, 64, 64]
        Conv block 1   [B,   3,  H,   W  ]  → [B,  32, 32, 32]
        Conv block 2   [B,  32, H/2, W/2]   → [B,  64, 16, 16]
        Conv block 3   [B,  64, H/4, W/4]   → [B,  128,  8,  8]
        Flatten        [B,  64,  8,  8]     → [B, 8192]
        Linear         [B, 8192]            → [B,  128]
    """
    
    def __init__(self, in_dims:int=3, out_dims:int=128): 
        super().__init__()
        # (64,64) => (32,32)
        self.conv1      = nn.Conv2d(in_channels=in_dims, 
                                    out_channels=32, 
                                    kernel_size=3, stride=2, padding=1)
        # (32,32) => (16,16)
        self.conv2      = nn.Conv2d(in_channels=32, 
                                    out_channels=64, 
                                    kernel_size=3, stride=2, padding=1)
        # (16,16) => (8,8)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)

        self.relu       = nn.ReLU()
        self.flatten    = nn.Flatten()
        self.embed      = nn.Linear(128 * 8 * 8, out_dims)

    def forward(self, x: torch.Tensor):
        
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.relu(x)

        # print(f"x.ndim => {x.ndim}")
        # print(f"x.shape => {x.shape}")

        x = self.flatten(x)

        # print(f"x.ndim => {x.ndim}")
        # print(f"x.shape => {x.shape}")

        x = self.embed(x)

        # print(f"x.ndim => {x.ndim}")
        # print(f"x.shape => {x.shape}")



# class ProprioEncoder(nn.Module):
#     """
#     :: input - [B, 9]
#     :: MLP [B,9] - [B,128]
#     :: output - [B,128]
#     """
    
#     def __init__(self):
    


# class VisionMLP(nn.Module): 
#     def __init__(): 
#         pass

    
