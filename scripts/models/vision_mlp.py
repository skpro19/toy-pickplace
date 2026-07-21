
from huggingface_hub.inference._generated.types import visual_question_answering
import torch.nn as nn
import torch

from scripts.constant import ACTION_DIMS, PROPRIO_DIMS

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
    
    def __init__(self, * , in_dims:int=3, out_dims:int=128):
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
        self.conv3      = nn.Conv2d(in_channels=64, 
                                    out_channels=128, 
                                    kernel_size=3, stride=2, padding=1)

        self.relu       = nn.ReLU()
        self.flatten    = nn.Flatten()
        self.embed      = nn.Linear(128 * 8 * 8, out_dims)

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

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.flatten(x)

        x = self.embed(x)

        if x.ndim != 2:
            raise ValueError(f"x.ndim != 2: {x.ndim}")
        if x.shape[1] != 128:
            raise ValueError(f"x.shape[1] != 128: {x.shape[1]}")

        return x


class VisionMLP(nn.Module):
    """


    """

    def __init__(self):
        super().__init__()

        self.vision_encoder = VisionEncoder(in_dims=3, out_dims=128)
        self.backbone       = nn.Sequential(
                                nn.Linear(PROPRIO_DIMS + 128, 128),
                                nn.ReLU(),
                                nn.Linear(128,128),
                                nn.ReLU(),
                                nn.Linear(128,128),
                                nn.ReLU()
                            )
        self.joint_head     = nn.Linear(128, ACTION_DIMS - 1)
        self.gripper_head   = nn.Linear(128, 1)


    def forward(self, proprio_obs: torch.Tensor, img_obs: torch.Tensor):

        # print(f"proprio_obs.shape => {proprio_obs.shape}")
        # print(f"img_obs.shape => {img_obs.shape}")

        img_embeddings  = self.vision_encoder(img_obs)
        # print(f"img_embeddings.shape => {img_embeddings.shape}")

        combined        = torch.concat([proprio_obs, img_embeddings], axis=1)
        # print(f"combined.shape => {combined.shape}")

        combined        = self.backbone(combined)

        joint_logits    = self.joint_head(combined)
        gripper_logits  = self.gripper_head(combined)

        # print(f"joint_logits.shape => {joint_logits.shape}")
        # print(f"gripper_logits.shape => {gripper_logits.shape}")

        return (joint_logits, gripper_logits)

    
