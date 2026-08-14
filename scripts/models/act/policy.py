import torch
import torch.nn as nn 

from models.act.vision_encoder import ACTVisionEncoder
from models.act.proprio_encoder import ACTProprioEncoder
from models.act.encoder import ACTEncoder
from models.act.decoder import ACTDecoder


class ACTPolicy(nn.Module): 
    def __init__(self,
        *, 
        hidden_dims:int = 128,
        imgH:int = 64, 
        imgW:int = 64, 
        obs_dims:int = 9, 
        num_img_tokens:int = 64, 
        num_proprio_tokens:int = 1,
        num_encoder_layers:int = 4, 
        num_attention_heads:int = 8,
        action_chunk_length: int = 10, 
        num_decoder_layers:int = 4,
        action_dims:int = 8
    ): 
        super().__init__()

        self.hidden_dims = hidden_dims
        self.imgH = imgH 
        self.imgW = imgW
        self.obs_dims = obs_dims
        self.num_img_tokens = num_img_tokens
        self.num_proprio_tokens= num_proprio_tokens
        self.num_encoder_layers = num_encoder_layers
        self.num_attention_heads = num_attention_heads
        self.k = action_chunk_length
        self.num_decoder_layers = num_decoder_layers
        self.action_dims = action_dims

        self.vision_encoder = ACTVisionEncoder(hidden_dims=self.hidden_dims, num_img_tokens=self.num_img_tokens)
        self.proprio_encoder = ACTProprioEncoder(hidden_dims=self.hidden_dims, obs_dims=self.obs_dims)
        self.act_encoder = ACTEncoder(
            hidden_dims = self.hidden_dims,
            num_layers= self.num_encoder_layers,
            num_attention_heads= self.num_attention_heads,
            num_img_tokens=self.num_img_tokens, 
            num_proprio_tokens=self.num_proprio_tokens
        )
        self.act_decoder = ACTDecoder(
            hidden_dims=self.hidden_dims, 
            num_layers=self.num_decoder_layers,
            num_attention_heads=self.num_attention_heads,
            num_query_slots=self.k, 
            num_proprio_tokens=self.num_proprio_tokens,
            num_img_tokens=self.num_img_tokens
        )

        self.decoder_queries = nn.Parameter(torch.randn(self.k, self.hidden_dims))
        self.action_head = nn.Linear(self.hidden_dims, self.action_dims)

    def forward(self, x: tuple):
        
        # if len(x) != 4:
        #     raise ValueError(f"len(x) != 4: {len(x)}")

        obs_proprio, _, obs_img, is_pad = x

        # if obs_img.ndim != 4: 
        #     raise ValueError(f"obs_img.ndim != 4: {obs_img.ndim}")
        # if obs_img.shape[1:] != (3, self.imgH, self.imgW):    
        #     raise ValueError(f"obs_img.shape[1:] != (3, {self.imgH}, {self.imgW}): {obs_img.shape[1:]}")
        
        img_tokens = self.vision_encoder(obs_img)

        # if img_tokens.ndim != 3: 
        #     raise ValueError(f"img_tokens.ndim != 3: {img_tokens.ndim}")
        # if img_tokens.shape[1:] != (self.num_img_tokens, self.hidden_dims): 
        #     raise ValueError(f"img_tokens.shape[1:] != ({self.num_img_tokens}, {self.hidden_dims}): {img_tokens.shape[1:]}")
        
        proprio_tokens = self.proprio_encoder(obs_proprio)

        # print(f"img_tokens.shape => {img_tokens.shape}")
        # print(f"proprio_tokens.shape => {proprio_tokens.shape}")

        # if proprio_tokens.ndim != 3: 
        #     raise ValueError(f"proprio_tokens.ndim != 3: {proprio_tokens.ndim}")
        # if proprio_tokens.shape[1:] != (self.num_proprio_tokens,self.hidden_dims): 
        #     raise ValueError(f"proprio_tokens.shape[1:] != ({self.num_proprio_tokens}, {self.hidden_dims}): {proprio_tokens.shape[1:]}")

        img_proprio_tokens = torch.concat((img_tokens, proprio_tokens), axis=1)

        # if img_proprio_tokens.shape[1:] != (self.num_img_tokens + self.num_proprio_tokens, self.hidden_dims):
        #     raise ValueError(f"img_proprio_tokens.shape[1:] != ({self.num_img_tokens + self.num_proprio_tokens}, {self.hidden_dims}): {img_proprio_tokens.shape[1:]}")

        # print(f"img_proprio_tokens.shape => {img_proprio_tokens.shape}")
        
        B = obs_proprio.shape[0]
        memory = self.act_encoder(img_proprio_tokens)
        tgt = self.decoder_queries.unsqueeze(0).expand(B, -1, -1)
        out = self.act_decoder(tgt, memory)
        actions = self.action_head(out)
        return actions