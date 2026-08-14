import torch.nn as nn
import torch

class ACTDecoder(nn.Module):
    def __init__(
        self, 
        *, 
        hidden_dims:int = 128,
        num_layers:int = 4,
        num_attention_heads:int =8,
        num_query_slots:int = 10, 
        num_proprio_tokens:int = 1,
        num_img_tokens:int = 64,
    ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.num_query_slots = num_query_slots
        self.num_proprio_tokens = num_proprio_tokens
        self.num_img_tokens = num_img_tokens
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dims,
            nhead=num_attention_heads, 
            dim_feedforward=512, 
            activation="relu", 
            dropout=0.1, 
            batch_first=True, 
            norm_first=True
        )

        self.act_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=self.num_layers
        )


    def forward(self, tgt: torch.Tensor, memory: torch.Tensor): 
        
        if memory.ndim != 3: 
            raise ValueError(f"memory.ndim != 3: {memory.ndim}")
        if memory.shape[1] != self.num_proprio_tokens + self.num_img_tokens: 
            raise ValueError(f"memory.shape[1] != {self.num_proprio_tokens + self.num_img_tokens}: {memory.shape[1]}")
        if memory.shape[2] != self.hidden_dims: 
            raise ValueError(f"memory.shape[2] != {self.hidden_dims}: {memory.shape[2]}")
        
        if tgt.ndim != 3: 
            raise ValueError(f"tgt.ndim != 3: {tgt.ndim}")
        if tgt.shape[1] != self.num_query_slots: 
            raise ValueError(f"tgt.shape[1] != {self.num_query_slots}: {tgt.shape[1]}")
        if tgt.shape[2] != self.hidden_dims: 
            raise ValueError(f"tgt.shape[2] != {self.hidden_dims}: {tgt.shape[2]}")


        tgt = self.act_decoder(tgt, memory)

        if tgt.ndim != 3: 
            raise ValueError(f"tgt.ndim != 3: {tgt.ndim}")
        if tgt.shape[1] != self.num_query_slots: 
            raise ValueError(f"tgt.shape[1] != {self.num_query_slots}: {tgt.shape[1]}")
        if tgt.shape[2] != self.hidden_dims: 
            raise ValueError(f"tgt.shape[2] != {self.hidden_dims}: {tgt.shape[2]}")

        return tgt
