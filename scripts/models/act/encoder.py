import torch
import torch.nn as nn


class ACTEncoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dims: int = 128,
        num_layers: int = 4,
        num_attention_heads: int = 8,
        num_img_tokens: int = 64,
        num_proprio_tokens: int = 1,
    ):
        super().__init__()

        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.num_img_tokens = num_img_tokens
        self.num_proprio_tokens = num_proprio_tokens
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dims,
            nhead=num_attention_heads,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            batch_first=True,
            norm_first=True
        )

        self.act_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> [image tokens + proprio tokens]
        # x -> [B, (64 + 1), 128]
        # print(f"[encoder.py] x.shape => {x.shape}")
        # print(f"self.num_image_tokens + self.num_proprio_tokens => {self.num_img_tokens + self.num_proprio_tokens}")
        
        if x.ndim != 3:
            raise ValueError(f"x.ndim != 3 : {x.ndim}")
        if x.shape[1] != self.num_proprio_tokens + self.num_img_tokens:
            raise ValueError(
                f"input tokens to ACT encoder != {self.num_img_tokens + self.num_proprio_tokens}: {x.shape[1]}"
            )
        if x.shape[2] != self.hidden_dims:
            raise ValueError(f"n_model != {self.hidden_dims} : {x.shape[2]}")

        x = self.act_encoder(x)

        if x.ndim != 3:
            raise ValueError(f"x.ndim != 3 : {x.ndim}")
        if x.shape[1] != self.num_proprio_tokens + self.num_img_tokens:
            raise ValueError(
                f"input tokens to ACT encoder != {self.num_img_tokens + self.num_proprio_tokens}: {x.shape[1]}"
            )
        if x.shape[2] != self.hidden_dims:
            raise ValueError(f"n_model != {self.hidden_dims} : {x.shape[2]}")

        return x
