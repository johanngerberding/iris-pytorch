from dataclasses import dataclass 
from typing import List, Dict, Tuple, Any
import torch 
import torch.nn as nn 
from perceptual_loss import PerceptualLoss

Batch = Dict[str, torch.Tensor]

@dataclass
class EncoderConfig:
    z_channels: int 


@dataclass 
class DecoderConfig:
    z_channels: int 


@dataclass
class TokenizerConfig: 
    resolution: int 
    vocab_size: int 
    embed_dim: int 
    encoder_cfg: EncoderConfig
    decoder_cfg: DecoderConfig


class LossWithIntermediateLosses:
    def __init__(self, **kwargs):
        self.loss_total = sum(kwargs.values())
        self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}
    
    def __truediv__(self, value):
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value 
        self.loss_total = self.loss_total / value
        return self


class Encoder(nn.Module): 
    def __init__(self, cfg: TokenizerConfig) -> None: 
        super().__init__()
        


class Decoder(nn.Module):
    def __init__(self, cfg: TokenizerConfig) -> None: 
        super().__init__()



class Tokenizer(nn.Module):
    def __init__(self, cfg: TokenizerConfig, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.vocab_size = cfg.vocab_size
        self.encoder = encoder 
        self.pre_quant_conv = nn.Conv2d(cfg.encoder_cfg.z_channels, cfg.embed_dim, 1)
        self.embedding = nn.Embedding(self.vocab_size, cfg.embed_dim)
        self.post_quant_conv = nn.Conv2d(cfg.embed_dim, cfg.decoder_cfg.z_channels, 1) 
        self.decoder = decoder
        self.embedding.weight.data.uniform_(-1.0 / self.vocab_size, 1.0 / self.vocab_size)
        self.perceptual_loss = PerceptualLoss()

    def forward(self, x: torch.Tensor, preprocess: bool = False, postprocess: bool = False) -> Tuple[torch.Tensor]:
        ...

    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:
        ...

    def encode(self, x: torch.Tensor, preprocess: bool = False) -> Tuple[torch.Tensor]: 
        ...
    
    def decode(self, z_q: torch.Tensor, postprocess: bool = False) -> torch.Tensor:
        ...