from dataclasses import dataclass 
from typing import List, Dict, Tuple, Any
import torch 
import torch.nn as nn 
from perceptual_loss import PerceptualLoss

Batch = Dict[str, torch.Tensor]

@dataclass
class EncoderConfig:
    z_channels: int 
    ch_mult: List[int]
    in_channels: int 
    out_channels: int 
    num_res_blocks: int     

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

def Normalize(in_channels: int) -> nn.Module: 
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def swish(x: torch.Tensor) -> torch.Tensor: 
    return x * torch.sigmoid(x)


class ResnetBlock(nn.Module): 
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            conv_shortcut: False, 
            dropout: float, 
            temb_channels: int = 512,
        ) -> None: 
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels 
        self.out_channels = out_channels 
        self.use_conv_shortcut = conv_shortcut 

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0: 
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout) 
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels: 
            if self.use_conv_shortcut: 
                self.conv_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )

            else: 
                self.nin_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor: 
        h = x 
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        if temb is not None: 
            h = h + self.temb_proj(swish(temb))[:, :, None, None]

        h = self.norm2(h)
        h = swish(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else: 
                x = self.nin_shortcut(x)
        
        return x + h 




class Encoder(nn.Module): 
    def __init__(self, cfg: TokenizerConfig) -> None: 
        super().__init__()
        self.config = cfg.encoder_cfg 
        self.num_resolutions = len(self.config.ch_mult)
        self.timestep_embedding_channels = 0 

        # downsampling 
        self.conv = nn.Conv2d(
            self.config.in_channels, 
            self.config.out_channels, 
            kernel_size=3, 
            stride=1,
            padding=1,
        ) 
        curr_resolution = cfg.resolution
        in_ch_mult = (1,) + tuple(self.config.ch_mult)
        self.down = nn.ModuleList()

        for level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = self.config.out_channels * in_ch_mult[level]
            block_out = self.config.out_channels * self.config.ch_mult[level]

            for b in range(self.config.num_res_blocks):
                block.append(

                )




class Decoder(nn.Module):
    def __init__(self, cfg: TokenizerConfig) -> None: 
        super().__init__()
        self.config = cfg.decoder_cfg 



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
        self.perceptual_loss = PerceptualLoss().eval()

    def forward(self, x: torch.Tensor, preprocess: bool = False, postprocess: bool = False) -> Tuple[torch.Tensor]:
        ...

    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:
        ...

    def encode(self, x: torch.Tensor, preprocess: bool = False) -> Tuple[torch.Tensor]: 
        ...
    
    def decode(self, z_q: torch.Tensor, postprocess: bool = False) -> torch.Tensor:
        ...
    

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """x is supposed to be channels first and in [0, 1]""" 
        return x.mul(2).sub(1)
    
    def postprocess_input(self, x: torch.Tensor) -> torch.Tensor: 
        """x is supposed to be channels first and in [-1, 1]""" 
        return x.add(1).div(2)