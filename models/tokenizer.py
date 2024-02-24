from dataclasses import dataclass 
from typing import List, Dict, Tuple, Any
import torch 
import torch.nn as nn 
from perceptual_loss import PerceptualLoss

Batch = Dict[str, torch.Tensor]

@dataclass
class TokenizerConfig: 
    resolution: int 
    vocab_size: int 
    embed_dim: int 
    z_channels: int 
    ch_mult: List[int] 
    ch: int 
    in_channels: int 
    out_channels: int  
    num_res_blocks: int 
    dropout: float 
    attn_resolutions: List[int]
    

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


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int) -> None: 
        super().__init__()
        self.in_channels = in_channels 
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=1, 
            stride=1,
            padding=0,
        )
        self.k = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1, 
            stride=1,
            padding=0,
        )
        self.v = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1, 
            stride=1,
            padding=0,
        )
        self.proj_out = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x 
        h_ = self.norm(h_) 
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape 
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = nn.functional.softmax(w_, dim=2)
        
        v = v.reshape(b, c, h * w) 
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)

        return x + h_

class ResnetBlock(nn.Module): 
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            dropout: float, 
            conv_shortcut: bool = False, 
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


class Downsample(nn.Module):
    def __init__(self, block_in: int, with_conv: bool = True) -> None:
        super().__init__()        
        self.with_conv = with_conv 
        if self.with_conv: 
            # no asymmetric padding in torch conv, so do it yourself 
            self.conv = nn.Conv2d(
                in_channels=block_in,
                out_channels=block_in,
                kernel_size=3,
                stride=2,
                padding=0,
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        if self.with_conv: 
            pad = (0, 1, 0, 1)
            x = nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else: 
            x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x 


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool = True) -> None: 
        super().__init__()
        self.with_conv = with_conv 
        if self.with_conv: 
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv: 
            x = self.conv(x)
        return x


class Encoder(nn.Module): 
    def __init__(self, cfg: TokenizerConfig) -> None: 
        super().__init__()
        self.config = cfg
        self.num_resolutions = len(self.config.ch_mult)
        self.temb = 0 # timestep embedding channels 

        # downsampling 
        self.conv = nn.Conv2d(
            self.config.in_channels, 
            self.config.ch, 
            kernel_size=3, 
            stride=1,
            padding=1,
        ) 
        curr_resolution = cfg.resolution
        in_ch_mult = (1,) + tuple(self.config.ch_mult)
        block_in = None
        self.down = nn.ModuleList()

        for level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = self.config.ch * in_ch_mult[level]
            block_out = self.config.ch * self.config.ch_mult[level]

            for b in range(self.config.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in, 
                        out_channels=block_out, 
                        temb_channels=self.temb,
                        dropout=cfg.dropout
                    )
                )
                block_in = block_out
                if curr_resolution in cfg.attn_resolutions:
                    attn.append(AttnBlock(block_in))

            down = nn.Module()
            down.block = block 
            down.attn = attn 
            if level != self.num_resolutions - 1: 
                down.downsample = Downsample(block_in)
                curr_resolution = curr_resolution // 2 
            self.down.append(down)
        
        # middle 
        assert block_in is not None 
        self.mid = nn.Module() 
        self.mid.block1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb,
            dropout=cfg.dropout,
        )
        self.mid.attn1 = AttnBlock(block_in) 
        self.mid.block2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb,
            dropout=cfg.dropout,
        )

        # end 
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            in_channels=block_in,
            out_channels=self.config.z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )      

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        temb = None # timestep embedding 
        # downsampling 
        hs = [self.conv(x)]
        for level in range(self.num_resolutions): 
            for block in range(self.config.num_res_blocks):
                h = self.down[level].block[block](hs[-1], temb)
                if len(self.down[level].attn) > 0: 
                    h = self.down[level].attn[block](h)
                hs.append(h)
            
            if level != self.num_resolutions - 1: 
                hs.append(self.down[level].downsample(hs[-1]))

        # middle 
        h = hs[-1]
        h = self.mid.block1(h, temb)
        h = self.mid.attn1(h)
        h = self.mid.block2(h, temb)

        # end 
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h 


class Decoder(nn.Module):
    def __init__(self, cfg: TokenizerConfig) -> None: 
        super().__init__()
        self.config = cfg
        self.temb = 0 
        self.num_resolutions = len(self.config.ch_mult)
        # INFO this seems to be not needed?
        # compute in_ch_mult, block_in and curr_res at lowest res 
        # in_ch_mult = (1,) + tuple(self.config.ch_mult)
        block_in = self.config.ch * self.config.ch_mult[self.num_resolutions - 1]
        curr_res = self.config.resolution // 2 ** (self.num_resolutions - 1)
        print(f"Tokenizer: shape of latent is {self.config.z_channels, curr_res, curr_res}.")

        # z to block in 
        self.conv_in = nn.Conv2d(
            in_channels=self.config.z_channels,
            out_channels=block_in,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # middle 
        self.mid = nn.Module()
        self.mid.block1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb,
            dropout=self.config.dropout,
        )
        self.mid.attn1 = AttnBlock(block_in)
        self.mid.block2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb,
            dropout=self.config.dropout,
        )

        # upsampling 
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = cfg.ch * cfg.ch_mult[i_level]
            for _ in range(cfg.num_res_blocks + 1): 
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb, dropout=cfg.dropout))
                block_in = block_out
                if curr_res in cfg.attn_resolutions: 
                    attn.append(AttnBlock(block_in))

            up = nn.Module()
            up.block = block 
            up.attn = attn 
            if i_level != 0: 
                up.upsample = Upsample(block_in, with_conv=True)
                curr_res = curr_res * 2 
            self.up.insert(0, up) # prepend to get consistent order 

        # end 
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in, cfg.out_channels, kernel_size=3, stride=1, padding=1
        )            



def forward(self, x: torch.Tensor) -> torch.Tensor: 
    temb = None  # timestep embedding 
    # z to block in 
    h = self.conv_in(x)
    # middle 
    h = self.mid.block1(h, temb)
    h = self.mid.attn1(h)
    h = self.mid.block2(h, temb)

    # upsampling 
    for i_level in reversed(range(self.num_resolutions)):
        for i_block in range(self.cfg.num_res_blocks): 
            h = self.up[i_level].block[i_block](h, temb)
            if len(self.up[i_level].attn) > 0: 
                h = self.up[i_level].attn[i_block](h)
        
        if i_level != 0: 
            h = self.up[i_level].upsample(h)

    # end 
    h = self.norm_out(h)
    h = swish(h)
    h = self.conv_out(h)
    return h 

class Tokenizer(nn.Module):
    def __init__(self, cfg: TokenizerConfig, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.vocab_size = cfg.vocab_size
        self.encoder = encoder 
        self.pre_quant_conv = nn.Conv2d(encoder.config.z_channels, cfg.embed_dim, 1)
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