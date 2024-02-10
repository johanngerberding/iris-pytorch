from collections import namedtuple
from pathlib import Path 

import torch 
import torch.nn as nn 
from torchvision import models 

from .utils import get_ckpt_path, normalize_tensor, spatial_average


class LinLayer(nn.Module): 
    def __init__(self, channels_in: int, channels_out: int, dropout: bool = False) -> None: 
        super().__init__()
        layers = [nn.Dropout()] if (dropout) else []
        layers += [nn.Conv2d(channels_in, channels_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)

        
class VGG16(nn.Module): 
    def __init__(self) -> None:
        super().__init__()
        vgg16_pretrained_features = models.vgg16(pretrained=True).features
        self.num_sclices = 5  
        self.slice1 = nn.Sequential() 
        self.slice2 = nn.Sequential() 
        self.slice3 = nn.Sequential() 
        self.slice4 = nn.Sequential() 
        self.slice5 = nn.Sequential() 

        for i in range(4):
            self.slice1.add_module(str(i), vgg16_pretrained_features[i])
        for i in range(4, 9):
            self.slice2.add_module(str(i), vgg16_pretrained_features[i])
        for i in range(9, 16):
            self.slice3.add_module(str(i), vgg16_pretrained_features[i])
        for i in range(16, 23):
            self.slice4.add_module(str(i), vgg16_pretrained_features[i])
        for i in range(23, 30):
            self.slice5.add_module(str(i), vgg16_pretrained_features[i])

        for param in self.parameters():
            param.requires_grad = False 

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        h = self.slice1(x) 
        h_relu1_2 = h 
        h = self.slice2(h) 
        h_relu2_2 = h 
        h = self.slice3(h) 
        h_relu3_3 = h 
        h = self.slice4(h) 
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        outputs = namedtuple("vgg_outputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        out = outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class Scaling(nn.Module):
    def __init__(self) -> None: 
        super().__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.shift) / self.scale 


class PerceptualLoss(nn.Module):
    def __init__(self, dropout: bool = True) -> None:
        super().__init__()
        self.scaling = Scaling()
        self.channels = [64, 128, 256, 512, 512]
        self.net = VGG16()
        self.lin0 = LinLayer(self.channels[0], dropout=dropout)
        self.lin1 = LinLayer(self.channels[1], dropout=dropout)
        self.lin2 = LinLayer(self.channels[2], dropout=dropout)
        self.lin3 = LinLayer(self.channels[3], dropout=dropout)
        self.lin4 = LinLayer(self.channels[4], dropout=dropout)
        self.load_from_pretrained()
        for param in self.parameters(): 
            param.requires_grad = False 

    def load_from_pretrained(self) -> None:
        ckpt = get_ckpt_path(root=Path.home() / ".cache/iris/tokenizer_pretrained_vgg16")
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor: 
        in0_input, in1_input = (self.scaling(input), self.scaling(input))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {} 
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.channels)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2 
        
        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.channels))]
        val = res[0]

        for i in range(1, len(self.channels)):
            val += res[i]
        return val




