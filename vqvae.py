from typing import Tuple 
import torch 
import torch.nn as nn 
from quantization import vq, vq_st

def weights_init(m):
    classname = m.__class__.__name__ 
    if classname.find('Conv') != -1:
        try: 
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x) 


class VQEmbedding(nn.Module):
    def __init__(self, embed_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=embed_size, 
            embedding_dim=embed_dim,
        )
        self.embedding.weight.data.uniform_(-1. / embed_size, 1. / embed_size)

    def forward(self, z_e_x: torch.Tensor) -> torch.Tensor:
        z_e_x = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x, self.embedding.weight)
        return latents  

    def straight_through(self, z_e_x: torch.Tensor) -> tuple: 
        z_e_x = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x, indices = vq_st(z_e_x, self.embedding.weight.detach())
        z_q_x = z_q_x.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight, dim=0, index=indices)
        z_q_x_bar = z_q_x_bar_flatten.view_as(z_e_x)
        z_q_x_bar = z_q_x_bar.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar


class VQVAE(nn.Module):
    def __init__(self, input_dim: int, dim: int, embed_size: int): 
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )
        self.codebook = VQEmbedding(embed_size=embed_size, embed_dim=dim)
        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim), 
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim), 
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh(),
        )
        self.apply(weights_init)


    def encode(self, x: torch.Tensor) -> torch.Tensor: 
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor: 
        # (B, D, H, W)
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)
        x_tilde = self.decoder(z_q_x)
        return x_tilde  
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e_x = self.encoder(x) 
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x 
    



def main():
    test = torch.randn((1, 3, 64, 64))
    print(test.size()) 
    model = VQVAE(input_dim=3, dim=64, embed_size=512)
    print(model)
    x_tilde, z_e_x, z_q_x = model(test)
    print(x_tilde.size())
    print(z_e_x.size())
    print(z_q_x.size())





if __name__ == "__main__":
    main() 
        
