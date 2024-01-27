import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    
    def __init__(self):
        super().__init__(
            # (batch_size, Channel, height, width) -> (batch_size, 128, height, width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            # (batch_size, 128, height, widht) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),
            
            VAE_ResidualBlock(128, 128), 
            
            # (betch_size, 128, height, width) -> (batch_size, 128, height/2, width/2)
            nn.Conv2d(128, 128, kernel_size=3, padding=0, stride=2),
            
            VAE_ResidualBlock(128, 128),
            
            VAE_ResidualBlock(256, 256),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=0, stride=2),
            
            VAE_ResidualBlock(256, 512),
            
            VAE_ResidualBlock(512, 512),
            
            nn.Conv2d(512, 512, kernal_size=3, padding=0, stride=2),
            
            VAE_ResidualBlock(512, 512),
            
            VAE_ResidualBlock(512, 512),
            
            VAE_ResidualBlock(512, 512),
            
            VAE_AttentionBlock(512),
            
            VAE_ResidualBlock(512, 512),
            
            nn.GroupNorm(32, 512),
            
            nn.SiLU(),
            
            nn.Conv2d(512, 8, kernal_size=3, padding=1),
            
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )
        
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x : (Batch_size, channel, height, width)
        # noise: (Batch_size, out_channel, Height/8, width/8)
        
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # (padding_left, p_right, p_top, p_bottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        
        # (Batch_size, 8, heicht, height/8, width/8) -> 2 tensors of shape (batch_size, 4, heidht/8, width/8)
        mean, log_variance = torch.chunck(x, 2, dim=1)
        
        log_variance = torch.clamp(log_variance, min=-30., max=20.)
        varience = log_variance.exp()
        
        stdev = varience.sqrt()
        
        # Z = N(0, 1) -> N(mean, varince)?
        # X = mean + stdev * Z
        x = mean + stdev * noise
        
        # scale the output by a constant
        x *= 0.18215
        
        
        return x
        