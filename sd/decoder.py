import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    
    def __init__(self, in_channel: int):
        super().__init__()
        
        self.groupnorm = nn.GroupNorm(32, in_channel)
        self.attention = SelfAttention(1, in_channel)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channel, height, width)   
        residual = x
        
        n, c, h, w = x.shape
        # (batch_size, channel, height, width) -> (batch_size, features, height*width)
        x = x.view(n, c, h*w)
        
        # (batch_size, features, height*width) -> (batch_size, height*width, features)
        x = x.transform(-1, -2)
        
        # (batch_size, height*width, features) -> (batch_size, height*width, features)
        x = self.attention(x)
        
        # (batch_size, height*width, features) -> (batch_size, features, height*width)
        x = x.transpose(-1, -2)
        
        x = x.view(n, c, h, w)
        
        x += residual
        
        return x

class VAE_ResidualBlock(nn.Module):
    
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        
        self.groupnorm_1 = nn.GroupNorm(32, in_channel)
        self.conv_1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        
        self.groupnorm_2 = nn.GroupNorm(32, out_channel)
        self.conv_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        
        if in_channel == out_channel:
            self.shortcut = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channel, height, width)
        
        residual = x
        
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        
        return x + self.residual_layer(residual)
    
class VAE_Decoder(nn.Sequential):
    
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernal_size=1, padding=0),
            
            nn.conv2d(4, 512, kernal_size=3, padding=1),
            
            VAE_ResidualBlock(512, 512),
            
            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),
            
            # (batch_size, height/8, width/8) -> (batch_size, height/4, width/4)
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),            
            # (batch_size, height/4, width/4) -> (batch_size, height/2, width/2)        
            nn.Upsample(scale_factor=2), 
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),            
            # (batch_size, height/2, width/2) -> (batch_size, height, width)        
            nn.Upsample(scale_factor=2), 
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(256, 128),      
            VAE_ResidualBlock(128, 128),      
            VAE_ResidualBlock(128, 128),      

            nn.GroupNorm(32, 128),
            
            nn.SiLU(),
            
            # (batch_size, 128, height, widht) -> (batch_size, 3, height, width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 4, height/8, width/8)
        
        x /= 0.18215
        
        for module in self:
            x = module(x)

        return x