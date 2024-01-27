import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd:int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4* n_embd)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 320)
        
        x = self.linear_1(x)
        
        x = F.silu(x)
        
        x = self.linear_2(x)
        # (1, 1280)        
        return x
    
class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels: int, n_time = 1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_features == nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
        
        self.groupnorm_merge = nn.GroupNorm(32, out_channels)
        self.conv_mearge = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer == nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            
    def forward(self, feature, time):
        # feature: (batch_size, in_channels, height, width)
        # time (1, 1280)
        
        residual = feature
        
        feature = self.groupnorm_feature(feature)
        
        feature = F.silu(feature)
        
        feature = self.conv_features(feature)
        
        time = F.silu(time)
        
        time - self.linear_time(time)
        
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        merged = self.groupnorm_merge(merged)
        
        merged = F.silu(merged)
        
        merged = self.conv_merge(merged)
        
        return merged + self.residual_layer(residual)
    
class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernal_size=1, padding=0)
        
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_gelu_1 = nn.Linear(channels, 4* channels * 2)
        self.linear_gelu_2 = nn.Linear(4*channels, channels)
        
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
    def forward(self, x, context):
        # x: (Batch_size, Features, Heidht, width)
        # context : (batch_size, seq_len, dim)
        
        residual_long = x
        x = self.groupnorm(x)
        
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        
        # (Batch_size, features, Height, width) -> (batch_size, features, height*width)
        x = x.view(n, c, h*w)
        
        # (Batch_size, features, heaight*width) -> (batch_size, heiagh*width, features)
        x = x.transpose(-1, -2)
        
        # normalization + self attention with skip connection 
        
        residue_shaort = x
        
        x = self.layernorm_1(x)
        self.attention_1(x)
        x += residue_shaort
        residue_shaort = x
        
        # normalization + corss Attention with skip connection
        x = self.layernorm_2(x)
        
        # cross attention
        self.attention_2(x, context)
        
        x += residue_shaort
        
        residue_shaort = x
        # normalization + feed forward with geglu and skip connection
        x = self.layernorm_3(x)
        
        x, gate = self.layer_geglu_1(x).chunk(2, dim=-1)
        
        x = x * F.gelu(gate)
        
        x = self.linear_geglu_2(x)
        
        x += residue_shaort
        
        # (batch_size, height * width, features) -> (batch_size, features, height*width)
        
        x = x.transpose(-1, -2)
        
        x = x.view(n, c, h, w)
        
        return self.conv_output(x) + residual_long
        
        
    
class Upsample(nn.Module):
    
    def __int__(self, channels: int):
        super().__init__()
        
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x: (batch_size, Features, height, width) ->  (batch_size, Features, height * 2, width * 2)
        
        x = F.interpolate(x, scale_factor=2, mode = " nearest")
        return self.conv(x)

    
class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif self.isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class UNET(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.encoders = nn.Module([
            # (batch_size, 4, height/8, width/8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            #  (batch_size, 320, height/8, width/8) -> (batch_size, 320, height/16, width/16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, padding=1, stride=2)),
            
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            
            # (batch_size, 640, height/16, width/16) -> (batch_size, 640, height/32, width/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, padding=1, stride=2)),

            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            # (batch_size, 1280, height/32, width/32) -> (batch_size, 1280, height/64, width/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, padding=1, stride=2)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280)),

            # (batch_size, 1280, height/64, width/64) -> (batch_size, 1280, height/64, width/64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            
            
        ])
        
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            
            UNET_AttentionBlock(8, 160),
        
            UNET_ResidualBlock(1280, 1280),
        )
        
        self.decoder = nn.ModuleList([
            # (batch_size, 2560, height/64, width / 64) -> (batch_size, 1280, height/64, width/64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UpSample(1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), UpSample(1280)),

            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), UpSample(640)),
            
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ]
            
        )
        
class UNET_OutputLayer(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x: (Batch_size, 320, height/8, widht/8)
        
        x = self.groupnorm(x)
        
        x = F.silu(x)
        
        x = self.conv(x)
        
        # (batch_size, 4, height/8, width/8)
        
        return x

class Diffusion(nn.Module):
    
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
        
    def forward(self, latent: torch.tensor, context: torch.Tensor, time:torch.Tensor):
        # latent : (batch_size, 4, height / 8, widhth /8) 
        # context : (batch_size, seq_len, dim)
        # time : (1, 320)
        
        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        
        # (batch_size, 4, height/8, width/8) -> (batch_size, 320, height/8, width/8)
        output = self.unet(latent, context, time)
        
        # (batch_size, 320, height/8, width/8) -> (batch_size, 4, height/8, width/8)
        output = self.final(output)
        
        return output
        
        
