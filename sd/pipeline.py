import torch
import numpy as np  
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = WIDTH // 8
LATENT_HEIGHT = HEIGHT // 8

def generate(prompt: str, 
             uncond_prompt: str, # negetive prompt or empty string
             input_image=None, 
             strength=0.8, 
             do_cfg=True, 
             cfg_scale=7.5, 
             sampler="ddpm", 
             n_inference_steps=50, 
             models={}, seed = None, 
             device = None, 
             idle_device = None, 
             tokenizer=None ):
    with torch.no_grad():
        
        if not (0 < strength <= 1):
            raise ValueError("Strength must be in (0, 1]")
        if idle_device:
            to_idle: lambda x : x.to(idle_device)
        else:
            to_idle = lambda x: x
        
        generator = torch.Generator(device=device)
        if seed is None:
            seed = generator.seed()
        else:
            generator.manual_seed(seed)
        
        clip = models["clip"]
        clip.to(device)
        
        if do_cfg:
            # convert the prompt into tokens using the tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            # (batch_size, seq_len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, divice=device)
            # (batch_size, seq_len) -> (batch_size, seq_len, dim)
            cond_context = clip(cond_tokens)
            
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (batch_size, seq_len) -> (batch_size, seq_len, dim)
            uncond_context = clip(uncond_tokens)
            
            # (2, seq_len, dim) -> (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])
            
        else:
            # converrt it into a list of tokens
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (1, 77, 768)
            context = clip(tokens)
            
        to_idle(clip)
        
        if sampler == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler {sampler}")
        
        latents_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH) 
        
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)
            
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            # (h, w, ch)
            input_image_tensor = torch.tensor(input_image_tensor , dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (h, w, ch) -> (batch_size, h, w, ch)
            input_image_tesnor = input_image_tensor.unsqueeze(0)
            # (batch_size, h, w, ch) -> (batch_size, ch, h, w)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)
            
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            # run the image through the encoder of the VAE
            latents = encoder(input_image_tensor, encoder_noise)
            
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timestep[0])
            
            to_idle(encoder)
        else:
            # if we are doing text to image start with random noise
            latents = torch.randn(latents_shape, generator=generator, device=device)

        # 999 ... 0
        # 1000 900, 960, 940, 920, 900, 880 ... 0

        diffusion = models("diffusion")
        diffusion.to(device)
        
        
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)
            
            # (batch_size, 4, latents_height, latents_width)
            model_input = latents
            
            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_width) -> (2* Batch_size, 4, Latents_Height, Latents_width)
                model_input = model_input.request(2, 1, 1, 1)
                
                # model_output is the predicted noise by the UNET
                model_output = diffusion(model_input, time_embedding)
                
            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
                
            # remove noise predicted by the UNET
            latents = sampler.step(timestep, latents, model_output)    
            
            to_idle(diffusion)
            
            decoder = models["decoder"]
            decoder.to(device)
            
            images = decoder(latents)
            to_idle(decoder)
            
            images = rescale(images, (-1, 1), (0, 255), clamp = True)
            
            # (Batch_size, channel, height, width) -> (batch_size, heidht, width, channel)
            images = images.permute(0, 2, 3, 1)
            images = images.to("cpu", torch.uint8).numpy()
            return images[0]
        
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    freq = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32)/160)
    # (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    
    return torch.cat(torch.cos(x), torch.sin(x), dim=-1)
            
                
        
        
        
        