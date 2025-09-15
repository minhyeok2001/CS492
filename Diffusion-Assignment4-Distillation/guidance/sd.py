from diffusers import DDIMScheduler, StableDiffusionPipeline

import torch
import torch.nn as nn

import random


class StableDiffusion(nn.Module):
    def __init__(self, args, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = args.device
        self.dtype = args.precision
        print(f'[INFO] loading stable diffusion...')

        model_key = "stabilityai/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype,
        )

        pipe.to(self.device)
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype,
        )

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.t_range = t_range
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings
    
    
    def get_noise_preds(self, latents_noisy, t, text_embeddings, guidance_scale=100):
        latent_model_input = torch.cat([latents_noisy] * 2)
            
        tt = torch.cat([t] * 2)
        noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        
        return noise_pred

    def get_sds_loss(
        self, 
        latents,
        text_embeddings, 
        guidance_scale=25, 
        grad_scale=1,
    ):
        # TODO: Implement the loss function for SDS

        ## 일단 t는 timestep list에서 꺼내와야할거같고, gt는 ddim 식에 의해서 결정해야할거같은데
        ## 그러고나서 깨끗한 latents에 노이즈를 섞고 그걸 맞추는 식으로 구현하자

        #print(self.alphas) ## [0.9991 ... 0.0047]
        #print(self.alphas.shape)    ## torch.Size([1000])        

        #print(latents.shape) torch.Size([1, 4, 64, 64])

        t = torch.tensor([random.randint(0, 1000)]).to(torch.long).to(self.device)
        gt_noise = torch.randn_like(latents)
        noisy_latents = torch.sqrt(self.alphas[t]) * latents + torch.sqrt(1-self.alphas[t]) * gt_noise
        
        loss = torch.mean(torch.abs(self.get_noise_preds(noisy_latents,t,text_embeddings,guidance_scale)-gt_noise))
        
        return loss
        
    def get_gt_noise(self,):
        self.alphas 
    
    
    def get_pds_loss(
        self, src_latents, tgt_latents, 
        src_text_embedding, tgt_text_embedding,
        guidance_scale=7.5, 
        grad_scale=1,
    ):
        
        # TODO: Implement the loss function for PDS
        raise NotImplementedError("PDS is not implemented yet.")
    
    
    @torch.no_grad()
    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    @torch.no_grad()
    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents
