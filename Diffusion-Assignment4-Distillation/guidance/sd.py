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

        
        t = torch.randint(self.min_step,self.max_step,size=(1,)).to(torch.long).to(self.device)
        alpha_bar = self.alphas[t].view(1,1,1,1)
        gt_noise = torch.randn_like(latents)
        noisy_latents = torch.sqrt(alpha_bar) * latents + torch.sqrt(1-alpha_bar) * gt_noise
        
        loss = torch.mean(torch.abs(self.get_noise_preds(noisy_latents,t,text_embeddings,guidance_scale)-gt_noise)**2)
        
        return loss * grad_scale
    
    
    def get_pds_loss(
        self, src_latents, tgt_latents, 
        src_text_embedding, tgt_text_embedding,
        guidance_scale=7.5, 
        grad_scale=1,
    ):
        
        B = src_latents.shape[0]
        
        t = torch.randint(self.min_step,self.max_step,size=(1,)).to(torch.long).to(self.device)

        gt_noise_t = torch.randn_like(src_latents)
        gt_noise_tm1 = torch.randn_like(src_latents)

        alpha_bar_t = self.alphas[t].view(B,1,1,1)
        alpha_bar_tm1 = self.alphas[torch.clamp(t-1,min=0)].view(B,1,1,1)


        def forward(x_0,alpha_bar,noise):
            return torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1-alpha_bar) * noise

        src_x_t = forward(src_latents,alpha_bar_t,gt_noise_t)
        src_x_tm1 = forward(src_latents,alpha_bar_tm1,gt_noise_tm1)

        tgt_x_t = forward(tgt_latents,alpha_bar_t)
        tgt_x_tm1 = forward(tgt_latents,alpha_bar_tm1)

        ## 이제 x_t-1과 x_t의 관계를 이용해서 z 식 구하기

        src_mu = self.get_noise_preds(src_x_t,t,src_text_embedding,guidance_scale) + torch.sqrt(alpha_bar_t) * src_latents
        src_z = (src_x_tm1-src_mu)/(alpha_bar_t)

        tgt_mu = self.get_noise_preds(tgt_x_t,t,tgt_text_embedding,guidance_scale) + torch.sqrt(alpha_bar_t) * tgt_latents
        tgt_z = (tgt_x_tm1-tgt_mu)/(alpha_bar_t)

        loss = torch.mean(torch.abs(src_z-tgt_z)**2)

        return loss * grad_scale
    
    
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
