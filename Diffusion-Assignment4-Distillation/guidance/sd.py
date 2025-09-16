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

        t = torch.randint(self.min_step,self.max_step,size=(1,)).to(torch.long).to(self.device)
        alpha_bar = self.alphas[t].view(1,1,1,1)
        gt_noise = torch.randn_like(latents)
        noisy_latents = torch.sqrt(alpha_bar) * latents + torch.sqrt(1-alpha_bar) * gt_noise
        
        with torch.no_grad():                                   
          eps_hat = self.get_noise_preds(noisy_latents, t, text_embeddings, guidance_scale)
        
        ## 헐. 이렇게 안해주면 결국 U-NET 야코비안이 backward시에 자동생성되네? 빼주고싶어도?

        w = 1 - alpha_bar
        grad = w * (eps_hat - gt_noise) 

        ## 아래에서 최종 loss 정리

        loss = (noisy_latents * grad).mean() * grad_scale

        #loss = torch.mean((eps_hat.detach() - gt_noise)**2)
        return loss
    


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

        tgt_x_t = forward(tgt_latents,alpha_bar_t,gt_noise_t)
        tgt_x_tm1 = forward(tgt_latents,alpha_bar_tm1,gt_noise_tm1)

        alpha_t  = (alpha_bar_t / alpha_bar_tm1).clamp(min=1e-12, max=1-1e-12)
        beta_t   = (1.0 - alpha_t).clamp(min=1e-12)
        sigma2_t = ((1.0 - alpha_bar_tm1) / (1.0 - alpha_bar_t)) * beta_t
        sigma_t  = torch.sqrt(sigma2_t + 1e-20)

        ## 이제 x_t-1과 x_t의 관계를 이용해서 z 식 구하기
        with torch.no_grad():
          src_noise = self.get_noise_preds(src_x_t,t,src_text_embedding,guidance_scale)

        src_x0 = (src_x_t - torch.sqrt(1-alpha_bar_t)*src_noise)/torch.sqrt(alpha_bar_t)
        src_mu = ((torch.sqrt(alpha_bar_tm1) * beta_t / (1.0 - alpha_bar_t)) * src_x0 +(torch.sqrt(alpha_t)* (1.0 - alpha_bar_tm1) / (1.0 - alpha_bar_t)) * src_x_t)

        src_z = (src_x_tm1 - src_mu) / sigma_t

        with torch.no_grad():
            tgt_noise = self.get_noise_preds(tgt_x_t, t, tgt_text_embedding, guidance_scale)

        tgt_x0 = (tgt_x_t - torch.sqrt(1.0 - alpha_bar_t) * tgt_noise) / torch.sqrt(alpha_bar_t)

        tgt_mu = (
            (torch.sqrt(alpha_bar_tm1) * beta_t / (1.0 - alpha_bar_t)) * tgt_x0
            + (torch.sqrt(alpha_t)      * (1.0 - alpha_bar_tm1) / (1.0 - alpha_bar_t)) * tgt_x_t
        )

        tgt_z = (tgt_x_tm1 - tgt_mu) / sigma_t

        grad_xt = (tgt_z - src_z)               
        loss = (tgt_x_t * grad_xt).sum(dim=(1,2,3)).mean() * grad_scale

        ## loss = (tgt_z - src_z).pow(2).mean() * grad_scale

        return loss
    
    ## 이유는 잘 모르겠으나 , MSE하면 안되고 직접 야코비안 제거한 loss를 다시 적분하는 식으로 가야함
    
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
