import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)  ## 1차원 텐서
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None] 
        """
        기존 a가 (10,) 였다면, a = a[None] -> a 는 (1,10)
        a = a[:, None] -> a 는 (10,1)
        """
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor):
        if t.ndim == 0:
            t = t.unsqueeze(-1)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class TimeLinear(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_timesteps: int):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_timesteps = num_timesteps

        self.time_embedding = TimeEmbedding(dim_out)
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.fc(x)
        alpha = self.time_embedding(t).view(-1, self.dim_out)

        ## Q. 분명 저번에 사용했던 U_net diffusion에서는 concat해서 conv 하는 형식이었는데 여기서는 그냥 x값에 곱해버리는데, 이게 맞는지?
        ## -> 그냥 받아들일것. 실제로 그렇게 많이 사용한다고 함 

        ## 일단 중요한건, forward의 결과가 time emb 값이 섞였다는 점임 
        return alpha * x


class SimpleNet(nn.Module):
    def __init__(
        self, dim_in: int, dim_out: int, dim_hids: List[int], num_timesteps: int
    ):
        super().__init__()
        """
        (TODO) Build a noise estimating network.

        Args:
            dim_in: dimension of input
            dim_out: dimension of output
            dim_hids: dimensions of hidden features
            num_timesteps: number of timesteps
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        ### 아아아 이게 그거네 그 Unet 부분에 해당하는 부분 !! 그래서 time_step이 똑같은거였음

        self.layers = nn.ModuleList()
 
        in_dim = dim_in
        for dim in dim_hids:
            self.layers.append(TimeLinear(in_dim,dim,num_timesteps))
            in_dim = dim

        self.layers.append(TimeLinear(dim_hids[-1],dim_out,num_timesteps))

        ######################
        
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        (TODO) Implement the forward pass. This should output
        the noise prediction of the noisy input x at timestep t.

        Args:
            x: the noisy data after t period diffusion
            t: the time that the forward diffusion has been running
        """
        ######## TODO ########
        # DO NOT change the code outside this part.

        for layer in self.layers[:-1]:
            x = layer(x,t)
            x = F.relu(x)
        
        x = self.layer[-1](x,t)

        ######################
        return x
