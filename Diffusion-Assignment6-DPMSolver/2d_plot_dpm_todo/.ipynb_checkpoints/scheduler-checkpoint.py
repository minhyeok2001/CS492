from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


class BaseScheduler(nn.Module):
    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float = 1e-4,
        beta_T: float = 0.02,
        mode="linear",
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def uniform_sample_t(
        self, batch_size, device: Optional[torch.device] = None
    ) -> torch.IntTensor:
        """
        Uniformly sample timesteps.
        """
        ts = np.random.choice(np.arange(self.num_train_timesteps), batch_size)
        ts = torch.from_numpy(ts)
        if device is not None:
            ts = ts.to(device)
        return ts


class DDPMScheduler(BaseScheduler):
    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float = 1e-4,
        beta_T: float = 0.02,
        mode="linear",
        sigma_type="small",
    ):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)

        # sigmas correspond to $\sigma_t$ in the DDPM paper.
        self.sigma_type = sigma_type
        if sigma_type == "small":
            # when $\sigma_t^2 = \tilde{\beta}_t$.
            alphas_cumprod_t_prev = torch.cat(
                [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
            )
            sigmas = (
                (1 - alphas_cumprod_t_prev) / (1 - self.alphas_cumprod) * self.betas
            ) ** 0.5
        elif sigma_type == "large":
            # when $\sigma_t^2 = \beta_t$.
            sigmas = self.betas**0.5

        self.register_buffer("sigmas", sigmas)

    def step(self, x_t: torch.Tensor, t: int, eps_theta: torch.Tensor):
        """
        One step denoising function of DDPM: x_t -> x_{t-1}.

        Input:
            x_t (`torch.Tensor [B,C,H,W]`): samples at arbitrary timestep t.
            t (`int`): current timestep in a reverse process.
            eps_theta (`torch.Tensor [B,C,H,W]`): predicted noise from a learned model.
        Ouptut:
            sample_prev (`torch.Tensor [B,C,H,W]`): one step denoised sample. (= x_{t-1})
        """

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Assignment 1. Implement the DDPM reverse step.
        sample_prev = None
        #######################

        return sample_prev

    # https://nn.labml.ai/diffusion/ddpm/utils.html
    def _get_teeth(self, consts: torch.Tensor, t: torch.Tensor):  # get t th const
        const = consts.gather(-1, t)
        return const.reshape(-1, 1, 1, 1)

    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.IntTensor,
        eps: Optional[torch.Tensor] = None,
    ):
        """
        A forward pass of a Markov chain, i.e., q(x_t | x_0).

        Input:
            x_0 (`torch.Tensor [B,C,H,W]`): samples from a real data distribution q(x_0).
            t: (`torch.IntTensor [B]`)
            eps: (`torch.Tensor [B,C,H,W]`, optional): if None, randomly sample Gaussian noise in the function.
        Output:
            x_t: (`torch.Tensor [B,C,H,W]`): noisy samples at timestep t.
            eps: (`torch.Tensor [B,C,H,W]`): injected noise.
        """

        if eps is None:
            eps = torch.randn(x_0.shape, device="cuda")

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Assignment 1. Implement the DDPM forward step.
        x_t = None
        #######################

        return x_t, eps


class DDIMScheduler(BaseScheduler):
    def __init__(self, num_train_timesteps, beta_1=1e-4, beta_T=0.02, mode="linear"):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)

        one = torch.tensor(1.0)
        self.register_buffer("alpha_prod_0", one)

    def set_timesteps(
        self, num_inference_timesteps: int, device: Union[str, torch.device] = None
    ):
        if num_inference_timesteps > self.num_train_timesteps:
            raise ValueError(
                f"num_inference_timesteps ({num_inference_timesteps}) cannot exceed self.num_train_timesteps ({self.num_train_timesteps})"
            )

        self.num_inference_timesteps = num_inference_timesteps

        step_ratio = self.num_train_timesteps // num_inference_timesteps
        timesteps = (
            (np.arange(0, num_inference_timesteps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps)

    def step(
        self,
        x_t: torch.Tensor,
        t: int,
        eps_theta: torch.Tensor,
        eta: float = 0.0,
    ):

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Assignment 2. Implement the DDIM reverse step.
        sample_prev = None
        #######################

        return sample_prev

    def add_noise(
        self,
        x_0,
        t: torch.IntTensor,
        eps: Optional[torch.Tensor] = None,
    ):
        """
        Input:
            x_0: [B,C,H,W]
            t: [B]
            eps: [B,C,H,W]
        Output:
            x_t: [B,C,H,W]
            eps: [B,C,H,W]
        """
        if eps is None:
            eps = torch.randn(x_0.shape, device=x_0.device)

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Assignment 2. Implement the DDIM forward step. Identical to the DDPM forward step.
        x_t = None
        #######################

        return x_t, eps


class DPMSolverScheduler(BaseScheduler):
    def __init__(self, num_train_timesteps, beta_1=1e-4, beta_T=0.02, mode="linear"):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)

    def set_timesteps(
        self, num_inference_timesteps: int, device: Union[str, torch.device] = None
    ):
        if num_inference_timesteps > self.num_train_timesteps:
            raise ValueError(
                f"num_inference_timesteps ({num_inference_timesteps}) cannot exceed self.num_train_timesteps ({self.num_train_timesteps})"
            )

        self.num_inference_timesteps = num_inference_timesteps

        step_ratio = self.num_train_timesteps // num_inference_timesteps
        timesteps = (
            (np.arange(0, num_inference_timesteps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps)

    def step(
        self,
        x_t: torch.Tensor,
        t: int,
        eps_theta: torch.Tensor,
        eta: float = 0.0,
    ):

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Assignment 2. Implement the DDIM reverse step.
        sample_prev = None
        #######################

        return sample_prev

    def add_noise(
        self,
        x_0,
        t: torch.IntTensor,
        eps: Optional[torch.Tensor] = None,
    ):
        """
        Input:
            x_0: [B,C,H,W]
            t: [B]
            eps: [B,C,H,W]
        Output:
            x_t: [B,C,H,W]
            eps: [B,C,H,W]
        """
        if eps is None:
            eps = torch.randn(x_0.shape, device=x_0.device)

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Assignment 2. Implement the DDIM reverse step. Identical to the DDPM forward step.
        x_t = None
        #######################

        return x_t, eps


class DPMSolverScheduler(BaseScheduler):
    def __init__(self, num_train_timesteps, beta_1, beta_T, mode="linear"):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)
        self._convert_notations_ddpm_to_dpm()

    def _convert_notations_ddpm_to_dpm(self):
        """
        Based on the forward passes of DDPM and DPM-Solver, convert the notations of DDPM to those of DPM-Solver.
        Refer to Eq. 4 in the DDPM paper and Eq. 2.1 in the DPM-Solver paper.
        """
        self.dpm_alphas = torch.sqrt(self.alphas_cumprod)
        self.dpm_sigmas = torch.sqrt(1 - self.alphas_cumprod)
        self.dpm_lambdas = torch.log(self.dpm_alphas) - torch.log(self.dpm_sigmas)

    def first_order_step(self, x_s, s, t, eps_theta):
        assert torch.all(s > t), f"timestep s should be larger than timestep t"
        alpha_s = self.dpm_alphas[s]
        alpha_t = self.dpm_alphas[t]
        sigma_t = self.dpm_sigmas[t]
        lambda_s = self.dpm_lambdas[s]
        lambda_t = self.dpm_lambdas[t]

        h = lambda_t - lambda_s

        x_t = alpha_t / alpha_s * x_s - sigma_t * (torch.exp(h) - 1) * eps_theta

        return x_t

    def step(
        self,
        x_t: torch.Tensor,
        t: Union[torch.IntTensor, int],
        eps_theta: torch.Tensor,
    ):
        """
        One step denoising function of DPM-Solver: x_t -> x_{t-1}.

        Input:
            x_t (`torch.Tensor [B,C,H,W]`): samples at arbitrary timestep t.
            t (`int` or `torch.Tensor [B]`): current timestep in a reverse process.
            eps_theta (`torch.Tensor [B,C,H,W]`): predicted noise from a learned model.
        Ouptut:
            sample_prev (`torch.Tensor [B,C,H,W]`): one step denoised sample. (= x_{t-1})

        """

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Assignment 7. Implement the DPM-Solver reverse step.
        # sample_prev = None
        t_prev = t - self.num_train_timesteps // self.num_inference_timesteps
        sample_prev = self.first_order_step(x_t, t, t_prev, eps_theta)
        #######################

        return sample_prev

    def add_noise(
        self,
        x_0,
        t: torch.IntTensor,
        eps: Optional[torch.Tensor] = None,
    ):
        """
        Input:
            x_0: [B,C,H,W]
            t: [B]
            eps: [B,C,H,W]
        Output:
            x_t: [B,C,H,W]
            eps: [B,C,H,W]
        """
        if eps is None:
            eps = torch.randn(x_0.shape, device=x_0.device)

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Assignment 7. Implement the DPM forward step.
        x_t = self.dpm_alphas[t] * x_0 + self.dpm_sigmas[t] * eps
        #######################

        return x_t, eps
