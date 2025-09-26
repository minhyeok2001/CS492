from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

def extract(input, t: torch.Tensor, x: torch.Tensor):
    if t.ndim == 0:
        t = t.unsqueeze(0)
    shape = x.shape
    t = t.long().to(input.device)
    out = torch.gather(input, 0, t)
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


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
        
        # The assignment only supports linear mode.
        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
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


class DPMSolverScheduler(BaseScheduler):
    def __init__(self, num_train_timesteps, beta_1=1e-4, beta_T=0.02, mode="linear"):
        assert mode == "linear", f"only linear scheduling is supported."
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)
        self._convert_notations_ddpm_to_dpm()
        # To access the model forward in high-order scheduling.
        self.net_forward_fn = None

    def _convert_notations_ddpm_to_dpm(self):
        """
        Based on the forward passes of DDPM and DPM-Solver, convert the notations of DDPM to those of DPM-Solver.
        Refer to Eq. 4 in the DDPM paper and Eq. 2.1 in the DPM-Solver paper.
        """
        dpm_alphas = torch.sqrt(self.alphas_cumprod)
        dpm_sigmas = torch.sqrt(1 - self.alphas_cumprod)
        dpm_lambdas = torch.log(dpm_alphas) - torch.log(dpm_sigmas)

        ## 아하 이게 SNR이겠네요

        self.register_buffer("dpm_alphas", dpm_alphas)
        self.register_buffer("dpm_sigmas", dpm_sigmas)
        self.register_buffer("dpm_lambdas", dpm_lambdas)

    def set_timesteps(
        self, num_inference_timesteps: int, device: Union[str, torch.device] = None
    ):
        if num_inference_timesteps > self.num_train_timesteps:
            raise ValueError(
                f"num_inference_timesteps ({num_inference_timesteps}) cannot exceed self.num_train_timesteps ({self.num_train_timesteps})"
            )

        self.num_inference_timesteps = num_inference_timesteps
        
        #Uniform t
        timesteps = (
            np.linspace(0, self.num_train_timesteps - 1, num_inference_timesteps + 1)
            .round()[::-1][:-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps)
         
    def inverse_lambda(self, lamb):
        """
        inverse function of lambda(t)
        """
        # ??? 이게 왜필요하지? 일단 코드 의미부터 천천히

        log_alpha_array = torch.log(self.dpm_alphas).reshape(1, -1)
        t_array = torch.linspace(0, 1, self.num_train_timesteps+1)[1:].reshape(1,-1).to(log_alpha_array)
        log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
        # lambda를 받아서 1과 1^-2lambda를 더하고 -0.5 곱하기 -> 이게 뭔의미 ??? -> 이게 DPM solver에서 log alpha네 !
        t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(log_alpha_array, [1]),
                           torch.flip(t_array, [1]))
        
        ## 위에서 구한 alpha에 해당하는 시간 t를 알아내는 과정

        """
        Convert continuous t in [1 / N, 1] to discrete one in [0, 1000 * (N-1)/N]
        """
        t = ((t - 1 / self.num_train_timesteps) * 1000).long()
        return t.squeeze()


    def first_order_step(self, x_s, s, t, eps_theta):
        """
        Implement Eq 4.1. in the DPM-Solver paper.
        Input:
            x_s (`torch.Tensor`): samples at timestep s.
            s (`torch.Tensor`): denoising starting timestep.
            t (`torch.Tensor`): denoising end timestep.
            eps_theta (`torch.Tensor`): noise prediction at s.
        Output:
            x_t (`torch.Tensor`): one step denoised sample.
        """
        assert torch.all(s > t), f"timestep s should be larger than timestep t"
        ######## TODO ########
        # DO NOT change the code outside this part.
        alpha_s = extract(self.dpm_alphas, s, x_s)
        alpha_t = extract(self.dpm_alphas, t, x_s) # 어차피 x_s 넣는 곳은 포맷만 보는거라서 괜찮을듯
        sigma_t = extract(self.dpm_sigmas, t, x_s)
        lambda_t = extract(self.dpm_lambdas, t, x_s)
        lambda_s = extract(self.dpm_lambdas, s, x_s)

        h = lambda_t - lambda_s
        
        x_t = alpha_t / alpha_s * x_s - sigma_t * (torch.exp(h) - 1) * eps_theta
        
        #print("========================")
        #print(alpha_s.shape)
        #  dpm_alphas = torch.sqrt(self.alphas_cumprod) 에서 나왔음

        ######################
        return x_t
    
    def second_order_step(self, x_ti1, t_i1, t_i, eps_theta):
        """
        Implement Algorithm 1 (DPM-Solver-2) in the DPM-Solver paper.
        You might need to use `self.net_forward_fn()` function call
        for the computation of \epsilon_\theta(u_i, s_i).
        Input:
            x_ti1 (`torch.Tensor`): samples at timestep t_{i-1}
            t_i1 (`torch.Tensor`): timestep at t_{i-1}
            t_i (`torch.Tensor`): timestep at t_i
            eps_theta (`torch.Tensor`): \epsilon_\theta(x_{t_{i-1}}, t_{i-1})
        Output:
            x_ti (`torch.Tensor`): one step denoised samples. (=x_{t_i})
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        lambda_i1 = extract(self.dpm_lambdas, t_i1, x_ti1)
        lambda_i = extract(self.dpm_lambdas, t_i, x_ti1)
        s_i = self.inverse_lambda((lambda_i1 + lambda_i)/2)
        
        alpha_s_i = extract(self.dpm_alphas, s_i, x_ti1)
        alpha_t_i1 = extract(self.dpm_alphas, t_i1, x_ti1) 
        alpha_t_i = extract(self.dpm_alphas, t_i, x_ti1) 
        sigma_s_i = extract(self.dpm_sigmas, s_i, x_ti1)
        sigma_t_i = extract(self.dpm_sigmas, t_i, x_ti1)

        h = lambda_i - lambda_i1
        #model_output = self.net_forward_fn(x_ti1, t_i1.to(x_ti1.device))
        u_i = alpha_s_i/alpha_t_i1 * x_ti1 - sigma_s_i * (torch.exp(h/2) -1) * eps_theta
        x_ti = alpha_t_i / alpha_t_i1 * x_ti1 - sigma_t_i * (torch.exp(h) -1) * self.net_forward_fn(u_i, s_i.to(u_i.device))
        ######################

        return x_ti


    def step(
        self,
        x_t: torch.Tensor,
        t: Union[torch.IntTensor, int],
        eps_theta: torch.Tensor,
        order=1,
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
        t_prev = (t - self.num_train_timesteps // self.num_inference_timesteps).clamp(0)
        if order == 1:
            sample_prev = self.first_order_step(x_t, t, t_prev, eps_theta)
        elif order == 2:
            sample_prev = self.second_order_step(x_t, t, t_prev, eps_theta)

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
        # Assignment 6. Implement the DPM forward step.

        ## 자자. recall을 해보면, DPM forward의 경우에는, 우선 ODE 꼴로 이루어져있었음 + 이거 SNR만 맞으면 상관없었던거같음
        # 아 이거 기본 식이 alpha x_0, sigma ^2 였음

        alpha = extract(self.dpm_alphas,t,x_0)
        sigmas = extract(self.dpm_sigmas,t,x_0)
        x_t = alpha * x_0 + sigmas**2 * eps
        
        #######################

        return x_t, eps

"""
Source: https://github.com/LuChengTHU/dpm-solver/blob/main/dpm_solver_pytorch.py
"""
def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand
