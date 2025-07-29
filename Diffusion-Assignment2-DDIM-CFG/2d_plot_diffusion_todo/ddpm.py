import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
<torch.gather의 예시>

# input이 2D 텐서라면
input = torch.tensor([
    [10, 11, 12],
    [20, 21, 22],
    [30, 31, 32],
])   # shape (3, 3)

# 우리가 0번째 차원(dim=0)을 기준으로 각 열에서 어떤 행을 뽑아올지 정하는 index
index = torch.tensor([
    [2, 0, 1],   # 첫 번째 “행”에서: 열0→row2, 열1→row0, 열2→row1
    [0, 2, 2],   # 두 번째 “행”에서: 열0→row0, 열1→row2, 열2→row2
])   # shape (2, 3)

out = torch.gather(input, dim=0, index=index)
# 결과 shape는 (2,3) 이고, 계산 방식은:
# out[0,0] = input[ index[0,0] , 0 ] = input[2, 0] = 30
# out[0,1] = input[ index[0,1] , 1 ] = input[0, 1] = 11
# ...
# out[1,2] = input[ index[1,2] , 2 ] = input[2, 2] = 32

print(out)
# tensor([[30, 11, 22],
#         [10, 31, 32]])
"""

def extract(input, t: torch.Tensor, x: torch.Tensor):
    if t.ndim == 0:
        t = t.unsqueeze(0)
    shape = x.shape
    t = t.long().to(input.device)
    out = torch.gather(input, 0, t)
    reshape = [t.shape[0]] + [1] * (len(shape) - 1) ## 파이썬은 이거 리스트 그냥 더해도 append 하는것 처럼 동작
    return out.reshape(*reshape) ## * 이거 하면 리스트 요소 떼어다가 펼쳐줌


class BaseScheduler(nn.Module):
    """
    Variance scheduler of DDPM.
    """

    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float = 1e-4,
        beta_T: float = 0.02,
        mode: str = "linear",
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        ).to(torch.device("mps"))

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = (
                torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
            )
        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        #self.register_buffer("timesteps", timesteps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)


class DiffusionModule(nn.Module):
    """
    A high-level wrapper of DDPM and DDIM.
    If you want to sample data based on the DDIM's reverse process, use `ddim_p_sample()` and `ddim_p_sample_loop()`.
    """

    def __init__(self, network: nn.Module, var_scheduler: BaseScheduler):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler

    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def image_resolution(self):
        # For image diffusion model.
        return getattr(self.network, "image_resolution", None)

    def q_sample(self, x0, t, noise=None):
        """
        sample x_t from q(x_t | x_0) of DDPM.

        Input:
            x0 (`torch.Tensor`): clean data to be mapped to timestep t in the forward process of DDPM.
            t (`torch.Tensor`): timestep
            noise (`torch.Tensor`, optional): random Gaussian noise. if None, randomly sample Gaussian noise in the function.
        Output:
            xt (`torch.Tensor`): noisy samples
        """
        if noise is None:
            noise = torch.randn_like(x0)

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Compute xt.

        alphas_prod_t = extract(self.var_scheduler.alphas_cumprod, t, x0)
        xt = x0 * torch.sqrt(alphas_prod_t) + noise * torch.sqrt(1-alphas_prod_t)

        #######################

        return xt

    @torch.no_grad()
    def p_sample(self, xt, t):
        """
        One step denoising function of DDPM: x_t -> x_{t-1}.

        Input:
            xt (`torch.Tensor`): samples at arbitrary timestep t.
            t (`torch.Tensor`): current timestep in a reverse process.
        Ouptut:
            x_t_prev (`torch.Tensor`): one step denoised sample. (= x_{t-1})

        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        # compute x_t_prev.
        if isinstance(t, int):
            t = torch.tensor([t]).to(self.device)
        eps_factor = (1 - extract(self.var_scheduler.alphas, t, xt)) / (
            1 - extract(self.var_scheduler.alphas_cumprod, t, xt)
        ).sqrt()
        eps_theta = self.network(xt, t)

        ## 아하 여기서는 reparameterization trick 써서 식 표현하면 되겠다

        ## 위에서 eps_factor가 network 앞에 붙는 계수이니까, 뒤에는 noise에 대한 scaling factor 구하면 되겠다
        if t!=0:
            noise_factor = extract(self.var_scheduler.betas,t,xt)*(1 - extract(self.var_scheduler.alphas_cumprod, t-1, xt))/(1 - extract(self.var_scheduler.alphas_cumprod, t, xt))
            noise_factor = noise_factor.sqrt()
        else :
            noise_factor = 0

        ### 주의 !!!
        ### 이거 T=0일때 핸들링 안해주면 마지막에 노이즈 잔뜩 뿌려서 문제생김 !!!

        scaling_factor = 1/(extract(self.var_scheduler.alphas,t,xt).sqrt())

        ## reparameterization trick 

        noise = torch.randn_like(xt)
        x_t_prev = scaling_factor*(xt-eps_factor*eps_theta) + noise_factor * noise

        #######################
        return x_t_prev

    @torch.no_grad()
    def p_sample_loop(self, shape):
        """
        The loop of the reverse process of DDPM.

        Input:
            shape (`Tuple`): The shape of output. e.g., (num particles, 2)
        Output:
            x0_pred (`torch.Tensor`): The final denoised output through the DDPM reverse process.
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        # sample x0 based on Algorithm 2 of DDPM paper.
        x0_pred = torch.zeros(shape).to(self.device)

        ## 아니 근데 timestep을 모르는데 어떻게 끝까지 reverse를 하는거지 ??
        ## -> ㅎㅎ.. self.var_sch~.~

        xt_pred = torch.randn_like(x0_pred).to(self.device)

        for time in self.var_scheduler.timesteps:    ## 예상대로라면, 여기서는 [...] 꼴의 1차원 텐서
            xt_pred = self.p_sample(xt_pred,time)
            
        x0_pred = xt_pred

        ######################
        return x0_pred

    @torch.no_grad()
    def ddim_p_sample(self, xt, t, t_prev, eta=0.0):
        """
        One step denoising function of DDIM: $x_t{\tau_i}$ -> $x_{\tau{i-1}}$.

        Input:
            xt (`torch.Tensor`): noisy data at timestep $\tau_i$.
            t (`torch.Tensor`): current timestep (=\tau_i)
            t_prev (`torch.Tensor`): next timestep in a reverse process (=\tau_{i-1})
            eta (float): correspond to η in DDIM which controls the stochasticity of a reverse process.
        Output:
           x_t_prev (`torch.Tensor`): one step denoised sample. (= $x_{\tau_{i-1}}$)
        """
        ######## TODO ########
        # NOTE: This code is used for assignment 2. You don't need to implement this part for assignment 1.
        # DO NOT change the code outside this part.
        # compute x_t_prev based on ddim reverse process.
        alpha_prod_t = extract(self.var_scheduler.alphas_cumprod, t, xt)
        if t_prev >= 0:
            alpha_prod_t_prev = extract(self.var_scheduler.alphas_cumprod, t_prev, xt)
        else:
            alpha_prod_t_prev = torch.ones_like(alpha_prod_t)

        noise = torch.randn_like(xt)
        beta = extract(self.var_scheduler.betas,t,xt)

        pred = self.network(xt,t)

        scaling_factor = 1/torch.sqrt(alpha_prod_t) 
        network_factor = torch.sqrt(1-alpha_prod_t)
        x_0 = scaling_factor*(xt - network_factor*pred)
        if t_prev !=0:
            sigma = eta * torch.sqrt(((1-alpha_prod_t_prev)/(1-alpha_prod_t))*beta)
        else:
            sigma = 0
        mu = torch.sqrt(alpha_prod_t_prev)*x_0 + torch.sqrt(1-alpha_prod_t_prev-sigma**2)*(xt-torch.sqrt(alpha_prod_t)*x_0)/torch.sqrt(1-alpha_prod_t)

        x_t_prev = mu + sigma * noise

        ######################
        return x_t_prev

    @torch.no_grad()
    def ddim_p_sample_loop(self, shape, num_inference_timesteps=50, eta=0.0):
        """
        The loop of the reverse process of DDIM.

        Input:
            shape (`Tuple`): The shape of output. e.g., (num particles, 2)
            num_inference_timesteps (`int`): the number of timesteps in the reverse process.
            eta (`float`): correspond to η in DDIM which controls the stochasticity of a reverse process.
        Output:
            x0_pred (`torch.Tensor`): The final denoised output through the DDPM reverse process.
        """
        ######## TODO ########
        # NOTE: This code is used for assignment 2. You don't need to implement this part for assignment 1.
        # DO NOT change the code outside this part.
        # sample x0 based on Algorithm 2 of DDPM paper.
        step_ratio = self.var_scheduler.num_train_timesteps // num_inference_timesteps
        timesteps = (
            (np.arange(0, num_inference_timesteps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        timesteps = torch.from_numpy(timesteps).to(torch.device("mps"))
        prev_timesteps = timesteps - step_ratio

        xt = torch.zeros(shape).to(self.device)
        xt = torch.randn_like(xt)
        
        for t, t_prev in zip(timesteps, prev_timesteps):
            xt = self.ddim_p_sample(xt,t,t_prev,eta)

        x0_pred = xt

        ######################

        return x0_pred

    def compute_loss(self, x0):
        """
        The simplified noise matching loss corresponding Equation 14 in DDPM paper.

        Input:
            x0 (`torch.Tensor`): clean data
        Output:
            loss: the computed loss to be backpropagated.
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        # compute noise matching loss.
        batch_size = x0.shape[0]
        t = (
            torch.randint(0, self.var_scheduler.num_train_timesteps, size=(batch_size,))
            .to(x0.device)
            .long()
        )
        ## 아 위 과정은 그거네 학습할 time 뽑는거 !! 즉 0~ num_train_timestep 안에서 배치사이즈만큼 뽑는것 

        ## 일단 이 loss가 Mu predictor인지, noise predictor인지, x_0 predictor인지 알아봐야할듯
        ## -> 앞에서 network.py에서 noise 예측을 했으므로, noise predictor라고 보는게 타당할듯.

        ## 이거 구현 어떻게 해야할지 몰라서 GPT 물어봤는데..
        ## 내가 헷갈린 포인트 : 내 생각에는, x_0에서 x_t 갈때 생성한 노이즈는 가우시안 분포를 따르긴 한데, 그 앞에 scaling factor가 엄청 많지 않나? 이걸 다 써서 식을 구성해야하나?
        ## GPT : 그런거 신경쓰지 말고, 어차피 noise predictor이므로 그냥 니가 노이즈를 만들고 그걸 q sample에 주고 그걸 예측하도록 해라
        
        gt_noise = torch.randn_like(x0)

        xt = self.q_sample(x0,t,gt_noise)
        pred_noise = self.network(xt,t)
        
        loss = ((gt_noise-pred_noise)**2).mean()

        ######################
        return loss

    def save(self, file_path):
        hparams = {
            "network": self.network,
            "var_scheduler": self.var_scheduler,
        }
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    def load(self, file_path):
        dic = torch.load(file_path, map_location=self.device)
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.var_scheduler = hparams["var_scheduler"]

        self.load_state_dict(state_dict)
