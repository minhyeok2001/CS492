from diffusers import DDIMScheduler
import torch


scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

# 전체 속성 보기 (dict형태)
print("=== Scheduler config ===")
print(scheduler.config)

# timesteps 확인
print("\n=== Timesteps ===")
print(scheduler.timesteps[:10], "...")   # 앞 10개만

# alphas_cumprod 확인
print("\n=== Alphas_cumprod ===")
print(scheduler.alphas_cumprod[:10])

# 기타 내부 항목들 한 번에 보기
print("\n=== Scheduler 내부 속성들 ===")
for k, v in scheduler.__dict__.items():
    print(k, ":", type(v))


# dir 로 모든 속성/메서드 확인
print([m for m in dir(scheduler) if "noise" in m.lower()])


print(scheduler.init_noise_sigma)

x_0 = torch.randn(3,32,32)
noise = torch.randn(3,32,32)
print(x_0)

timestep = torch.tensor([999])

print(scheduler.timesteps.shape)

print(scheduler.add_noise(x_0,noise,timestep))


num_inference_steps = 50
scheduler.set_timesteps(num_inference_steps)
print(scheduler.timesteps)