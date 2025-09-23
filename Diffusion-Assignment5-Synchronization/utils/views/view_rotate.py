from PIL import Image
import torch

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from .view_base import BaseView


class Rotate90CWView(BaseView):
    def __init__(self):
        pass

    def view(self, im, background=None, **kwargs):
        return torch.rot90(im, k=-1, dims=(-2, -1))

    def inverse_view(self, noise, background=None, **kwargs):
        ## 여기서 노이즈도 동일하게 돌려야하는 이유가?,?
        return torch.rot90(noise, k=1, dims=(-2, -1))
