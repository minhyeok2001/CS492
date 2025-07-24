import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

from .view_base import BaseView

def get_circle_mask(img_size: int, r: int):
    # TODO: Implement get_circle_mask
    raise NotImplementedError(f"get_circle_mask not implemented")

def inner_rotate_func_with_mask(im: torch.Tensor, mask: torch.Tensor, angle, interpolate=False):
    # TODO: Implement inner_rotate_func_with_mask
    raise NotImplementedError(f"inner_rotate_func_with_mask not implemented")
 
class InnerRotateView(BaseView):
    """
    Implements an "inner circle" view, where a circle inside the image spins
    but the border stays still. Inherits from `PermuteView`, which implements
    the `view` and `inverse_view` functions as permutations. We just make
    the correct permutation here, and implement the `make_frame` method
    for animation
    """

    def __init__(self, angle):
        """
        Make the correct "inner circle" permutations and pass it to the
        parent class constructor.
        """
        self.angle = angle
        self.stage_1_mask = get_circle_mask(64, 24)
        self.stage_2_mask = get_circle_mask(256, 96)

    def view(self, im, **kwargs):
        im_size = im.shape[-1]
        if im_size == 64:
            mask = self.stage_1_mask.to(im)
            self.stage_1_mask = mask
        elif im_size == 256:
            mask = self.stage_2_mask.to(im)
            self.stage_2_mask = mask

        inner_rotated = inner_rotate_func_with_mask(im, mask, -self.angle, interpolate=False)

        return inner_rotated

    def inverse_view(self, im, **kwargs):
        im_size = im.shape[-1]
        if im_size == 64:
            mask = self.stage_1_mask.to(im)
            self.stage_1_mask = mask
        elif im_size == 256:
            mask = self.stage_2_mask.to(im)
            self.stage_2_mask = mask

        inner_rotated = inner_rotate_func_with_mask(im, mask, self.angle, interpolate=False)

        return inner_rotated

