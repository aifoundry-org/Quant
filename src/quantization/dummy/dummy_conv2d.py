from typing import Tuple
from torch import nn


class QuantizedConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int | Tuple[int, int], stride: int | Tuple[int, int] = 1, 
                 padding: str | int | Tuple[int, int] = 0, dilation: int | Tuple[int, int] = 1, 
                 groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', 
                 device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels, 
                         kernel_size, stride,
                         padding, dilation, 
                         groups, bias, 
                         padding_mode, device, dtype)
