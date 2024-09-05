import torch

from typing import Tuple
from torch import nn, inf

from src.aux.types import QScheme
from src.quantization.rniq.rniq import Quantizer

from src.aux.qutils import attrsetter, is_biased


class NoisyConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int | Tuple[int, int] = 1,
        padding: str | int | Tuple[int, int] = 0,
        dilation: int | Tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        qscheme: QScheme = QScheme.PER_TENSOR,
        log_s_init: float = -12,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self.qscheme = qscheme

        if self.qscheme == QScheme.PER_TENSOR:
            self.log_s = nn.Parameter(torch.Tensor([log_s_init]), requires_grad=True)
        elif self.qscheme == QScheme.PER_CHANNEL:
            self.log_s = nn.Parameter(
                torch.empty((out_channels, 1, 1, 1)).fill_(log_s_init),
                requires_grad=True,
            )
        self.noise_ratio = nn.Parameter(
            torch.Tensor([1]),
            requires_grad=False,
        )
        self.Q = Quantizer(torch.exp2(self.log_s), 0, -inf, inf)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        s = torch.exp2(self.log_s)
        self.Q.scale = s

        if self.training:
            self.Q.rnoise_ratio = self.noise_ratio
        else:
            self.Q.rnoise_ratio = torch.Tensor([0])

        weight = self.Q.dequantize(self.Q.quantize(self.weight))

        return self._conv_forward(input, weight, self.bias)

    def extra_repr(self) -> str:
        return "in_channels={}, out_channels={}, kernel_size={}, stride={}, \
              padding={}, dilation={}, groups={}, bias={}, log_s={}, noise_ratio={}".format(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            is_biased(self),
            self.log_s,
            self.noise_ratio,
        )
