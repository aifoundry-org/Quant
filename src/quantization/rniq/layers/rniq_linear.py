import torch
import torch.nn.functional as F

from typing import Tuple
from torch import nn, inf

from src.aux.types import QScheme
from src.quantization.rniq.rniq import Quantizer


class NoisyLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        qscheme: QScheme = QScheme.PER_TENSOR,
        log_s_init: float = -12,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.qscheme = qscheme

        self.noise_ratio = nn.Parameter(torch.Tensor([1,]), requires_grad=False)
        self.Q = Quantizer(torch.exp2(self.log_s), 0, -inf, inf)
        
        if self.qscheme == QScheme.PER_TENSOR:
            self.log_s = nn.Parameter(torch.Tensor([log_s_init]), requires_grad=True)
        elif self.qscheme == QScheme.PER_CHANNEL:
            self.log_s = nn.Parameter(
                torch.empty((out_features, 1, 1, 1)).fill_(log_s_init),
                requires_grad=True,
            )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        s = torch.exp2(self.log_s)
        self.Q.scale = s
        
        if self.training:
            self.Q.rnoise_ratio = self.noise_ratio
        else:
            self.Q.rnoise_ratio = torch.Tensor([0])
        
        weight = self.Q.dequantize(self.Q.quantize(self.weight))
            
        
        return F.linear(input, weight, self.bias)
