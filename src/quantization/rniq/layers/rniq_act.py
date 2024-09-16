import torch
from torch import nn, inf

from src.aux.types import QScheme
from src.quantization.rniq.rniq import Quantizer
from src.quantization.rniq.utils.enums import QMode


class NoisyAct(nn.Module):
    def __init__(self, init_s=-10, init_q=10, signed=True, noise_ratio=1) -> None:
        super().__init__()
        self._act_b = torch.tensor([0]).float()
        self._log_act_s = torch.tensor([init_s]).float()
        self._log_act_q = torch.tensor([init_q]).float()
        self.signed = signed
        self._noise_ratio = torch.tensor(noise_ratio)
        self._log_act_s = torch.tensor([init_s]).float()
        self.log_act_q = torch.nn.Parameter(self._log_act_q, requires_grad=True)
        if signed:
            self.act_b = torch.nn.Parameter(self._act_b, requires_grad=True)
        else:
            self.act_b = torch.nn.Parameter(torch.exp2(self._log_act_q) / 2, requires_grad=False)

        self.log_act_s = torch.nn.Parameter(self._log_act_s, requires_grad=True)
        self.mode = QMode.ROUND_VAL
        self.Q = Quantizer(torch.exp2(self._log_act_s), 0, -inf, inf)

    def forward(self, x):
        s = torch.exp2(self.log_act_s)
        q = torch.exp2(self.log_act_q)

        q_2 = q / 2
        act_b = self.act_b if self.signed else q_2

        self.Q.scale = s
        self.Q.min_val = act_b - q_2
        self.Q.max_val = act_b + q_2
        self.Q.rnoise_ratio = self._noise_ratio

        # kinda obsolete when we can change noise ratio directly
        if self.mode == QMode.ROUND_VAL:
            self.Q.rnoise_ratio = torch.tensor(0)
        elif self.mode == QMode.SOURCE_VAL:
            self.Q.rnoise_ratio = torch.tensor(-1)
        
        # return self.Q.quantize(x)
        return self.Q.dequantize(self.Q.quantize(x))
