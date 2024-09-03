from typing import Tuple
from torch import nn

from src.aux.types import QScheme


class RniqLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        qscheme: QScheme = QScheme.PER_TENSOR,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.qscheme = qscheme
