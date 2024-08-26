from torch import nn


class QuantizedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, 
                 bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, 
                         bias, device, dtype)
