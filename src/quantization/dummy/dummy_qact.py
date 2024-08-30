from typing import Any
from torch import nn

class QuantizedAct(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, x: Any) -> Any:
        return x
    
    