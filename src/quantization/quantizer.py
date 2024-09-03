from typing import Any, Dict

import src.quantization as compose_quantization


class Quantizer:
    def __init__(self, config: Dict) -> None:
        self.config = config

    def __call__(self) -> Any:
        return getattr(compose_quantization, 
                       self.config.quantization.name)(self.config)
