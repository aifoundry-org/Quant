import lightning.pytorch as pl

from torch import nn
from typing import Dict
from abc import ABC, abstractclassmethod


class BaseQuant(ABC):
    def __init__(self, config: Dict):
        self.config = config
    
    @abstractclassmethod
    def module_mappings(self) -> Dict:
        """
        This method should define the mappeing between source
        layers of nn architecture and their quantized counterparts.
        
        Example:
                {
                    nn.Conv2d: QuantizedConv2d,
                    nn.Linear: QuantizedLinear
                }

        Returns:
            Dict: The dictionary of layers mapping.
        """
        
    @abstractclassmethod    
    def get_quantized(self, model: nn.Module) -> pl.LightningModule:
        """
        Base method to get quantization-ready version
        of your model.
        

        Args:
            model (nn.Module): Target model that need to be quantized

        Returns:
            pl.LightningModule: Quantization-ready version of the target model 
                                wrapped in lightning module.
        """
    @abstractclassmethod
    def get_wrapped(self, model: nn.Module) -> pl.LightningModule:
        """
        Base method to get bare wrapped version of the target method
        without any quantization.
        

        Args:
            model (nn.Module): Target model that need to be wrapped

        Returns:
            pl.LightningModule: Target model wrapped in lightning module.
        """

    @abstractclassmethod
    def _quantize_module(self, module: nn.Module) -> nn.Module:
        """
        Base method for layer (module) quantization.
        Should inherit calls from different methods assigning
        types of modules and their quantizers. 
        
        Example:
                if isinstance(module, nn.Conv2d):
                    return self._quantize_module_conv2d


        Args:
            module (nn.Module): A separate module (layer) of the neural network
        
        Returns:
            nn.Module: Quantized counterpart of source module (layer)
        """