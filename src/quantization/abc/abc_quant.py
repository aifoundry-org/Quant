import lightning.pytorch as pl
import re

from torch import nn
from typing import Dict, List, Type
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
    def quantize(self, model: pl.LightningModule) -> pl.LightningModule:
        """
        Base method to get quantization-ready version
        of your model.


        Args:
            model (pl.LightningModule): Target model that need to be quantized

        Returns:
            pl.LightningModule: Quantization-ready version of the target model 
                                wrapped in lightning module.
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

    def _get_layers(self, model: nn.Module, exclude_layers: List[str] = []):
        """
        Simple method to get quantizable layers names from the model
        with regards to supported layer types and excluded layers

        Args:
            model (nn.Module): Target model to extract layers from
            exclude_layers (List[str], optional): Names of model layers to exclude. Defaults to [].

        Raises:
            AttributeError: Apperars if you provide incorrect values of excluded layers

        Returns:
            Dict: Layers names and types of quantizable layers 
        """

        quantizable_layers = {n: type(m) for n, m in model.named_modules() \
            if issubclass(type(m), tuple(self.module_mappings().keys()))}
        
        for layer_name in exclude_layers:
            if layer_name in quantizable_layers:
                quantizable_layers.remove(layer_name)
            else:
                raise AttributeError(f"Layer name {layer_name} is not found in the model.")

        return quantizable_layers
    
    
