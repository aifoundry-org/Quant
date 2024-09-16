import lightning.pytorch as pl

from torch import nn
from typing import Dict, List
from abc import ABC, abstractmethod


class BaseQuant(ABC):
    def __init__(self, config: Dict):
        self.config = config
        self.act_bit: int
        self.weight_bit: int
        self.excluded: List
        self._init_config()
        
    @classmethod
    @abstractmethod
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
    @classmethod
    @abstractmethod
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
    @classmethod
    @abstractmethod
    def _quantize_module(self, module: nn.Module) -> nn.Module:
        """
        Base method for layer (module) quantization.
        Should inherit calls from different methods assigning
        types of modules and their quantizers. 

        Usage:

            if isinstance(module, nn.Conv2d):
                    return self._quantize_module_conv2d


        Args:
            module (nn.Module): A separate module (layer) of the neural network

        Returns:
            nn.Module: Quantized counterpart of source module (layer)
        """
    @classmethod
    @abstractmethod
    def _get_quantization_sequence(self, qmodule: nn.Module) -> nn.Module:
        """
        When quantizing neural network you need also quantize activations to prevent \n
        accumulator values of exceeding target bit width (https://arxiv.org/abs/2106.08295, Figure.2)
        
        In general, two approaches exist for activation quantization.
        1. Quantize layer input activations: `QuantizedActivation(INT) -> QuantizedLayer(INT)`
        2. Quantize layer output activations: `QuantizedLayer(INT) -> QuantizedActivation(INT)`
        
        The idea is to replace `qmodule` with:
            `nn.Sequential(qmodule, qact)`
            or
            `nn.Sequential(qact, qmodule)`    

        Args:
            qmodule (nn.Module): The source module to combine with activation quantizer

        Returns:
            nn.Module: Resulting quantization sequence
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
                quantizable_layers.pop(layer_name)
            else:
                raise AttributeError(f"Layer name {layer_name} is not found in the model.")

        return quantizable_layers
    
    def _init_config(self):
        """
        The method is for mapping quantization config attributs into 
        class instance values. Should be expaneded for the sake of the specific
        quantizers.
        """
        if self.config:
            quant_config = self.config.quantization
            self.act_bit = quant_config.act_bit
            self.weight_bit = quant_config.weight_bit
            self.excluded_layers = quant_config.excluded_layers
    
    
