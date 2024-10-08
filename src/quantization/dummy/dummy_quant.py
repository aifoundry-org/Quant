from torch import nn

from src.quantization.abc.abc_quant import BaseQuant
from src.quantization.dummy.dummy_conv2d import QuantizedConv2d
from src.quantization.dummy.dummy_linear import QuantizedLinear
from src.quantization.dummy.dummy_qact import QuantizedAct
from src.aux.qutils import attrsetter, is_biased
from src.loggers.default_logger import logger

from copy import deepcopy
from operator import attrgetter
from collections import OrderedDict


def new_forward(self, x):
    logger.debug("Altered forward...")
    return self.model(x)
    

class DummyQuant(BaseQuant):
    def module_mappings(self):
        return {
            nn.Conv2d: QuantizedConv2d,
            nn.Linear: QuantizedLinear
        }

    def quantize(self, model, in_place=False):
        if in_place:
            qmodel = deepcopy(model)
        else:
            qmodel = model
        
        # altering the forward pass
        qmodel.forward = new_forward.__get__(qmodel, type(qmodel))
            
        qlayers = self._get_layers(model)
        
        for layer in qlayers.keys():
            module = attrgetter(layer)(qmodel)
            qmodule = self._quantize_module(module)
            attrsetter(layer)(qmodel, qmodule)
        
        return qmodel

    def _quantize_module(self, module):
        if isinstance(module, nn.Conv2d):
            qmodule = self._quantize_module_conv2d(module)
        elif isinstance(module, nn.Linear):
            qmodule = self._quantize_module_linear(module)
        else:
            raise NotImplementedError(
                f"Module not supported {type(module)}")
        
        qmodule.weight.data = module.weight.data
        
        if is_biased(module):
            qmodule.bias.data = module.bias.data
        
        qmodule = self._get_quantization_sequence(qmodule)
        
        return qmodule
    
    def _get_quantization_sequence(self, qmodule):
        sequence = nn.Sequential(OrderedDict([
            ("activations_quantizer", QuantizedAct()),
            ("0", qmodule)
        ]))
        
        return sequence

    def _quantize_module_conv2d(self, module: nn.Conv2d):
        return QuantizedConv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            is_biased(module),
            module.padding_mode
        )

    def _quantize_module_linear(self, module: nn.Linear):
        return QuantizedLinear(
            module.in_features,
            module.out_features,
            is_biased(module)
        )
