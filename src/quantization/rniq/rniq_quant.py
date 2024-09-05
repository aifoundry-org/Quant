from torch import nn

from src.quantization.abc.abc_quant import BaseQuant
from src.quantization.rniq.layers.rniq_conv2d import NoisyConv2d
from src.quantization.rniq.layers.rniq_linear import NoisyLinear
from src.aux.qutils import attrsetter, is_biased

from copy import deepcopy
from operator import attrgetter
from collections import OrderedDict

class RNIQQuant(BaseQuant):
    def module_mappings(self):
        return {
            nn.Conv2d: NoisyConv2d,
            nn.Linear: NoisyLinear,
        }
    
    def quantize(self, model, in_place=False):
        if in_place:
            qmodel = deepcopy(model)
        else:
            qmodel = model
            
        qlayers = self._get_layers(model)
        
        for layer in qlayers.keys():
            module = attrgetter(layer)(qmodel)
            qmodule = self._quantize_module(module)
            attrsetter(layer)(qmodel, qmodule)
        
        return qmodel
        