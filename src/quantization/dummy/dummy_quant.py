from torch import nn

from src.quantization.abc.abc_quant import BaseQuant
from src.quantization.dummy.dummy_conv2d import QuantizedConv2d
from src.quantization.dummy.dummy_linear import QuantizedLinear


class DummyQuant(BaseQuant):
    def module_mappings(self):
        return {
            nn.Conv2d: QuantizedConv2d,
            nn.Linear: QuantizedLinear
        }

    def get_quantized(self, model):
        pass

    def get_wrapped(self, model):
        pass

    def _quantize_module(self, module):
        if isinstance(module, nn.Conv2d):
            return self._quantize_module_conv2d(module)
        elif isinstance(module, nn.Linear):
            return self._quantize_module_linear(module)
        else:
            raise NotImplementedError(
                f"Unknown type for quantization {type(module)}")

    def _quantize_module_conv2d(self, module: nn.Conv2d):
        qmodule = QuantizedConv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.bias,
            module.padding_mode
        )
        return qmodule

    def _quantize_module_linear(self, module: nn.Linear):
        qmodule = QuantizedLinear(
            module.in_features,
            module.out_features,
            module.bias
        )
        return qmodule
