import lightning.pytorch as pl

from src.quantization.abc.abc_quant import BaseQuant
from src.quantization.rniq.layers.rniq_conv2d import NoisyConv2d
from src.quantization.rniq.layers.rniq_linear import NoisyLinear
from src.quantization.rniq.layers.rniq_act import NoisyAct
from src.quantization.rniq.utils.model_helper import ModelHelper
from src.quantization.rniq.rniq_loss import PotentialLoss
from src.aux.qutils import attrsetter, is_biased

from torch import nn
from copy import deepcopy
from operator import attrgetter
from collections import OrderedDict


class RNIQQuant(BaseQuant):
    def module_mappings(self):
        return {
            nn.Conv2d: NoisyConv2d,
            nn.Linear: NoisyLinear,
        }

    def quantize(self, lmodel: pl.LightningModule, in_place=False):
        if in_place:
            qmodel = lmodel
        else:
            qmodel = deepcopy(lmodel)
            
        layer_names, layer_types = zip(
            *[(n, type(m)) for n, m in qmodel.named_modules()]) 

        # The part when original LModule structure gets changed
        qmodel.qscheme = self.qscheme
        qmodel.wrapped_criterion = PotentialLoss(
            qmodel.criterion,
            alpha=(1,1,1),
            # alpha=self.alpha,
            lmin=0,
            p=1,
            a=self.act_bit,
            w=self.weight_bit,
            scale_momentum=0.9,
        )

        # Important step. Replacing training and validation steps
        # with alternated ones.
        qmodel.training_step = RNIQQuant.noisy_training_step.__get__(
            qmodel, type(qmodel)
        )
        qmodel.validation_step = RNIQQuant.noisy_validation_step.__get__(
            qmodel, type(qmodel)
        )

        # Replacing layers directly
        qlayers = self._get_layers(lmodel)
        for layer in qlayers.keys():
            module = attrgetter(layer)(lmodel)
            preceding_layer_type = layer_types[layer_names.index(layer) - 1]
            if issubclass(preceding_layer_type, nn.ReLU):
                qmodule = self._quantize_module(module, signed_Activations=False)
            else:
                qmodule = self._quantize_module(module, signed_Activations=True)
                
            attrsetter(layer)(qmodel, qmodule)
        
        return qmodel

    @staticmethod  # yes, it's a static method with self argument
    def noisy_step(self, x):
        # now that we set qmodule.qscheme, we can address it in replaced step
        return (self.model(x), *ModelHelper.get_model_values(self.model, self.qscheme))

    @staticmethod
    def noisy_training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = RNIQQuant.noisy_step(self, inputs)
        loss = self.wrapped_criterion(outputs, targets)
        self.log("Train loss", loss, prog_bar=True)
        self.log("Base train loss", self.wrapped_criterion.base_loss, prog_bar=True)
        self.log("Wloss", self.wrapped_criterion.wloss, prog_bar=False)
        self.log("Aloss", self.wrapped_criterion.aloss, prog_bar=False)
        self.log("Weight reg loss", self.wrapped_criterion.weight_reg_loss, prog_bar=False)

        return loss

    @staticmethod
    def noisy_validation_step(self, val_batch, val_index):
        inputs, targets = val_batch
        outputs = RNIQQuant.noisy_step(self, inputs)

        val_loss = self.wrapped_criterion(outputs, targets)
        for name, metric in self.metrics:
            metric_value = metric(outputs[0], targets)
            self.log(f"{name}", metric_value, prog_bar=False)

        self.log("Validation loss", val_loss, prog_bar=False)

    def _init_config(self):
        if self.config:
            self.quant_config = self.config.quantization
            self.act_bit = self.quant_config.act_bit
            self.weight_bit = self.quant_config.weight_bit
            self.excluded_layers = self.quant_config.excluded_layers
            self.qscheme = self.quant_config.qscheme

    def _quantize_module(self, module, signed_Activations):
        if isinstance(module, nn.Conv2d):
            qmodule = self._quantize_module_conv2d(module)
        elif isinstance(module, nn.Linear):
            qmodule = self._quantize_module_linear(module)
        else:
            raise NotImplementedError(
                f"Module not supported {type(module)}"
            )

        qmodule.weight = module.weight

        if is_biased(module):
            qmodule.bias = module.bias

        qmodule = self._get_quantization_sequence(qmodule, signed_Activations)

        return qmodule

    def _get_quantization_sequence(self, qmodule, signed_activations):
        sequence = nn.Sequential(OrderedDict([
            ("activations_quantizer", NoisyAct(signed=signed_activations)),
            ("0", qmodule)
        ]))

        return sequence

    def _quantize_module_conv2d(self, module: nn.Conv2d):
        return NoisyConv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            is_biased(module),
            module.padding_mode,
            qscheme=self.qscheme,
            log_s_init=-12,
        )

    def _quantize_module_linear(self, module: nn.Linear):
        return NoisyLinear(
            module.in_features,
            module.out_features,
            is_biased(module),
            qscheme=self.qscheme,
            log_s_init=-12,
        )
