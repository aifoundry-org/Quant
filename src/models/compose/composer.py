import torch
import lightning.pytorch as pl

from torch import nn
from typing import Dict
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from src.aux.types import MType
from src.models.compose.vision.vision_cls_module import LVisionCls

class ModelComposer():
    def __init__(self, config: Dict | None) -> None:
        self.config = config
        self.model_type: MType
        self.model: nn.Module
        self.criterion: _Loss
        self.optimizer: Optimizer
    
    def compose(self) -> pl.LightningModule:
        if self.config:
            # TODO compose model from config
            pass
        else:
            assert(self.model)
            assert(self.model_type)
            assert(self.criterion)
            assert(self.optimizer)
        
        if self.model_type == MType.VISION_CLS:
            module = LVisionCls(self.__dict__())
        elif self.model_type == MType.VISION_DNS:
            raise NotImplementedError()
        elif self.model_type == MType.VISION_SR:
            raise NotImplementedError()
        elif self.model_type == MType.LM:
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        
        return module