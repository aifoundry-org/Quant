import torch
import lightning.pytorch as pl

from torch import nn
from typing import Dict
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

class ModelComposer():
    def __init__(self, config: Dict | None) -> None:
        self.config = config
        self.model: nn.Module
        self.criterion: _Loss
        self.optimizer: Optimizer
    
    def compose(self) -> pl.LightningModule:
        if self.config:
            # TODO compose model from config
            pass
        else:
            # TODO compose model from provided modules
            pass