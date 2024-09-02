import lightning.pytorch as pl
import src.models as compose_models

from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from src.aux.types import MType
from src.models.compose.vision.vision_cls_module import LVisionCls

class ModelComposer():
    def __init__(self, config=None) -> None:
        self.config = config
        self.model_type: MType
        self.model: nn.Module
        self.criterion: _Loss
        self.optimizer: Optimizer
        self.lr: float = 1e-4
    
    def compose(self) -> pl.LightningModule:
        if self.config:
            model_config = self.config.model
            training_config = self.config.training
            
            self.model_type = MType[model_config.type]
            self.model = getattr(compose_models, model_config.name)(**model_config.params)
            self.criterion = getattr(nn, training_config.criterion)()
            self.optimizer = getattr(optim, training_config.optimizer)
            self.lr = training_config.learning_rate
        else:
            assert(self.model)
            assert(self.model_type)
            assert(self.criterion)
            assert(self.optimizer)
        
        self.optimizer = self.optimizer(self.model.parameters(), self.lr)
        
        if self.model_type == MType.VISION_CLS:
            module = LVisionCls(self.__dict__)
        elif self.model_type == MType.VISION_DNS:
            raise NotImplementedError()
        elif self.model_type == MType.VISION_SR:
            raise NotImplementedError()
        elif self.model_type == MType.LM:
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        
        return module