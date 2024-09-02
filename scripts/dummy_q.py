import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from torch import nn, optim
from torchvision.models import resnet18
from lightning import pytorch as pl

from src.aux.types import MType
from src.training.trainer import Trainer
from src.quantization.dummy.dummy_quant import DummyQuant
from src.models.compose.composer import ModelComposer
from src.data import CIFAR10DataModule
from src.config.config_loader import load_and_validate_config

config = load_and_validate_config("config/dummy_config.yaml")
composer = ModelComposer(config=config)
quantizer = DummyQuant(config=config)

data = CIFAR10DataModule()
data.batch_size_train = config.data.batch_size
data.num_workers = config.data.num_workers

# trainer = pl.Trainer(max_epochs=config.training.epochs)
trainer = Trainer(config=config)

# composer.model = resnet18(num_classes=10)
# composer.criterion = nn.CrossEntropyLoss()
# composer.optimizer = optim.Adam
# composer.model_type = MType.VISION_CLS

model = composer.compose()
qmodel = quantizer.quantize(model)

trainer.fit(qmodel, data)
