import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from torch import nn, optim
from torchvision.models import resnet18
from lightning import pytorch as pl

from src.aux.types import MType
from src.quantization.dummy.dummy_quant import DummyQuant
from src.models.compose.composer import ModelComposer
from src.data import CIFAR10DataModule

composer = ModelComposer()
quantizer = DummyQuant(config={})
data = CIFAR10DataModule()
data.batch_size_train = 2000
trainer = pl.Trainer(max_epochs=25)

composer.model = resnet18(num_classes=10)
composer.criterion = nn.CrossEntropyLoss()
composer.optimizer = optim.Adam
composer.model_type = MType.VISION_CLS

model = composer.compose()
qmodel = quantizer.quantize(model)

trainer.fit(qmodel, data)


