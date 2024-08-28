import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torch import nn, optim
from torchvision.models import resnet18

from src.aux.types import MType
from src.quantization.dummy.dummy_quant import DummyQuant
from src.models.compose.composer import ModelComposer

composer = ModelComposer()
quantizer = DummyQuant(config={})

composer.model = resnet18()
composer.criterion = nn.CrossEntropyLoss()
composer.optimizer = optim.Adam
composer.model_type = MType.VISION_CLS

model = composer.compose()
pass

