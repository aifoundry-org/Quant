import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.config.config_loader import load_and_validate_config
from src.data import CIFAR10DataModule
from src.models.compose.composer import ModelComposer
from src.quantization.quantizer import Quantizer
from src.training.trainer import Trainer


config = load_and_validate_config("config/rniq_config.yaml")
composer = ModelComposer(config=config)
quantizer = Quantizer(config=config)()
trainer = Trainer(config=config)

data = CIFAR10DataModule()
data.batch_size_train = config.data.batch_size
data.num_workers = config.data.num_workers

model = composer.compose()
qmodel = quantizer.quantize(model, in_place=True)

trainer.fit(qmodel, data)
