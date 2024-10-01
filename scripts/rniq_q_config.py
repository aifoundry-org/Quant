import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.config.config_loader import load_and_validate_config
from src.data.composer import DataComposer
from src.models.compose.composer import ModelComposer
from src.quantization.quantizer import Quantizer
from src.training.trainer import Trainer

torch.set_float32_matmul_precision('high')

config = load_and_validate_config("config/rniq_config_resnet20.yaml")
composer = ModelComposer(config=config)
quantizer = Quantizer(config=config)()
trainer = Trainer(config=config)
data_composer = DataComposer(config=config)

data = data_composer.compose()
model = composer.compose()
qmodel = quantizer.quantize(model, in_place=True)

# Test model befor quantization
trainer.test(qmodel, datamodule=data)

# Finetune model
trainer.fit(qmodel, datamodule=data)

# Test model after quantization
trainer.test(model, datamodule=data)
