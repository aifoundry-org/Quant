import os
import sys
import torch
from typing import List, Optional
from flytekit import task, workflow, Resources

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.config.config_loader import load_and_validate_config
from src.data.composer import DataComposer
from src.models.compose.composer import ModelComposer
from src.quantization.quantizer import Quantizer
from src.training.trainer import Trainer

torch.set_float32_matmul_precision('high')

PRECONFIGURED_CONFIGS = [
    "config/rniq_config_resnet20.yaml",
]

# Flyte task to load and validate the config
@task(limits=Resources(cpu="1", mem="1Gi"))
def load_config(selected_config: Optional[str] = None) -> dict:
    if selected_config:
        config_path = selected_config
    else:
        config_path = PRECONFIGURED_CONFIGS[0]
    
    return load_and_validate_config(config_path)

@task
def initialize_composer(config):
    return ModelComposer(config=config)

@task
def initialize_quantizer(config):
    return Quantizer(config=config)()

@task
def initialize_trainer(config):
    return Trainer(config=config)

@task
def initialize_data_composer(config):
    return DataComposer(config=config)

@task
def compose_data(data_composer):
    return data_composer.compose()

@task
def compose_model(composer):
    return composer.compose()

@task
def quantize_model(quantizer, model):
    return quantizer.quantize(model, in_place=True)

@task
def test_model(trainer, model, data):
    trainer.test(model, datamodule=data)

@task
def fit_model(trainer, model, data):
    trainer.fit(model, datamodule=data)

@workflow
def model_quantization_workflow(selected_config: Optional[str] = None):
    config = load_config(selected_config=selected_config)
    composer = initialize_composer(config=config)
    quantizer = initialize_quantizer(config=config)
    trainer = initialize_trainer(config=config)
    data_composer = initialize_data_composer(config=config)

    data = compose_data(data_composer=data_composer)
    model = compose_model(composer=composer)
    qmodel = quantize_model(quantizer=quantizer, model=model)

    test_model(trainer=trainer, model=qmodel, data=data)
    fit_model(trainer=trainer, model=qmodel, data=data)
    test_model(trainer=trainer, model=model, data=data)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    
    selected_config = PRECONFIGURED_CONFIGS[0]
    
    model_quantization_workflow(selected_config=selected_config)
