# MHAQ: Moderately Hackable Quantization framework

## Introduction

This repository provides a customizable and automated environment for developing and testing different quantization methods.

## Features

- **Customizable Quantization:** The framework allows the definition and integration of custom quantization methods. It is designed to work seamlessly with pre-configured data pipelines.

- **Ease of Use:** Built on PyTorch Lightning, providing a streamlined API that simplifies model management and training, while leveraging Lightning’s features for distributed training and logging.

- **Broad Model Support:** Designed to support models across multiple domains, including Computer Vision, Natural Language Processing, and Audio, with flexibility for different architectures.

- **Modular and Hackable Design:** The framework's modular architecture makes it easy to modify and extend, enabling straightforward customization for specific needs.

## Installation

Instructions to setup framework:

1. Clone this repo with

     ```bash
    git clone https://github.com/aifoundry-org/Quant.git
    ```

2. Install project dependencies *(it is better to use wit virtual environment)*

    ```bash
    pip3 install -r requirements.txt
    ```

## Getting started with QAT

### 1. Define pipeline with config

*Config schema placed in `src/config/config_schema.py`*

```python
from src.config.config_loader import load_and_validate_config

config = load_and_validate_config("config/{PATH_TO_YOUR_CONFIG}")
```

### 2. Initialize model to quantize

```python
from src.models.compose.composer import ModelComposer

composer = ModelComposer(config=config)

model = composer.compose()
```

### 3. Initialize trainer

```python
from src.training.trainer import Trainer

trainer = Trainer(config=config)
```

### 4. Initialize quantizer

```python
from src.quantization.quantizer import Quantizer

quantizer = Quantizer(config=config)()
```

### 5. Define your data

```python
from src.data import YOURDATAMODULE

data = YOURDATAMODULE
```

### 6. Quantize and train

```python
qmodel = quantizer.quantize(model, in_place=True)

trainer.fit(qmodel, datamodule=data)
```

## Customization and Hackability

-  **Modifying Quantization Schemes**
    1. In order to add your quantization method, you should create new folder in `src/quantization` folder named after your method.
    2. It is convinient to inherit your quantization class from `BaseQuant` at `src/quantization/abc/abc_quant.py`
    and redefine the abstract methods. However, it's not mandatory and `BaseQuant`
    class may not satisfy your needs.
    3. You can define schema for your quantization config in `src/config/config_schema.py`
    4. After that you should be able to wrap model with your quantization approach
    and perform training/tuning.

    *You can look into implemented Dummy and RNIQ methods to understand it further.*


## Project structure

Execute ```git ls-files | tree --fromfile -d```

```bash
Quant 
├── config  # for training/quant config files
├── data    # for datasets placement
├── scripts # for scripts utilizing the main functionality
├── src
│   ├── aux       # for additional funcs / small utils
│   ├── callbacks # for lightning callbacks
│   ├── config    # for configs handlers
│   ├── data      # for data processing
│   │   └── vision_cls
│   ├── models    # for models definition
│   │   └── compose
│   │       └── vision
│   ├── quantization # for quantization methods definition
│   │   ├── abc
│   │   ├── dummy
│   │   └── rniq
│   └── training    # for specific training regimes
└── tests           # for tests
```
