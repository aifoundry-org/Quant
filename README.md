# QUANT project

This repository provides a customizable and automated environment for developing and testing different quantization methods.

## Performance Metrics

| Model     | Dataset  | Method | QW | QA | Best Top-1 |
|-----------|----------|--------|----|----|------------|
| Resnet-20 | CIFAR-10 | FP     | -  | -  | 91.7%      |
| Resnet-20 | CIFAR-10 | RNIQ   | 2  | -  | 90.5%      |
| Resnet-20 | CIFAR-10 | [link](https://github.com/linkinpark213/quantization-networks-cifar10/tree/master)   | 2  | -  | 91.1%      |


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
│   │   └── dummy
│   └── training    # for specific training regimes
└── tests           # for tests
```
