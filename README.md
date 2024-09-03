# QUANT project

This repository provides a customizable and automated environment for developing and testing different quantization methods.

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
