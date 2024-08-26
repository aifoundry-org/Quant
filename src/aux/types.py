from enum import Enum

class MType(Enum):
    # Vision-based model types
    VISION_CLS = 1
    VISION_SR = 2
    VISION_DNS = 3
    
    # Language-based model types
    LM = 10
    