from .vision_cls.mnist import MNISTDataModule
from .vision_cls.cifar_dali import CIFAR10DALIDataModule
from .vision_cls.cifar import CIFAR10DataModule
__all__ = ["MNISTDataModule", "CIFAR10DataModule", "CIFAR10DALIDataModule"]