from .vision_cls.mnist import MNISTDataModule as MNIST
from .vision_cls.cifar_dali import CIFAR10DALIDataModule as CIFAR10_DALI
from .vision_cls.cifar import CIFAR10DataModule as CIFAR10
__all__ = ["MNIST", "CIFAR10", "CIFAR10_DALI"]