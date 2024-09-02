import torch
from lightning import pytorch as pl

from torchvision.datasets import CIFAR10
from torchvision import transforms

from torch.utils.data import random_split, DataLoader


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data") -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size_train = 32
        self.batch_size_test = 32
        self.num_workers = 5

        self.transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                self._normalize(),
            ]
        )

        self.transform_test = transforms.Compose(
            [transforms.ToTensor(), self._normalize()]
        )

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            cifar_data = CIFAR10(
                self.data_dir, train=True, transform=self.transform_train
            )
            self.cifar_train, self.cifar_val = random_split(
                cifar_data, [45000, 5000], generator=torch.Generator().manual_seed(42)
            )

        if stage == "test":
            self.cifar_test = CIFAR10(
                self.data_dir, train=False, transform=self.transform_test
            )

        if stage == "predict":
            self.cifar_test = CIFAR10(
                self.data_dir, train=False, transform=self.transform_test
            )

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_val,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.batch_size_test,
            num_workers=self.num_workers,
        )

    @staticmethod
    def _normalize():
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
