from lightning import pytorch as pl
from src import data as datamodules

class DataComposer():
    def __init__(self, config=None) -> None:
        self.config = config
    
    def compose(self) -> pl.LightningDataModule:
        data_config = self.config.data
        datamodule = getattr(datamodules, data_config.dataset_name)()
        datamodule.batch_size = data_config.batch_size
        datamodule.num_workers = data_config.num_workers
        
        return datamodule