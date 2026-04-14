import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from src.data_loaders.temporal_dataset import UrbanTemporalDataset

class MultiCityDataModule(pl.LightningDataModule):
    def __init__(self, city_paths: list, batch_size: int = 128):
        super().__init__()
        self.city_paths = city_paths
        self.batch_size = batch_size

    def setup(self, stage=None):
        train_datasets, val_datasets, test_datasets = [], [], []
        
        # Load and split each city independently to prevent chronological leakage
        for path in self.city_paths:
            train_datasets.append(UrbanTemporalDataset(path, split='train'))
            val_datasets.append(UrbanTemporalDataset(path, split='val'))
            test_datasets.append(UrbanTemporalDataset(path, split='test'))
            
        # Stitch them together into massive shared datasets
        self.train_dataset = ConcatDataset(train_datasets)
        self.val_dataset = ConcatDataset(val_datasets)
        self.test_dataset = ConcatDataset(test_datasets)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)