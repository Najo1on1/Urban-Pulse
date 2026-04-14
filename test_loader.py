import logging
from src.data_loaders.temporal_dataset import UrbanTemporalDataset

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load the Rotterdam data (a nice middle-sized dataset)
dataset = UrbanTemporalDataset(parquet_path="data/processed/rotterdam_universal_twin.parquet", split="train")

print(f"Total training windows available: {len(dataset)}")

# Grab the very first sample
x, y = dataset[0]

print(f"Input Tensor (X) Shape: {x.shape} -> (Lookback, Features)")
print(f"Target Tensor (Y) Shape: {y.shape} -> (Horizon)")
