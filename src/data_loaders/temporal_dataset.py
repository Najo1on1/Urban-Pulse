import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import logging

class UrbanTemporalDataset(Dataset):
    def __init__(self, parquet_path: str, split: str = 'train'):
        """
        Loads the Universal Twin matrix, extracts cyclical time features, 
        and enforces strict chronological splitting to prevent future leakage.
        """
        self.split = split
        # 12 hours lookback (48 intervals of 15m) | 1 hour horizon (4 intervals)
        self.lookback = 48 
        self.horizon = 4
        self.window_size = self.lookback + self.horizon
        
        # Load the data
        logging.info(f"Loading {split} data from {parquet_path}")
        self.df = pd.read_parquet(parquet_path)
        self.df.index = pd.to_datetime(self.df.index)
        self.timestamps = self.df.index
        
        # 1. Feature Engineering: The Contextual Signals
        self.features = self._generate_time_features()
        
        # 2. Chronological Splitting (Preventing Data Leakage)
        self._apply_temporal_split()
        
        # Extract raw flow values as a numpy array for fast tensor slicing
        self.flow_data = self.df.values.astype(np.float32)
        
        # Calculate valid starting indices for the sliding window
        self.num_samples = len(self.df) - self.window_size
        self.num_segments = self.flow_data.shape[1]

    def _generate_time_features(self):
        """Creates cyclical Sine/Cosine embeddings and Weekend flags."""
        features = pd.DataFrame(index=self.timestamps)
        
        # Day of week (0-6) mapped to a circle
        day_of_week = self.timestamps.dayofweek
        features['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
        features['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7.0)
        
        # Hour of day (0-23) + minute adjustments mapped to a circle
        hour_float = self.timestamps.hour + (self.timestamps.minute / 60.0)
        features['hour_sin'] = np.sin(2 * np.pi * hour_float / 24.0)
        features['hour_cos'] = np.cos(2 * np.pi * hour_float / 24.0)
        
        # Weekend Flag (1 if Saturday/Sunday, 0 otherwise)
        features['is_weekend'] = (day_of_week >= 5).astype(np.float32)
        
        return features.values.astype(np.float32)

    def _apply_temporal_split(self):
        """Splits data strictly by time: 70% Train, 15% Val, 15% Test."""
        n = len(self.df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        if self.split == 'train':
            self.df = self.df.iloc[:train_end]
            self.features = self.features[:train_end]
        elif self.split == 'val':
            self.df = self.df.iloc[train_end:val_end]
            self.features = self.features[train_end:val_end]
        elif self.split == 'test':
            self.df = self.df.iloc[val_end:]
            self.features = self.features[val_end:]
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'.")

    def __len__(self):
        # Total sliding windows = (Timesteps - Window Size) * Number of Road Segments
        if self.num_samples <= 0:
            return 0
        return self.num_samples * self.num_segments

    def __getitem__(self, idx):
        # Decode the 1D index into a 2D position (time_idx, segment_idx)
        time_idx = idx // self.num_segments
        segment_idx = idx % self.num_segments
        
        # Slicing indices
        lookback_end = time_idx + self.lookback
        horizon_end = lookback_end + self.horizon
        
        # Extract Flow Data for the specific road segment
        # Shape: (48, 1) for X, (4, 1) for Y
        x_flow = self.flow_data[time_idx:lookback_end, segment_idx].reshape(-1, 1)
        y_flow = self.flow_data[lookback_end:horizon_end, segment_idx]
        
        # Extract Time Features (Shared across all segments for this time window)
        # Shape: (48, 5)
        x_time = self.features[time_idx:lookback_end]
        
        # Combine Flow and Time features into a single input tensor
        # Final X Shape: (48, 6) -> 1 flow feature + 5 time features
        x_combined = np.concatenate([x_flow, x_time], axis=1)
        
        return torch.tensor(x_combined), torch.tensor(y_flow)
