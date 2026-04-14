import torch
torch.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from src.data_loaders.datamodule import MultiCityDataModule
from src.models.transformer import UrbanTransformer
import logging

logging.basicConfig(level=logging.INFO)

def main():
    # 1. The Multi-City Roster (Notice London is missing!)
    training_cities = [
        "data/processed/madrid_universal_twin.parquet",
        "data/processed/zurich_universal_twin.parquet",
        "data/processed/rotterdam_universal_twin.parquet"
    ]
    
    data_module = MultiCityDataModule(city_paths=training_cities, batch_size=128)
    
    model = UrbanTransformer(input_dim=6, d_model=64, nhead=4, horizon=4)
    
    # 2. The Referee
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode="min"
    )
    
    # 3. The Endurance Run
    trainer = pl.Trainer(
        max_epochs=50, # Cranked up to 50
        accelerator="gpu", 
        devices=1,
        precision="16-mixed",
        callbacks=[early_stop_callback],
        log_every_n_steps=10
    )
    
    logging.info("Starting Multi-City GPU Training with Early Stopping...")
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()