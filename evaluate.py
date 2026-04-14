import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.data_loaders.temporal_dataset import UrbanTemporalDataset
from src.models.transformer import UrbanTransformer
import logging

torch.set_float32_matmul_precision('medium')

logging.basicConfig(level=logging.INFO)

def main():
    # 1. Load the unseen London Data
    logging.info("Loading Zero-Shot Test City: London")
    london_dataset = UrbanTemporalDataset(parquet_path="data/processed/london_universal_twin.parquet", split='test')
    london_loader = DataLoader(london_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    # 2. Load the Best Model from Phase 2
    checkpoint_path = "lightning_logs/version_3/checkpoints/epoch=19-step=64900.ckpt" 
    
    logging.info(f"Loading trained weights from: {checkpoint_path}")
    model = UrbanTransformer.load_from_checkpoint(checkpoint_path)
    
    # 3. Evaluate
    trainer = pl.Trainer(accelerator="gpu", devices=1)
    logging.info("Running Zero-Shot Inference on London...")
    results = trainer.test(model, dataloaders=london_loader)
    
    print("\n--- ZERO-SHOT TEST RESULTS (LONDON) ---")
    print(f"Mean Squared Error (MSE): {results[0]['test_loss']:.4f}")

if __name__ == "__main__":
    main()
