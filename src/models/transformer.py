import torch
import torch.nn as nn
import pytorch_lightning as pl
import math

class PositionalEncoding(nn.Module):
    """Injects sequence order into the Transformer since it processes everything at once."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class UrbanTransformer(pl.LightningModule):
    def __init__(self, input_dim=6, d_model=64, nhead=4, num_layers=2, horizon=4, lookback=48, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Project input features to the Transformer's internal dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 2. The Multi-Head Attention Engine
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Output Regressor (Flattens the memory state to predict the 4 horizon steps)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(lookback * d_model, 128),
            nn.ReLU(),
            nn.Linear(128, horizon)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # x shape: (Batch, 48, 6)
        x = self.input_projection(x) 
        x = self.pos_encoder(x)
        x = self.transformer(x) 
        out = self.regressor(x) 
        return out # Shape: (Batch, 4)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
