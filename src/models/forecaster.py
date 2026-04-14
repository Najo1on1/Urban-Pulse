import torch
import torch.nn as nn
import pytorch_lightning as pl

class BaselineForecaster(pl.LightningModule):
    def __init__(self, input_dim=6, hidden_dim=64, horizon=4, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # The sequential memory engine
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        # The output layer mapping to our 4-step horizon
        self.regressor = nn.Linear(hidden_dim, horizon)
        
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # x shape: (Batch, 48, 6)
        _, hidden = self.rnn(x) 
        # hidden shape: (1, Batch, 64) -> We squeeze the layer dimension out
        out = self.regressor(hidden.squeeze(0))
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
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
