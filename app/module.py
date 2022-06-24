import pytorch_lightning as pl
from timm import create_model
from timm.optim import create_optimizer_v2
from torch import nn


class ImageModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "resnet50",
        optimizer_name: str = "madgradw",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.model = create_model(
            model_name, pretrained=True, num_classes=2, in_chans=1
        )
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss
