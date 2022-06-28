import pytorch_lightning as pl
import torch
from timm import create_model
from torchmetrics import AUROC

from utils import PESG, AUCMLoss


class ImageModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "vit_tiny_patch16_224",
        optimizer_name: str = "adamw",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.model = create_model(
            model_name, pretrained=True, num_classes=2, in_chans=1, img_size=512
        )
        self.loss_fn = AUCMLoss()

        self.train_AUROC = AUROC(pos_label=1)
        self.val_AUROC = AUROC(pos_label=1)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = PESG(
            self.model,
            a=self.loss_fn.a,
            b=self.loss_fn.b,
            alpha=self.loss_fn.alpha,
            lr=self.learning_rate,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        prob = torch.softmax(y_hat, dim=1)[:, 1]
        loss = self.loss_fn(prob, y)
        self.train_AUROC.update(prob, y)
        self.log_dict(
            {"train_loss": loss, "train_AUROC": self.train_AUROC},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        prob = torch.softmax(y_hat, dim=1)[:, 1]
        loss = self.loss_fn(prob, y)
        self.val_AUROC(prob, y)
        self.log_dict(
            {"val_loss": loss, "val_AUROC": self.val_AUROC},
            prog_bar=True,
            on_epoch=True,
        )
        return {"val_loss": loss, "val_AUROC": self.val_AUROC}
