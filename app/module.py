import pytorch_lightning as pl
import torch
from timm import create_model
from torchmetrics import AUROC, F1Score

from utils import PESG, AUCMLoss


class ImageModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "vit_tiny_patch16_224",
        optimizer_name: str = "madgradw",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        img_size: int = 512,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        try:
            self.model = create_model(
                model_name,
                pretrained=True,
                num_classes=2,
                in_chans=1,
                img_size=img_size,
            )
        except TypeError:
            self.model = create_model(
                model_name,
                pretrained=True,
                num_classes=2,
                in_chans=1,
            )
        self.loss_fn = AUCMLoss()

        self.train_AUROC = AUROC(pos_label=1)
        self.train_f1 = F1Score(2, average="macro")
        self.val_AUROC = AUROC(pos_label=1)
        self.val_f1 = F1Score(2, average="macro")

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

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            0.1,
            total_steps=self.trainer.estimated_stepping_batches,
            cycle_momentum=False,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        prob = torch.softmax(y_hat, dim=1)[:, 1]
        loss = self.loss_fn(prob, y)

        self.train_f1.update(y_hat, y)
        self.train_AUROC.update(prob, y)
        self.log_dict(
            {
                "train_loss": loss,
                "train_AUROC": self.train_AUROC,
                "train_f1": self.train_f1,
            },
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

        self.val_f1.update(y_hat, y)
        self.val_AUROC.update(prob, y)
        self.log_dict(
            {"val_loss": loss, "val_AUROC": self.val_AUROC, "val_f1": self.val_f1},
            prog_bar=True,
            on_epoch=True,
        )
        return {"val_loss": loss, "val_AUROC": self.val_AUROC}
