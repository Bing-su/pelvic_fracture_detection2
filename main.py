import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from typer import Option, Typer

from app.dataset import ImageDataModule
from app.module import ImageModel

cli = Typer()


@cli.command()
def train(
    model_name: str = Option(
        "vit_tiny_patch16_224", "--model-name", "-m", help="Model name"
    ),
    optimizer_name: str = Option(
        "madgradw", "--optimizer-name", "-o", help="Optimizer name"
    ),
    learning_rate: float = Option(1e-4, "--learning-rate", "-l", help="Learning rate"),
    weight_decay: float = Option(1e-4, "--weight-decay", "-w", help="Weight decay"),
    epochs: int = Option(5, "--epochs", "-e", min=1, help="Total epochs"),
    batch_size: int = Option(32, "--batch-size", "-b", min=1, help="Batch size"),
    auto_lr_find: bool = Option(False, help="Auto learning rate find"),
    img_size: int = Option(512, "--img-size", "-i", help="Input image size"),
    data_path: str = Option("data/data.csv", "--data-path", "-d", help="Data path"),
):
    model = ImageModel(
        model_name=model_name,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    df = pd.read_csv(data_path)
    used_data = "mura" if "mura" in data_path else "default"
    datamodule = ImageDataModule(df, img_size=img_size, batch_size=batch_size)

    logger = WandbLogger(name="pelvic-fracture-detection-auroc")
    checkpoints = ModelCheckpoint(
        monitor="val_AUROC",
        mode="max",
        filename=f"{model_name}-{used_data}" + "-{epoch:02d}-{val_AUROC:.3f}",
    )

    trainer = pl.Trainer(
        accelerator="auto",
        logger=logger,
        log_every_n_steps=20,
        gpus=1,
        callbacks=[RichProgressBar(), checkpoints],
        precision=16,
        max_epochs=epochs,
        auto_lr_find=auto_lr_find,
    )

    if auto_lr_find:
        trainer.tune(model, datamodule=datamodule)

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    cli()
