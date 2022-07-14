from typing import Optional

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
        "densenet121",
        "--model-name",
        "-m",
        help="Model name",
        rich_help_panel="Model",
    ),
    optimizer_name: str = Option(
        "madgradw",
        "--optimizer-name",
        "-o",
        help="Optimizer name",
        rich_help_panel="Model",
    ),
    learning_rate: float = Option(
        1e-3, "--learning-rate", "-l", help="Learning rate", rich_help_panel="Training"
    ),
    weight_decay: float = Option(
        1e-4, "--weight-decay", "-w", help="Weight decay", rich_help_panel="Training"
    ),
    epochs: int = Option(
        5, "--epochs", "-e", min=1, help="Total epochs", rich_help_panel="Training"
    ),
    batch_size: int = Option(
        32, "--batch-size", "-b", min=1, help="Batch size", rich_help_panel="Training"
    ),
    auto_lr_find: bool = Option(
        False, help="Auto learning rate find", rich_help_panel="Training"
    ),
    img_size: int = Option(
        512, "--img-size", "-i", help="Input image size", rich_help_panel="Model"
    ),
    data_path: str = Option("data/data.csv", "--data-path", "-d", help="Data path"),
    project_name: Optional[str] = Option(
        None, "--project-name", "-p", help="Project name"
    ),
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

    if not project_name:
        project_name = f"{model_name}_{used_data}"

    logger = WandbLogger(name=project_name)
    checkpoints = ModelCheckpoint(
        monitor="val_AUROC",
        mode="max",
        filename=f"{model_name}-{used_data}" + "-{epoch:02d}-{val_AUROC:.3f}",
    )

    trainer = pl.Trainer(
        accelerator="auto",
        logger=logger,
        log_every_n_steps=30,
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
