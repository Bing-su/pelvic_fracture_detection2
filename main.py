from typing import Optional

import pytorch_lightning as pl
from loguru import logger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from typer import Argument, Option, Typer

from app.dataset import ImageDataModule
from app.module import ImageModel

cli = Typer(pretty_exceptions_show_locals=False)


@cli.command(no_args_is_help=True)
def train(
    model_name: str = Argument(
        ...,
        help="Model name from timm",
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
        1e-4, "--learning-rate", "-l", help="Learning rate", rich_help_panel="Training"
    ),
    weight_decay: float = Option(
        1e-5, "--weight-decay", "-w", help="Weight decay", rich_help_panel="Training"
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
    fast_dev_run: bool = Option(False, help="run test", rich_help_panel="Training"),
    data_path: str = Option(
        "data/data.csv", "--data-path", "-d", help="path of data csv file"
    ),
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
    logger.debug("model ready.")

    datamodule = ImageDataModule(data_path, img_size=img_size, batch_size=batch_size)
    logger.debug("datamodule ready.")

    if not project_name:
        project_name = f"{model_name}_{learning_rate}"

    logger.info(f"project name: {project_name}")

    wandb_logger = WandbLogger(name=project_name)
    checkpoints = ModelCheckpoint(
        monitor="val_AUROC",
        mode="max",
        filename=f"{model_name}-{learning_rate}" + "-{epoch:02d}-{val_AUROC:.3f}",
    )

    trainer = pl.Trainer(
        accelerator="auto",
        logger=wandb_logger,
        log_every_n_steps=50,
        devices=1,
        callbacks=[RichProgressBar(), LearningRateMonitor("step"), checkpoints],
        precision=16,
        max_epochs=epochs,
        auto_lr_find=auto_lr_find,
        fast_dev_run=fast_dev_run,
    )

    if auto_lr_find:
        logger.info("Learning rate finder started.")
        trainer.tune(model, datamodule=datamodule)

    logger.info("Training started.")
    trainer.fit(model, datamodule=datamodule)
    logger.info("Training finished.")


if __name__ == "__main__":
    cli()
