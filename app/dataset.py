from __future__ import annotations

import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.io import ImageReadMode, read_image


def get_train_transform(size: int = 512):
    train_transform = T.Compose(
        [
            T.RandomInvert(),
            T.RandomAutocontrast(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30, T.InterpolationMode.BICUBIC),
            T.Resize(size),
            T.CenterCrop(size),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.445], std=[0.269]),
        ]
    )
    return train_transform


def get_val_transform(size: int = 512):
    val_transform = T.Compose(
        [
            T.Resize(size),
            T.CenterCrop(size),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.445], std=[0.269]),
        ]
    )
    return val_transform


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, size: int = 512, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.train = train
        self.transform = get_train_transform(size) if train else get_val_transform(size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        img = read_image(self.df.loc[idx, "image"], read_mode=ImageReadMode.GRAY)
        img = self.transform(img)
        label = self.df.loc[idx, "label"]
        return img, label


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df: pd.DataFrame,
        img_size: int = 512,
        batch_size: int = 64,
        num_workers: int = 8,
    ):
        super().__init__()
        self.df = df
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None):
        train_df, val_df = train_test_split(
            self.df, test_size=0.1, random_state=42, stratify=self.df["label"]
        )
        self.train_dataset = ImageDataset(train_df, self.img_size, train=True)
        self.val_dataset = ImageDataset(val_df, self.img_size, train=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
