from __future__ import annotations

import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


def get_train_transform(size: int = 512):
    train_transform = T.Compose(
        [
            T.Resize([size, size], T.InterpolationMode.BICUBIC),
            T.RandomInvert(),
            T.RandomAutocontrast(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.445], std=[0.269]),
        ]
    )
    return train_transform


def get_val_transform(size: int = 512):
    val_transform = T.Compose(
        [
            T.Resize([size, size], T.InterpolationMode.BICUBIC),
            T.ToTensor(),
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
        img = Image.open(self.df.loc[idx, "image"]).convert("L")
        img = self.transform(img)
        label = self.df.loc[idx, "label"]
        return img, label


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df_path: str,
        img_size: int = 512,
        batch_size: int = 32,
        num_workers: int = 8,
    ):
        super().__init__()
        self.df_path = df_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None):
        df = pd.read_csv(self.df_path)
        train_df, val_df = train_test_split(
            df, test_size=0.1, random_state=42, stratify=df["label"]
        )
        self.train_dataset = ImageDataset(train_df, self.img_size, train=True)
        self.val_dataset = ImageDataset(val_df, self.img_size, train=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
