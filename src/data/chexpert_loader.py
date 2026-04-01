from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CheXpertMeta:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)

    def show_basic_info(self):
        print("CSV path:", self.csv_path)
        print("Shape:", self.df.shape)

    def show_columns(self):
        print("\nColumns:")
        for col in self.df.columns:
            print(col)

    def target_summary(self, target_cols=None):
        if target_cols is None:
            target_cols = ["Pneumothorax", "Pleural Effusion"]

        print("\nTarget label summary:")
        for col in target_cols:
            print(f"\n{col}")
            print(self.df[col].value_counts(dropna=False))
            
class CheXpertDataset(Dataset):
    TARGET_COLS = ["Pneumothorax", "Pleural Effusion"]

    def __init__(self, csv_path: str, data_root: str, transform=None):
        self.data_root = Path(data_root)
        self.transform = transform

        df = pd.read_csv(csv_path)

        # keep frontal views only
        df = df[df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)

        # U-Ones policy: NaN -> 0, uncertain(-1) -> positive(1)
        for col in self.TARGET_COLS:
            df[col] = df[col].fillna(0.0).replace(-1.0, 1.0).astype(np.float32)

        # fix path: remove "CheXpert-v1.0-small/" prefix
        df["image_path"] = df["Path"].apply(
            lambda p: str(self.data_root / p.replace("CheXpert-v1.0-small/", ""))
        )

        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
    
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
    
        labels = torch.tensor(
            [row[col] for col in self.TARGET_COLS],
            dtype=torch.float32
        )
    
        return image, labels