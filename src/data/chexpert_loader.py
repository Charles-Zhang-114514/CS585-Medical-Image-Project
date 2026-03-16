from pathlib import Path
import pandas as pd


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