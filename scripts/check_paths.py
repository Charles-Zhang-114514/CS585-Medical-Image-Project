from pathlib import Path
import pandas as pd

root_dir = Path(r"E:\585project\data\raw\chexpert")
csv_path = root_dir / "train.csv"

df = pd.read_csv(csv_path)

print("First 5 paths from CSV:")
for p in df["Path"].head():
    print(p)

print("\nCheck if files exist:")
for p in df["Path"].head():
    full_path = root_dir / p
    print(full_path, "->", full_path.exists())