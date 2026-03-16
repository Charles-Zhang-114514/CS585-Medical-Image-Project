import pandas as pd

csv_path = r"E:\585project\data\raw\chexpert\train.csv"
df = pd.read_csv(csv_path)

print("Shape:", df.shape)

print("\nColumns:")
for col in df.columns:
    print(col)

target_cols = ["Pneumothorax", "Pleural Effusion"]

print("\nTarget label summary:")
for col in target_cols:
    print(f"\n{col}")
    print(df[col].value_counts(dropna=False))