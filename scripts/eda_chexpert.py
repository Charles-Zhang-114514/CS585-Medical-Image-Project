import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Path setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "CheXpert"))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

# CSV files for train and validation
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
VALID_CSV = os.path.join(DATA_ROOT, "valid.csv")

# Making sure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS = ["Pneumothorax", "Pleural Effusion"]


# Label distribution
def count_label_states(df, split_name):
    for label in LABELS:
        series = df[label]

        # Count each type of label value
        counts = {
            "Positive (1.0)": (series == 1.0).sum(),
            "Negative (0.0)": (series == 0.0).sum(),
            "Uncertain (-1.0)": (series == -1.0).sum(),
            "Blank (NaN)": series.isna().sum()
        }

        # Printing the results
        print(f"\n{split_name} - {label}")
        for k, v in counts.items():
            print(f"{k}: {v}")

        # Plot as bar chart
        plt.figure(figsize=(8, 5))
        plt.bar(counts.keys(), counts.values())
        plt.title(f"{split_name}: {label} Distribution")
        plt.ylabel("Count")
        plt.xticks(rotation=20)
        plt.tight_layout()

        # Save figure
        filename = f"{split_name.lower()}_{label.lower().replace(' ', '_')}_distribution.png"
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()



# View distribution
def count_view_types(df, split_name):
    # Count frontal vs lateral
    counts = df["Frontal/Lateral"].value_counts(dropna=False)

    print(f"\n{split_name} - View Type")
    print(counts)

    # Bar plot
    plt.figure(figsize=(6, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title(f"{split_name}: Frontal vs Lateral")
    plt.ylabel("Count")
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, f"{split_name.lower()}_view_distribution.png"))
    plt.close()



# Co-occurrence
def cooccurrence_table(df, split_name):
    # Check where each condition is positive
    pneumo = df["Pneumothorax"] == 1.0
    effusion = df["Pleural Effusion"] == 1.0

    both = (pneumo & effusion).sum()
    neither = (~pneumo & ~effusion).sum()

    print(f"\n{split_name} - Co-occurrence")
    print(f"Both positive: {both}")
    print(f"Neither: {neither}")

    # Build 2x2 table
    table = pd.DataFrame({
        "Pleural Effusion = 1": [
            (pneumo & effusion).sum(),
            (~pneumo & effusion).sum()
        ],
        "Pleural Effusion != 1": [
            (pneumo & ~effusion).sum(),
            (~pneumo & ~effusion).sum()
        ]
    }, index=["Pneumothorax = 1", "Pneumothorax != 1"])

    print(table)

    # Save table as csv
    table.to_csv(os.path.join(OUTPUT_DIR, f"{split_name.lower()}_cooccurrence.csv"))



# Sample images
def visualize_samples(df, split_name, n=5):
    # Looking only at frontal images
    frontal_df = df[df["Frontal/Lateral"] == "Frontal"].copy()
    
    # Randomly pick a few examples
    samples = frontal_df.sample(n=min(n, len(frontal_df)), random_state=42)

    plt.figure(figsize=(15, 4))

    for i, (_, row) in enumerate(samples.iterrows(), start=1):
        img_rel_path = row["Path"]

        img_rel_path = img_rel_path.replace("CheXpert-v1.0-small/", "")

        img_path = os.path.join(DATA_ROOT, img_rel_path)

        # Skip if image missing
        if not os.path.exists(img_path):
            print("Missing image:", img_path)
            continue

        # Load grayscale image
        img = Image.open(img_path).convert("L")

        plt.subplot(1, len(samples), i)
        plt.imshow(img, cmap="gray")
        plt.title(
            f"P:{row['Pneumothorax']}\nE:{row['Pleural Effusion']}",
            fontsize=9
        )
        plt.axis("off")

    plt.suptitle(f"{split_name} - Sample Frontal Images")
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, f"{split_name.lower()}_samples.png"))
    plt.close()


# Main Function
def main():
    print("Loading data...")

    train_df = pd.read_csv(TRAIN_CSV)
    valid_df = pd.read_csv(VALID_CSV)

    print("Train shape:", train_df.shape)
    print("Valid shape:", valid_df.shape)

    # Running each part
    count_label_states(train_df, "Train")
    count_label_states(valid_df, "Valid")

    count_view_types(train_df, "Train")
    count_view_types(valid_df, "Valid")
    
    cooccurrence_table(train_df, "Train")
    cooccurrence_table(valid_df, "Valid")

    visualize_samples(train_df, "Train")
    visualize_samples(valid_df, "Valid")

    print("\nEDA complete. Check outputs/figures/")


if __name__ == "__main__":
    main()