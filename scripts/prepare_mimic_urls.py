import pandas as pd
from pathlib import Path

MIMIC_DIR = Path(r"E:\CS585-Project\data\raw\mimic")

labels = pd.read_csv(MIMIC_DIR / "mimic-cxr-2.0.0-chexpert.csv")
splits = pd.read_csv(MIMIC_DIR / "mimic-cxr-2.0.0-split.csv")

# merge to get labels for each image, filter to test
df = splits.merge(labels, on=["subject_id", "study_id"], how="inner")
test = df[df["split"] == "test"].reset_index(drop=True)

print(f"Test images: {len(test)}")

# build URLs
BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0"

def build_url(row):
    subj = str(row["subject_id"])
    prefix = "p" + subj[:2]
    return f"{BASE_URL}/files/{prefix}/p{subj}/s{row['study_id']}/{row['dicom_id']}.jpg"

test["url"] = test.apply(build_url, axis=1)

# save URLs to a text file for wget
url_file = MIMIC_DIR / "test_urls.txt"
with open(url_file, "w") as f:
    for url in test["url"]:
        f.write(url + "\n")

print(f"Saved {len(test)} URLs to {url_file}")
print("\nFirst 3 URLs:")
print(test["url"].head(3).tolist())

