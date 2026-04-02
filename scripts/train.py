import sys
from pathlib import Path

# make sure we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score

from src.data.chexpert_loader import CheXpertDataset
from src.models.classifiers import get_resnet50

# ---- config ----
DATA_ROOT = r"E:\CS585-Project\data\raw\chexpert"
TRAIN_CSV = DATA_ROOT + r"\train.csv"
VAL_CSV   = DATA_ROOT + r"\valid.csv"
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ---- transforms ----
train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---- data ----
train_ds = CheXpertDataset(TRAIN_CSV, DATA_ROOT, transform=train_tf)
val_ds   = CheXpertDataset(VAL_CSV, DATA_ROOT, transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f"Train: {len(train_ds)} samples")
print(f"Val:   {len(val_ds)} samples")

# ---- model ----
model = get_resnet50(num_classes=2, pretrained=True)
model = model.to(DEVICE)

# load saved checkpoint
model.load_state_dict(torch.load(str(PROJECT_ROOT / "outputs" / "checkpoints" / "resnet50_best.pth")))
print("Loaded checkpoint")


# ---- loss and optimizer ----
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=LR)

# ---- training loop ----
best_auc = 0.0

for epoch in range(3, NUM_EPOCHS + 1):
    # -- train --
    model.train()
    train_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if (i + 1) % 200 == 0:
            print(f"  Epoch {epoch} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    train_loss /= len(train_loader)

    # -- validate --
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    auc_pneumo = roc_auc_score(all_labels[:, 0], all_probs[:, 0])
    auc_pleural = roc_auc_score(all_labels[:, 1], all_probs[:, 1])
    mean_auc = (auc_pneumo + auc_pleural) / 2

    print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | "
          f"Pneumothorax AUC: {auc_pneumo:.4f} | Pleural Eff AUC: {auc_pleural:.4f} | "
          f"Mean AUC: {mean_auc:.4f}")

    # -- save best model --
    if mean_auc > best_auc:
        best_auc = mean_auc
        torch.save(model.state_dict(), str(PROJECT_ROOT / "outputs" / "checkpoints" / "resnet50_best.pth"))
        print(f"  -> Saved best model (AUC={best_auc:.4f})")

print(f"\nDone! Best Val Mean AUC: {best_auc:.4f}")

