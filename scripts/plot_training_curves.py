import os
import matplotlib.pyplot as plt

# PROJECT_ROOT = main project folder
# OUTPUT_DIR = where we save plots
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

# make sure the output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# epochs from 1 to 10
epochs = list(range(1, 11))

# training loss values recorded during training
resnet_losses = [0.3948, 0.3698, 0.3604, 0.3539, 0.3490, 0.3446, 0.3402, 0.3362, 0.3329, 0.3294]
densenet_losses = [0.3958, 0.3745, 0.3648, 0.3582, 0.3533, 0.3489, 0.3446, 0.3408, 0.3377, 0.3336]


# plot the curves
plt.figure(figsize=(8, 5))

plt.plot(epochs, resnet_losses, marker="o", label="ResNet-50")
plt.plot(epochs, densenet_losses, marker="o", label="DenseNet-121")

# basic labels and title
plt.title("Training Loss Curves")
plt.xlabel("Epoch")
plt.ylabel("Train Loss")

plt.xticks(epochs)

# show legend to distinguish models
plt.legend()

plt.tight_layout()
plt.grid(True, linestyle="--", alpha=0.5)


# save as png in outputs/figures
save_path = os.path.join(OUTPUT_DIR, "training_loss_curves.png")
plt.savefig(save_path, dpi=300)

plt.close()

print(f"Saved plot to: {save_path}")