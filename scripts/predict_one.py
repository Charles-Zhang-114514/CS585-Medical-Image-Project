import numpy as np
import torch
import torchvision
import torchxrayvision as xrv

model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.eval()

transform = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(224),
])

# fake grayscale image, just for pipeline shape test
img = np.zeros((1, 256, 256), dtype=np.float32)
img = transform(img)
img = torch.from_numpy(img).unsqueeze(0)

with torch.no_grad():
    out = model(img)

pneumothorax_idx = model.pathologies.index("Pneumothorax")
effusion_idx = model.pathologies.index("Effusion")

print("Pneumothorax score:", out[0, pneumothorax_idx].item())
print("Effusion score:", out[0, effusion_idx].item())