import sys
import torch
import torchxrayvision as xrv

print("Python executable:", sys.executable)
print("torch:", torch.__version__)
print("torchxrayvision:", xrv.__version__)

model = xrv.models.DenseNet(weights="densenet121-res224-all")
print("Loaded model successfully.")

print("Pathologies:")
for i, p in enumerate(model.pathologies):
    print(i, p)
    
dummy = torch.zeros((1, 1, 224, 224))
with torch.no_grad():
    out = model(dummy)

print("Output shape:", out.shape)
print("First output vector:", out[0])