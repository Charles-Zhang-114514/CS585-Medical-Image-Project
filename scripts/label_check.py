import torchxrayvision as xrv

model = xrv.models.DenseNet(weights="densenet121-res224-all")

target_labels = ["Pneumothorax", "Effusion"]
label_to_idx = {label: model.pathologies.index(label) for label in target_labels}

print(label_to_idx)