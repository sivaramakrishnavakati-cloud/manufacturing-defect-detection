from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VAL_DIR = PROJECT_ROOT / "data" / "NEU-DET" / "validation" / "images"
MODEL_PATH = PROJECT_ROOT / "models" / "defect_resnet18.pth"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
BATCH_SIZE = 16

# Match the preprocessing used during ResNet18 training.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

dataset = datasets.ImageFolder(
    root=str(VAL_DIR),
    transform=transform,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

class_names = dataset.classes

# Load the saved checkpoint.
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

# Re-create ResNet18 with the correct number of output classes.
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in loader:
        outputs = model(images.to(DEVICE))
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.cpu().tolist())

print("\nRESNET18 CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=3,
        zero_division=0,
    )
)

matrix = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names))),
)

print("\nCONFUSION MATRIX")
print("Rows = actual class; columns = predicted class")
print(matrix)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=class_names,
)

display.plot(xticks_rotation=45, values_format="d")
plt.title("ResNet18 Steel Defect Classification")
plt.tight_layout()

output_path = RESULTS_DIR / "resnet18_confusion_matrix.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")

print(f"\nConfusion matrix saved to: {output_path}")

plt.show()