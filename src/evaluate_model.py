from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VAL_DIR = PROJECT_ROOT / "data" / "NEU-DET" / "validation" / "images"
MODEL_PATH = PROJECT_ROOT / "models" / "defect_classifier.pth"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 128
BATCH_SIZE = 16
DEVICE = torch.device("cpu")


# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Load validation dataset
dataset = datasets.ImageFolder(
    root=str(VAL_DIR),
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class_names = dataset.classes


# Define the same CNN architecture used during training
class DefectCNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Load trained model
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model = DefectCNN(len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()


# Generate predictions
all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in loader:

        outputs = model(images.to(DEVICE))
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy().tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())


# Print classification metrics
print("\nCLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=3,
        zero_division=0
    )
)


# Calculate and display confusion matrix
matrix = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names)))
)

print("\nCONFUSION MATRIX")
print("Rows = actual class; columns = predicted class")
print(matrix)


display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=class_names
)

display.plot(
    xticks_rotation=45,
    values_format="d"
)

plt.title("Steel Defect Classification - Confusion Matrix")
plt.tight_layout()

output_path = RESULTS_DIR / "confusion_matrix.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")

print(f"\nConfusion matrix saved to: {output_path}")

plt.show()