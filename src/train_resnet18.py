from pathlib import Path
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# --------------------------------------------------
# 1. Project settings
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_ROOT / "data" / "NEU-DET" / "train" / "images"
VAL_DIR = PROJECT_ROOT / "data" / "NEU-DET" / "validation" / "images"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "defect_resnet18.pth"

BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.001
SEED = 42

DEVICE = torch.device("cpu")


# --------------------------------------------------
# 2. Reproducibility
# --------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# --------------------------------------------------
# 3. Image transformations
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# 4. Load datasets
# --------------------------------------------------

train_dataset = datasets.ImageFolder(
    root=str(TRAIN_DIR),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=str(VAL_DIR),
    transform=val_transform
)

class_names = train_dataset.classes

if train_dataset.class_to_idx != val_dataset.class_to_idx:
    raise ValueError(
        "Training and validation class mappings do not match."
    )

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Classes:", class_names)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Device:", DEVICE)


# --------------------------------------------------
# 5. Load pretrained ResNet18
# --------------------------------------------------

print("\nLoading pretrained ResNet18...")

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Freeze the pretrained feature extractor.
for parameter in model.parameters():
    parameter.requires_grad = False

# Replace the final layer with our six defect classes.
input_features = model.fc.in_features
model.fc = nn.Linear(input_features, len(class_names))

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

# Train only the newly created classification layer.
optimizer = Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# 6. Training and validation
# --------------------------------------------------

best_val_accuracy = 0.0
best_state = copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):

    # Training
    model.train()

    # Keep the frozen feature extractor in evaluation mode.
    model.eval()
    model.fc.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= train_total
    train_accuracy = train_correct / train_total

    # Validation
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_accuracy = val_correct / val_total

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2%} "
        f"Validation Loss: {val_loss:.4f} "
        f"Validation Accuracy: {val_accuracy:.2%}"
    )

    # Save the best validation checkpoint.
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_state = copy.deepcopy(model.state_dict())

        torch.save(
            {
                "model_state_dict": best_state,
                "classes": class_names,
                "image_size": 224,
                "best_val_accuracy": best_val_accuracy,
                "model_name": "ResNet18",
            },
            MODEL_PATH
        )


# --------------------------------------------------
# 7. Final summary
# --------------------------------------------------

print("\nTraining completed!")
print(f"Best validation accuracy: {best_val_accuracy:.2%}")
print(f"Saved model to: {MODEL_PATH}")