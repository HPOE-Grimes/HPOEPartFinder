"""
Train EfficientNet-B0 on part images.

Expects images organized as:
  data/training_images/<Part Name>/image1.jpg
                                   image2.jpg
                                   ...

Run:  python model/train.py
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

BASE = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE, '..', 'data', 'training_images')
CHECKPOINT_DIR = os.path.join(BASE, 'checkpoints')
CLASS_NAMES_PATH = os.path.join(BASE, '..', 'data', 'class_names.json')

EPOCHS = 40
BATCH_SIZE = 16
LR = 1e-4
IMG_SIZE = 224
VAL_SPLIT = 0.2

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def build_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.2, hue=0.08),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def main():
    if not os.path.isdir(IMAGES_DIR) or not os.listdir(IMAGES_DIR):
        print(f"No images found in {IMAGES_DIR}")
        print("Run setup_images.py first, then add your part photos to each folder.")
        return

    train_tf, val_tf = build_transforms()

    full_dataset = datasets.ImageFolder(IMAGES_DIR, transform=train_tf)
    class_names = full_dataset.classes
    n_classes = len(class_names)
    print(f"Found {len(full_dataset)} images across {n_classes} classes.")

    with open(CLASS_NAMES_PATH, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved to {CLASS_NAMES_PATH}")

    n_val = max(1, int(len(full_dataset) * VAL_SPLIT))
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val])

    # Use val transforms for validation split
    val_dataset = datasets.ImageFolder(IMAGES_DIR, transform=val_tf)
    val_set = torch.utils.data.Subset(val_dataset, val_set.indices)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(1280, n_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        train_correct = train_total = 0
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = outputs.argmax(1)
            train_correct += preds.eq(labels).sum().item()
            train_total += labels.size(0)

        # --- Val ---
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                preds = model(inputs).argmax(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS} | loss {train_loss/len(train_loader):.3f} | train {train_acc:.3f} | val {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print(f"  -> Saved best model (val acc {best_val_acc:.3f})")

    print(f"\nDone. Best val accuracy: {best_val_acc:.3f}")
    print("Restart the backend server to load the new model.")


if __name__ == '__main__':
    main()
