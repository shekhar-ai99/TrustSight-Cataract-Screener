import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from collections import Counter

# Import from src
from src.model import CataractModel
from src.utils import set_seed

# Set seed for reproducibility
set_seed(42)

# Custom Dataset for binary classification
class CataractDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        self.class_to_idx = {'normal': 0, 'cataract': 1}

        for class_name, label in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append(os.path.join(class_dir, img_name))
                        self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# Transforms matching preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = CataractDataset('data/train', transform=transform)
val_dataset = CataractDataset('data/val', transform=transform)

# Handle class imbalance
train_labels = [label for _, label in train_dataset]
label_counts = Counter(train_labels)
num_normal = label_counts[0]
num_cataract = label_counts[1]
pos_weight = torch.tensor(num_normal / num_cataract) if num_cataract > 0 else torch.tensor(1.0)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Model
model = CataractModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Optimizer and Loss
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.sigmoid(outputs) > 0.5
        train_correct += (preds == labels.byte()).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total
    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            val_correct += (preds == labels.byte()).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    val_loss /= len(val_loader)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

# Save model weights
torch.save(model.backbone.state_dict(), 'final_model.pth')
print('Model saved as final_model.pth')</content>
<parameter name="filePath">/workspaces/TrustSight-Cataract-Screener/scripts/train.py