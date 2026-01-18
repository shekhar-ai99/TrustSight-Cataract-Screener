import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from collections import Counter
from sklearn.metrics import f1_score

# Import from src
from model import CataractModel
from utils import set_seed

# Set seed for reproducibility
set_seed(42)

# Custom Dataset for 4-class classification
class CataractDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        self.class_to_idx = {
            'No_Cataract': 0,
            'Immature_Cataract': 1,
            'Mature_Cataract': 2,
            'IOL_Inserted': 3
        }

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
        return image, torch.tensor(label, dtype=torch.long)

# Transforms matching preprocessing
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = CataractDataset('data/train', transform=transform)
val_dataset = CataractDataset('data/val', transform=transform)

# Handle class imbalance
train_labels = [label for _, label in train_dataset]
label_counts = Counter(train_labels)
total_samples = len(train_labels)
class_weights = []
for i in range(4):
    count = label_counts.get(i, 0)
    weight = total_samples / (4 * count) if count > 0 else 1.0
    class_weights.append(weight)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

# Model
model = CataractModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Optimizer and Loss
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_preds = []
    train_labels_list = []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_labels_list.extend(labels.cpu().numpy())

    train_f1 = f1_score(train_labels_list, train_preds, average='macro')
    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels_list = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels_list.extend(labels.cpu().numpy())

    val_f1 = f1_score(val_labels_list, val_preds, average='macro')
    val_loss /= len(val_loader)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')

# Save model weights
torch.save(model, 'model.pth')
print('Model saved as model.pth')

# Log the run
# import sys
# import os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# from utils.log_run import log_run
# log_run(f"Training completed: Train F1 {train_f1:.4f}, Val F1 {val_f1:.4f}")
print(f"Logged run outcome at {__import__('datetime').datetime.now()}")