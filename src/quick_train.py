import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from collections import Counter
from sklearn.metrics import f1_score
import datetime

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

# Transforms with data augmentation to prevent overfitting
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
data_root = os.path.join(os.path.dirname(__file__), '..', 'Dataset')
print("Loading training dataset...")
train_dataset_full = CataractDataset(os.path.join(data_root, 'training_images'), transform=transform)
print(f"Training dataset loaded: {len(train_dataset_full)} images")

print("Loading test dataset...")
test_dataset_full = CataractDataset(os.path.join(data_root, 'test_extracted'), transform=transform)
print(f"Test dataset loaded: {len(test_dataset_full)} images")

# Combine both datasets
combined_samples = train_dataset_full.samples + test_dataset_full.samples
combined_labels = train_dataset_full.labels + test_dataset_full.labels

class CombinedDataset(Dataset):
    def __init__(self, samples, labels, transform=None):
        self.samples = samples
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# Create combined dataset
combined_dataset = CombinedDataset(combined_samples, combined_labels, transform=transform)
print(f"Combined dataset: {len(combined_dataset)} images total")

# Split into train and val
train_size = int(0.8 * len(combined_dataset))
val_size = len(combined_dataset) - train_size
full_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

# Handle class imbalance
train_indices = full_dataset.indices
train_labels = [combined_labels[i] for i in train_indices]
label_counts = Counter(train_labels)
total_samples = len(train_labels)
class_weights = []
for i in range(4):
    count = label_counts.get(i, 0)
    weight = total_samples / (4 * count) if count > 0 else 1.0
    class_weights.append(weight)
class_weights = torch.tensor(class_weights, dtype=torch.float32)
print(f"Class weights: {class_weights}")

# DataLoaders
train_loader = DataLoader(full_dataset, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# Model
print("Creating model...")
model = CataractModel()
device = torch.device('cpu')  # Use CPU to avoid memory issues
model.to(device)

# Optimizer and Loss
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

# Training loop
num_epochs = 3  # Fewer epochs for quick training
print(f"Starting training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_preds = []
    train_labels_list = []

    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}")
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
    print("Validating...")
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
print("Saving model...")
ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
date_str = datetime.datetime.now(ist).strftime('%Y-%m-%d')
time_str = datetime.datetime.now(ist).strftime('%H-%M-%S')
folder_name = f'model_{date_str}_{time_str}_F1_{val_f1:.4f}'
root_dir = os.path.join(os.path.dirname(__file__), '..', 'test_submission')
folder_path = os.path.join(root_dir, folder_name)
os.makedirs(folder_path, exist_ok=True)
model_path = os.path.join(folder_path, 'model.pth')
torch.save(model, model_path)
print(f'Model saved as {model_path}')

# Also save in root for submission
root_model_path = os.path.join(root_dir, 'model.pth')
torch.save(model, root_model_path)
print('Model also saved as model.pth')
print(f"Training completed: Train F1 {train_f1:.4f}, Val F1 {val_f1:.4f}")
