"""
TRAINING SPEED OPTIMIZATIONS (does NOT affect inference contract):
1. Reduced batch_size from 8 → 4 for better CPU cache utilization
2. Added batch-level progress logging to show training is not frozen
3. Optimized Dataset.__getitem__ to avoid unnecessary ast.literal_eval
4. Added optional DEBUG_SUBSAMPLE flag for fast prototyping (disabled by default)
5. Added comments for num_workers optimization (kept at 0 for safety)

Original backup saved as: train_cpu_original.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import ast
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import datetime
import time

# Focal Loss - REMOVED (see train.py line 472: using CrossEntropyLoss instead)

# MixUp Augmentation
from mixup import mixup, mixup_criterion, should_skip_mixup

# Setup logging to file
class TeeLogger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# Create run folder and setup logging
ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
run_timestamp = datetime.datetime.now(ist).strftime('%Y-%m-%d_%H-%M-%S')
run_folder = os.path.join(os.path.dirname(__file__), '..', 'run_reports', f'run_{run_timestamp}')
os.makedirs(run_folder, exist_ok=True)
log_file_path = os.path.join(run_folder, 'training.log')

# Redirect stdout to both console and log file
tee_logger = TeeLogger(log_file_path)
sys.stdout = tee_logger

print(f"Training run started at: {run_timestamp}")
print(f"Logs will be saved to: {log_file_path}\n")

# Import from src
from model import CataractModel
from utils import set_seed
from config import (
    DRY_RUN, DRY_RUN_SAMPLES, DRY_RUN_EPOCHS,
    DEBUG_SUBSAMPLE, DEBUG_SAMPLE_SIZE,
    NUM_EPOCHS, BATCH_SIZE, NUM_WORKERS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, SCHEDULER_FACTOR, SCHEDULER_PATIENCE,
    TRAIN_RESOLUTION, VAL_RESOLUTION,
    AUGMENTATION_BRIGHTNESS, AUGMENTATION_CONTRAST, AUGMENTATION_BRIGHTNESS_P,
    AUGMENTATION_BLUR_SIGMA, AUGMENTATION_BLUR_P,
    AUGMENTATION_MOTION_BLUR, AUGMENTATION_MOTION_BLUR_P,
    AUGMENTATION_JPEG_QUALITY_LOWER, AUGMENTATION_JPEG_P,
    AUGMENTATION_SHADOW_P, AUGMENTATION_ROTATION, AUGMENTATION_ROTATION_P,
    AUGMENTATION_CLAHE_P, AUGMENTATION_CLAHE_CLIP,
    IMMATURE_EXTRA_AUG, IMMATURE_COLOR_JITTER_BRIGHTNESS, IMMATURE_COLOR_JITTER_CONTRAST,
    IMMATURE_GAUSSIAN_BLUR_KERNEL, IMMATURE_GAUSSIAN_BLUR_SIGMA,
    CLASS_WEIGHTS, FREEZE_BACKBONE_UNTIL_EPOCH, UNFREEZE_LR, CLASS_WEIGHT_POWER,
    LABEL_SMOOTHING, GRADIENT_ACCUMULATION_STEPS, RANDOM_SEED,
    PROGRESS_LOG_INTERVAL, SLOW_EPOCH_THRESHOLD, DEVICE, PARQUET_DATASET, SUBMISSION_DIR,
    CLASS_WEIGHT_MIN, CLASS_WEIGHT_MAX, MIXUP_PROB, IOL_SAMPLING_FRACTION
)

# Set seed for reproducibility
set_seed(RANDOM_SEED)

# ============ CONFIG LOADED FROM config.py ============
# All configuration is now centralized in config.py for easy modification
print(f"\n{'='*60}")
print("CONFIGURATION LOADED FROM config.py")
print(f"{'='*60}")
print(f"  DRY_RUN: {DRY_RUN}")
print(f"  NUM_EPOCHS: {NUM_EPOCHS}")
print(f"  BATCH_SIZE: {BATCH_SIZE}")
print(f"  LEARNING_RATE: {LEARNING_RATE}")
print(f"  TRAIN_RESOLUTION: {TRAIN_RESOLUTION}")
print(f"  EARLY_STOPPING_PATIENCE: {EARLY_STOPPING_PATIENCE}")
print(f"  FREEZE_BACKBONE_UNTIL_EPOCH: {FREEZE_BACKBONE_UNTIL_EPOCH}")
print(f"  LABEL_SMOOTHING: {LABEL_SMOOTHING}")
print(f"  GRADIENT_ACCUMULATION_STEPS: {GRADIENT_ACCUMULATION_STEPS}")
print(f"{'='*60}\n")

# Custom Dataset for 4-class classification
class CataractDataset(Dataset):
    def __init__(self, parquet_path, transform=None, is_train=True):
        self.transform = transform
        self.is_train = is_train  # FIX-4: Track whether this is training or validation
        self.df = pd.read_parquet(parquet_path)
        # EXACT label mapping (canonical form with spaces)
        self.class_to_idx = {
            'No Cataract': 0,
            'Immature Cataract': 1,
            'Mature Cataract': 2,
            'IOL Inserted': 3
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Accept either "label" (our format) or "Cataract Type" (organizer format) with underscore variants
        candidate_label_cols = [
            'label', 'Cataract Type', 'cataract type', 'Cataract_Type', 'cataract_type'
        ]
        self.label_column = None
        for col in candidate_label_cols:
            if col in self.df.columns:
                self.label_column = col
                break
        if self.label_column is None:
            raise KeyError(
                "No label column found. Expected one of: "
                f"{candidate_label_cols}. Found: {list(self.df.columns)}"
            )

        # Map lowercased/underscore variants to canonical labels
        self.label_aliases = {
            'no cataract': 'No Cataract',
            'no_cataract': 'No Cataract',
            'immature cataract': 'Immature Cataract',
            'immature_cataract': 'Immature Cataract',
            'mature cataract': 'Mature Cataract',
            'mature_cataract': 'Mature Cataract',
            'iol inserted': 'IOL Inserted',
            'iol_inserted': 'IOL Inserted',
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get label from accepted columns and normalize aliases/underscores
        raw_label = row[self.label_column]
        label_key = str(raw_label).replace('_', ' ').strip().lower()
        label_str = self.label_aliases.get(label_key, str(raw_label))
        
        # Strict label validation (fail fast on unknown labels)
        if label_str not in self.class_to_idx:
            raise ValueError(
                f"Unknown label '{label_str}' at index {idx}. "
                f"Valid labels: {list(self.class_to_idx.keys())}"
            )
        label = self.class_to_idx[label_str]
        
        # Parse image_vector: comma-separated numeric string → numpy array
        image_vector = row['image_vector']
        
        if isinstance(image_vector, str):
            # Fast parsing for comma-separated values
            img_array = np.fromstring(image_vector, sep=',', dtype=np.float32)
            if img_array.size != 786432:
                raise ValueError(
                    f"Expected 786432 values (512×512×3), got {img_array.size} at index {idx}"
                )
        elif isinstance(image_vector, (list, np.ndarray)):
            # Already parsed (e.g., from parquet)
            img_array = np.array(image_vector, dtype=np.float32)
        else:
            raise TypeError(
                f"image_vector must be string or array-like, got {type(image_vector)}"
            )
        
        # Reshape to (512, 512, 3)
        img_array = img_array.reshape(512, 512, 3)
        
        # Scale to 0-255 if needed (parquet stores normalized [0,1] values)
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
        
        # Apply albumentations transform (expects numpy uint8 array, returns tensor if ToTensorV2 is used)
        if self.transform:
            image = self.transform(image=img_array)['image']
            # Convert to float and apply ImageNet normalization
            image = image.float() if image.dtype != torch.float32 else image
            
            # FIX-4: CLASS-SPECIFIC AUGMENTATION ONLY DURING TRAINING
            # Extra augmentation for Immature Cataract (label=1) - reduces No ↔ Immature confusion
            if self.is_train and IMMATURE_EXTRA_AUG and label == 1:
                # Apply ColorJitter (convert tensor back to PIL temporarily)
                pil_img = transforms.ToPILImage()(image)
                pil_img = transforms.ColorJitter(
                    brightness=IMMATURE_COLOR_JITTER_BRIGHTNESS,
                    contrast=IMMATURE_COLOR_JITTER_CONTRAST
                )(pil_img)
                pil_img = transforms.GaussianBlur(
                    kernel_size=IMMATURE_GAUSSIAN_BLUR_KERNEL,
                    sigma=IMMATURE_GAUSSIAN_BLUR_SIGMA
                )(pil_img)
                image = transforms.ToTensor()(pil_img)
            
            image = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(image)
            return image, torch.tensor(label, dtype=torch.long)
        else:
            # Fallback if no transform
            image = Image.fromarray(img_array).convert('RGB')
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(image)
            return image, torch.tensor(label, dtype=torch.long)

# Transforms: Train with heavy augmentation, Val with basic preprocessing (using albumentations)
# All augmentation parameters are configurable in config.py
# Normalization is applied in dataset __getitem__ after albumentations
train_transform = A.Compose([
    A.RandomBrightnessContrast(
        brightness_limit=AUGMENTATION_BRIGHTNESS,
        contrast_limit=AUGMENTATION_CONTRAST,
        p=AUGMENTATION_BRIGHTNESS_P
    ),
    A.GaussianBlur(blur_limit=AUGMENTATION_BLUR_SIGMA, p=AUGMENTATION_BLUR_P),
    A.MotionBlur(blur_limit=AUGMENTATION_MOTION_BLUR, p=AUGMENTATION_MOTION_BLUR_P),
    A.ImageCompression(
        quality_lower=AUGMENTATION_JPEG_QUALITY_LOWER,
        quality_upper=100,
        p=AUGMENTATION_JPEG_P
    ),
    A.RandomShadow(p=AUGMENTATION_SHADOW_P),
    A.Rotate(limit=AUGMENTATION_ROTATION, p=AUGMENTATION_ROTATION_P),
    A.CLAHE(clip_limit=AUGMENTATION_CLAHE_CLIP, p=AUGMENTATION_CLAHE_P),
    A.Resize(TRAIN_RESOLUTION, TRAIN_RESOLUTION),
    ToTensorV2()
], bbox_params=None)

val_transform = A.Compose([
    A.Resize(VAL_RESOLUTION, VAL_RESOLUTION),
    ToTensorV2()
], bbox_params=None)

# Load datasets from parquet file (from config)
parquet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), PARQUET_DATASET))

# FIX-1: STRATIFIED TRAIN/VAL SPLIT - Ensure IOL class represented proportionally
print("Loading and splitting dataset...")
df_full = pd.read_parquet(parquet_path)

# Use stratified split to ensure minority classes (IOL) appear in validation
train_df, val_df = train_test_split(
    df_full,
    test_size=0.2,
    random_state=42,
    stratify=df_full['label']  # Stratify on label column
)

# Reset indices for clean iteration
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
print(f"✓ Stratified split applied (random_state=42, stratify on label)")

# Create datasets with split dataframes
train_dataset = CataractDataset(parquet_path, transform=train_transform, is_train=True)
val_dataset = CataractDataset(parquet_path, transform=val_transform, is_train=False)
train_dataset.df = train_df
val_dataset.df = val_df

print(f"✓ Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

# DRY_RUN mode: Fast validation run
if DRY_RUN:
    print(f"\n{'='*60}")
    print(f"⚠️  DRY_RUN MODE ACTIVE")
    print(f"   Limiting to {DRY_RUN_SAMPLES} samples per split")
    print(f"   Limiting to {DRY_RUN_EPOCHS} epoch(s)")
    print(f"{'='*60}\n")
    train_indices = list(range(min(DRY_RUN_SAMPLES, len(train_dataset))))
    val_indices = list(range(min(DRY_RUN_SAMPLES, len(val_dataset))))
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

# OPTIONAL: Subsample for debugging (disabled by default)
if DEBUG_SUBSAMPLE and not DRY_RUN:
    print(f"⚠️  DEBUG MODE: Training on {DEBUG_SAMPLE_SIZE} samples only")
    indices = list(range(min(DEBUG_SAMPLE_SIZE, len(train_dataset))))
    train_dataset = Subset(train_dataset, indices)
    val_dataset = Subset(val_dataset, indices[:min(DEBUG_SAMPLE_SIZE//4, len(val_dataset))])

# ============ DATASET VALIDATION ============
print(f"\n{'='*60}")
print("DATASET VALIDATION")
print(f"{'='*60}")
print(f"Dataset columns: {list(train_df.columns)}")
print("\nDataset Sanity Check:")
try:
    # Use the detected label column from the dataset for validation
    label_col = train_dataset.dataset.label_column if isinstance(train_dataset, Subset) else train_dataset.label_column
    sample_label = train_df[label_col].iloc[0]
    # Handle both full dataset and Subset wrapper
    class_to_idx = (
        train_dataset.dataset.class_to_idx 
        if isinstance(train_dataset, Subset) 
        else train_dataset.class_to_idx
    )
    if sample_label not in class_to_idx:
        raise ValueError(f"Invalid label in dataset: '{sample_label}'")
    print(f"✓ Label column 'label' found")
    print(f"✓ Sample label: '{sample_label}'")
except KeyError:
    raise KeyError(
        "Column 'label' not found in dataset. "
        f"Available columns: {list(train_df.columns)}"
    )
except IndexError:
    raise ValueError("Dataset is empty!")

print(f"\nFinal dataset sizes after any subsampling:")
print(f"  Train: {len(train_dataset)} samples")
print(f"  Val:   {len(val_dataset)} samples")
print(f"{'='*60}\n")

# ============ CLASS DISTRIBUTION ============
# Handle class imbalance
# Extract labels properly (dataset returns tensors, so we need .item())
if isinstance(train_dataset, Subset):
    # For Subset, we need to access the underlying dataset
    train_labels = [int(train_dataset.dataset[i][1].item()) for i in train_dataset.indices]
else:
    # For full dataset, iterate and extract tensor values
    train_labels = [int(label.item()) for _, label in train_dataset]

label_counts = Counter(train_labels)
total_samples = len(train_labels)

# Stronger class weights using numpy bincount with configurable power boost
class_counts = np.bincount(train_labels, minlength=4)
weights = 1.0 / (class_counts + 1e-6)   # inverse frequency
weights = weights ** CLASS_WEIGHT_POWER # configurable boost for rare classes
weights = weights / weights.sum() * 4   # normalize (sum = num_classes)

# FIX-2: CLAMP WEIGHTS to prevent gradient starvation
weights = np.clip(weights, CLASS_WEIGHT_MIN, CLASS_WEIGHT_MAX)
weights = weights / weights.sum() * 4   # Re-normalize after clamping

class_weights = torch.tensor(weights, dtype=torch.float32)

print(f"\n{'='*60}")
print("CLASS DISTRIBUTION:")
class_names = ['No Cataract', 'Immature Cataract', 'Mature Cataract', 'IOL Inserted']
print(f"Class counts: {class_counts}")
print(f"Class weights: {class_weights.cpu().numpy()}")
for i in range(4):
    count = label_counts.get(i, 0)
    pct = 100.0 * count / total_samples if total_samples > 0 else 0
    print(f"  {class_names[i]:20s}: {count:4d} samples ({pct:5.1f}%) | weight: {class_weights[i]:.3f}")

# Fail fast if all classes are empty
if all(label_counts.get(i, 0) == 0 for i in range(4)):
    raise ValueError("All class counts are zero! Check label mapping and dataset.")
print(f"{'='*60}\n")

# ============ DATA LOADERS ============
print(f"{'='*60}")
print("INITIALIZING DATA LOADERS")
print(f"{'='*60}")

# FIX-3: REDUCE IOL SAMPLING PRESSURE (1 IOL every 2-3 batches instead of every batch)
# Create sample weights inversely proportional to class frequency, but reduce IOL weight
if isinstance(train_dataset, Subset):
    train_labels = [int(train_dataset.dataset[i][1].item()) for i in train_dataset.indices]
else:
    train_labels = [int(label.item()) for _, label in train_dataset]

class_counts = np.bincount(train_labels, minlength=4)
weights_per_class = 1.0 / (class_counts + 1e-6)

# Reduce IOL weight (class 3) by multiplying by IOL_SAMPLING_FRACTION
weights_per_class[3] = weights_per_class[3] * IOL_SAMPLING_FRACTION

sample_weights = np.array([weights_per_class[label] for label in train_labels])
sample_weights = sample_weights / sample_weights.sum()  # Normalize

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)
print(f"✓ FIX-3: WeightedRandomSampler with reduced IOL pressure (fraction={IOL_SAMPLING_FRACTION})")
print(f"  IOL class gets ~1 sample every {int(1/IOL_SAMPLING_FRACTION)} batches instead of every batch")

# Use batch size and workers from config
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    sampler=sampler,  # Use weighted sampler instead of shuffle
    num_workers=NUM_WORKERS,
    pin_memory=False  # Explicit: no benefit for CPU training
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=False
)

# Cache loader lengths (avoid repeated len() calls in hot loops)
train_batches = len(train_loader)
val_batches = len(val_loader)

print(f"✓ Batch size: {BATCH_SIZE}")
print(f"✓ Num workers: {NUM_WORKERS}")
print(f"✓ Train batches: {train_batches}")
print(f"✓ Val batches: {val_batches}")
print(f"{'='*60}\n")

# Model
model = CataractModel()
device = torch.device(DEVICE)  # From config (default: 'cpu')
model.to(device)

# Freeze backbone for first N epochs (configurable from config.py)
if FREEZE_BACKBONE_UNTIL_EPOCH > 0:
    print(f"🔒 Freezing backbone for first {FREEZE_BACKBONE_UNTIL_EPOCH} epochs...")
    # Freeze all parameters except the classifier head (_fc)
    for name, param in model.backbone.named_parameters():
        if '_fc' not in name and 'fc' not in name:
            param.requires_grad = False
    print(f"✓ Backbone frozen (except classifier head). Only head trainable.")
else:
    print(f"✓ All parameters trainable (no backbone freezing).")

print(f"✓ Model: CataractModel (EfficientNet-B0 backbone)")
print(f"✓ Device: {device}")

# DEBUG: Check trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"✓ Trainable parameters: {trainable_params:,}")
print(f"✓ Frozen parameters: {frozen_params:,}")

# DEBUG: Print FC layer status
for name, param in model.backbone.named_parameters():
    if 'fc' in name:
        print(f"  FC param '{name}': requires_grad={param.requires_grad}")

print(f"{'='*60}\n")

# ============ OPTIMIZER & SCHEDULER ============
print(f"{'='*60}")
print("CONFIGURING OPTIMIZER & SCHEDULER")
print(f"{'='*60}")

# FIX-2: USE COMPUTED INVERSE-FREQUENCY WEIGHTS (not hard-coded config weights)
# Computed weights are already calculated above; convert to device
computed_class_weights = class_weights.to(device)
print(f"✓ Class weights (inverse frequency): {computed_class_weights.cpu().numpy()}")
print(f"  [No Cataract, Immature, Mature, IOL] with CLASS_WEIGHT_POWER={CLASS_WEIGHT_POWER}")

# Optimizer with weight decay for regularization
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# FIX-1: Replace Focal Loss with CrossEntropyLoss
# Focal Loss was causing gradient starvation on majority classes (No Cataract, Mature)
# CrossEntropyLoss with clamped class weights is sufficient for imbalance handling
criterion = nn.CrossEntropyLoss(weight=computed_class_weights, label_smoothing=LABEL_SMOOTHING)

# Learning rate scheduler (reduce LR on plateau with configurable parameters)
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=SCHEDULER_FACTOR,
    patience=SCHEDULER_PATIENCE,
    verbose=True
)
print(f"✓ Optimizer: AdamW (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
print(f"✓ Loss: CrossEntropyLoss (label_smoothing={LABEL_SMOOTHING}, class-weighted, no focal)")
print(f"✓ Scheduler: ReduceLROnPlateau (factor={SCHEDULER_FACTOR}, patience={SCHEDULER_PATIENCE})")
print(f"{'='*60}\n")

# Training loop
num_epochs = DRY_RUN_EPOCHS if DRY_RUN else NUM_EPOCHS
print(f"\n{'='*60}")
print(f"STARTING TRAINING: {num_epochs} EPOCH(S) ON {device}")
print(f"EARLY STOPPING: patience={EARLY_STOPPING_PATIENCE} epochs (informational only - will run all epochs)")
print(f"BEST MODEL SAVING: Enabled - saves to best_model/model.pth")
print(f"{'='*60}\n")

# Create best_model directory and define path (MANDATORY - persists best model to disk)
BEST_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'best_model')
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, 'model.pth')
print(f"✓ Best model will be saved to: {BEST_MODEL_PATH}\n")

# Best model tracking
best_val_f1 = 0.0
patience_counter = 0

for epoch in range(num_epochs):
    epoch_start = time.time()
    
    # DEBUG: On first epoch, check if model has trainable parameters
    if epoch == 0:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("ERROR: No trainable parameters in model! Check parameter freezing.")
        print(f"\nDEBUG: Found {len(trainable_params)} trainable parameter tensors")
    
    # Unfreeze backbone at FREEZE_BACKBONE_UNTIL_EPOCH if enabled
    if epoch == FREEZE_BACKBONE_UNTIL_EPOCH and FREEZE_BACKBONE_UNTIL_EPOCH > 0:
        print(f"\n🔓 Unfreezing backbone at epoch {epoch}. Switching to lower LR: {UNFREEZE_LR}")
        for param in model.backbone.parameters():
            param.requires_grad = True
        # Reduce learning rate when unfreezing
        for param_group in optimizer.param_groups:
            param_group['lr'] = UNFREEZE_LR
        print(f"✓ Backbone unfrozen. All parameters now trainable.\n")
    
    model.train()
    train_loss = 0.0
    train_preds = []
    train_labels_list = []
    accumulated_loss = 0.0  # For gradient accumulation

    # ============ TRAINING PHASE ============
    print(f"\nEpoch {epoch+1}/{num_epochs} - Training...")
    train_phase_start = time.time()
    batch_times = []
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        batch_start = time.time()
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Safety check
        if outputs is None:
            raise RuntimeError(f"Model forward pass returned None. Check model.py forward() method.")
        
        # DEBUG: Check output properties
        if batch_idx == 0 and epoch == 0:
            print(f"\nDEBUG - Outputs properties:")
            print(f"  outputs shape: {outputs.shape}")
            print(f"  outputs.requires_grad: {outputs.requires_grad}")
            print(f"  outputs.grad_fn: {outputs.grad_fn}")
            print(f"  outputs dtype: {outputs.dtype}")
            print(f"  labels: {labels}")
            print(f"  labels dtype: {labels.dtype}")
            print(f"  Model training: {model.training}\n")
        
        # FIX-5: DISABLE MixUp entirely (MIXUP_PROB = 0.0)
        # MixUp was contributing to gradient starvation. Re-enable at macro-F1 > 0.75
        if MIXUP_PROB > 0 and np.random.rand() < MIXUP_PROB and not should_skip_mixup(labels):
            images_mixed, labels_a, labels_b, lam = mixup(images, labels, alpha=0.2)
            outputs_mixed = model(images_mixed)
            loss = mixup_criterion(criterion, outputs_mixed, labels_a, labels_b, lam)
        else:
            loss = criterion(outputs, labels)
        
        # Scale loss by gradient accumulation steps
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        
        # DEBUG: Check loss properties
        if batch_idx == 0 and epoch == 0:
            print(f"\nDEBUG - Loss tensor properties BEFORE backward:")
            print(f"  loss value: {loss.item() if loss.requires_grad else 'scalar'}")
            print(f"  loss.requires_grad: {loss.requires_grad}")
            print(f"  loss.grad_fn: {loss.grad_fn}")
            print(f"  loss dtype: {loss.dtype}")
            if loss.grad_fn:
                print(f"  loss.grad_fn.next_functions: {loss.grad_fn.next_functions}\n")
            else:
                print(f"  WARNING: loss has no grad_fn!\n")
        
        loss.backward()
        
        # FIX-9: Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        accumulated_loss += loss.item()
        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_labels_list.extend(labels.cpu().numpy())
        
        # Perform optimizer step every GRADIENT_ACCUMULATION_STEPS batches
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            accumulated_loss = 0.0
        
        # Progress update with simple ASCII progress bar
        if (batch_idx + 1) % PROGRESS_LOG_INTERVAL == 0 or (batch_idx + 1) == train_batches:
            batch_time = time.time() - batch_start
            progress_pct = 100.0 * (batch_idx + 1) / train_batches
            bar_width = 20
            filled = int(bar_width * (batch_idx + 1) / train_batches)
            bar = '█' * filled + '░' * (bar_width - filled)
            print(f"  [{bar}] {progress_pct:5.1f}% | Batch {batch_idx+1}/{train_batches} | Loss: {loss.item():.4f} | {batch_time:.2f}s")

    # Flush any remaining accumulated gradients
    if GRADIENT_ACCUMULATION_STEPS > 1:
        optimizer.step()
        optimizer.zero_grad()

    train_f1 = f1_score(train_labels_list, train_preds, average='macro')
    train_loss /= train_batches
    train_phase_time = time.time() - train_phase_start
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
    effective_batch = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    
    print(f"  ✓ Training complete: {train_phase_time:.1f}s (avg {avg_batch_time:.2f}s/batch)")
    print(f"  ℹ️  Effective batch size: {effective_batch} (physical: {BATCH_SIZE} x accumulation: {GRADIENT_ACCUMULATION_STEPS})")

    # ============ VALIDATION PHASE ============
    print(f"\n{'─'*60}")
    print(f"EPOCH {epoch+1}/{num_epochs} - VALIDATION")
    print(f"{'─'*60}")
    val_phase_start = time.time()
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels_list = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels_list.extend(labels.cpu().numpy())
            
            # Simple validation progress (less verbose than training)
            if (batch_idx + 1) == val_batches or (val_batches > 20 and (batch_idx + 1) % 10 == 0):
                progress_pct = 100.0 * (batch_idx + 1) / val_batches
                print(f"  Validated {batch_idx+1}/{val_batches} batches ({progress_pct:.0f}%)")

    val_f1 = f1_score(val_labels_list, val_preds, average='macro')
    val_loss /= val_batches
    val_phase_time = time.time() - val_phase_start
    
    print(f"  ✓ Validation complete: {val_phase_time:.1f}s")
    
    # Best model saving to disk (MANDATORY - survives crashes)
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        patience_counter = 0
        print(f"  🏆 New BEST model saved (Val F1: {best_val_f1:.4f}) → {BEST_MODEL_PATH}")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"  ℹ️  Model saturated around epoch {epoch+1} (patience={EARLY_STOPPING_PATIENCE})")
    
    # Per-class F1 scores (diagnostic)
    train_f1_per_class = f1_score(train_labels_list, train_preds, average=None, zero_division=0)
    val_f1_per_class = f1_score(val_labels_list, val_preds, average=None, zero_division=0)
    
    # Step scheduler based on validation loss
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    epoch_time = time.time() - epoch_start
    
    # ============ EPOCH SUMMARY ============
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch+1}/{num_epochs} SUMMARY")
    print(f"{'='*60}")
    print(f"  Metrics:")
    print(f"    Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
    print(f"    Val Loss:   {val_loss:.4f} | Val F1:   {val_f1:.4f}")
    print(f"  Timing:")
    print(f"    Total:      {epoch_time:.1f}s ({epoch_time/60:.1f}m)")
    print(f"    Training:   {train_phase_time:.1f}s ({100*train_phase_time/epoch_time:.0f}%)")
    print(f"    Validation: {val_phase_time:.1f}s ({100*val_phase_time/epoch_time:.0f}%)")
    print(f"    Avg batch:  {avg_batch_time:.2f}s")
    print(f"  Learning Rate: {current_lr:.6f}")
    
    # Warn if epoch is slow
    if epoch_time > SLOW_EPOCH_THRESHOLD:
        print(f"\n  ⚠️  Warning: Epoch took {epoch_time:.0f}s (>{SLOW_EPOCH_THRESHOLD}s threshold)")
    
    print(f"\n  Per-Class F1 (Train):")
    for i, class_name in enumerate(class_names):
        print(f"    {class_name:20s}: {train_f1_per_class[i]:.4f}")
    print(f"\n  Per-Class F1 (Val):")
    for i, class_name in enumerate(class_names):
        print(f"    {class_name:20s}: {val_f1_per_class[i]:.4f}")
    print(f"{'='*60}\n")

# Save model weights
print("\n" + "="*60)
print("Saving model...")
print("="*60)

# CRITICAL: Load best model before packaging (not last epoch)
print("Loading BEST model for submission...")
best_state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
model.load_state_dict(best_state_dict)
print(f"✓ Loaded best model (Val F1: {best_val_f1:.4f}) from {BEST_MODEL_PATH}\n")

ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
date_str = datetime.datetime.now(ist).strftime('%Y-%m-%d')
time_str = datetime.datetime.now(ist).strftime('%H-%M-%S')
folder_name = f'model_{date_str}_{time_str}_BEST_F1_{best_val_f1:.4f}'
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), SUBMISSION_DIR))
folder_path = os.path.join(root_dir, folder_name)
os.makedirs(folder_path, exist_ok=True)
model_path = os.path.join(folder_path, 'model.pth')
torch.save(model.state_dict(), model_path)
print(f'✓ Best model saved as {model_path}')
# Copy fixed files to folder for tar.gz
import shutil
import tarfile
from pathlib import Path

shutil.copy(os.path.join(os.path.dirname(__file__), '..', 'requirements.txt'), folder_path)
shutil.copy(os.path.join(os.path.dirname(__file__), '..', 'README.md'), folder_path)
shutil.copytree(os.path.join(os.path.dirname(__file__), '..', 'code'), os.path.join(folder_path, 'code'), dirs_exist_ok=True)

# Create tar.gz in the folder (Windows-safe, cross-platform)
print("Creating submission archive...")
tar_path = Path(folder_path) / "model.tar.gz"

with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(Path(folder_path) / "model.pth", arcname="model.pth")
    tar.add(Path(folder_path) / "requirements.txt", arcname="requirements.txt")
    tar.add(Path(folder_path) / "README.md", arcname="README.md")
    tar.add(Path(folder_path) / "code", arcname="code")

print(f'✓ Tar.gz created as {tar_path}')
# Also save in root for submission
root_model_path = os.path.join(root_dir, 'model.pth')
torch.save(model.state_dict(), root_model_path)
print(f'✓ Model also saved as {root_model_path}')
print("="*60)
print("Training complete!")
print("="*60)
print(f"\n{'='*60}")
print("BEST MODEL SUMMARY")
print(f"{'='*60}")
print(f"✓ Best validation F1: {best_val_f1:.4f}")
print(f"✓ Best model saved to: {BEST_MODEL_PATH}")
print(f"✓ Submission package: {tar_path}")
print(f"  📌 Submission contains BEST model (F1: {best_val_f1:.4f}), not last epoch")
print(f"{'='*60}\n")

# Log the run
# import sys
# import os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# from utils.log_run import log_run
# log_run(f"Training completed: Train F1 {train_f1:.4f}, Val F1 {val_f1:.4f}")
print(f"Logged run outcome at {__import__('datetime').datetime.now()}")

# Save training summary to run folder
summary_path = os.path.join(run_folder, 'summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(f"Training Run Summary\n")
    f.write(f"{'='*60}\n")
    f.write(f"Run Timestamp: {run_timestamp}\n")
    f.write(f"Best Validation F1: {best_val_f1:.4f}\n")
    f.write(f"Total Epochs: {num_epochs}\n")
    f.write(f"Train Resolution: {TRAIN_RESOLUTION}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}\n")
    f.write(f"Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}\n")
    f.write(f"Label Smoothing: {LABEL_SMOOTHING}\n")
    f.write(f"Freeze Backbone Until Epoch: {FREEZE_BACKBONE_UNTIL_EPOCH}\n")
    f.write(f"Best Model Path: {BEST_MODEL_PATH}\n")
    f.write(f"Submission Package: {tar_path}\n")
    f.write(f"{'='*60}\n")

print(f"\n✓ Training summary saved to: {summary_path}")

# Close logger
tee_logger.close()
sys.stdout = tee_logger.terminal