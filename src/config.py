"""
Configuration file for training pipeline.
Modify these values to change training behavior without editing train.py
"""

# ============ DATASET & DRY-RUN ============
DRY_RUN = False              # Set to True for quick testing, False for real training
DRY_RUN_SAMPLES = 20         # Samples per split when DRY_RUN=True
DRY_RUN_EPOCHS = 1           # Epochs to run when DRY_RUN=True
DEBUG_SUBSAMPLE = False      # Alternative debug mode
DEBUG_SAMPLE_SIZE = 50       # Only used if DEBUG_SUBSAMPLE=True

# ============ TRAINING HYPERPARAMETERS ============
NUM_EPOCHS = 20              # Total epochs (will stop early if val_f1 plateaus)
BATCH_SIZE = 2               # Batch size (keep small for CPU)
NUM_WORKERS = 0              # DataLoader workers (0=safe for Windows, 1-2=faster on Linux)
LEARNING_RATE = 1e-4         # Initial learning rate for AdamW
WEIGHT_DECAY = 1e-4          # L2 regularization (prevents overfitting)

# ============ CLASS IMBALANCE HANDLING ============
# Class weights for CrossEntropyLoss [No, Immature, Mature, IOL]
# Based on observed confusion: Immature gets 2x weight
CLASS_WEIGHTS = [1.0, 2.0, 1.2, 1.0]  # Boost Immature Cataract recall

# ============ EARLY STOPPING ============
EARLY_STOPPING_PATIENCE = 5  # Stop if val_f1 doesn't improve for N epochs (informational only)
SAVE_BEST_MODEL = True       # Save best model checkpoint during training

# ============ SCHEDULER ============
SCHEDULER_FACTOR = 0.5       # Multiply LR by this when plateau detected
SCHEDULER_PATIENCE = 3       # Epochs without improvement before reducing LR

# ============ RESOLUTION ============
TRAIN_RESOLUTION = 384       # Training image resolution (faster than 512)
VAL_RESOLUTION = 384         # Validation image resolution
INFERENCE_RESOLUTION = 512   # Inference stays 512×512 (no contract violation)

# ============ AUGMENTATION PARAMETERS ============
AUGMENTATION_BRIGHTNESS = 0.4      # RandomBrightnessContrast range
AUGMENTATION_CONTRAST = 0.4        # RandomBrightnessContrast range
AUGMENTATION_BRIGHTNESS_P = 0.8    # Probability of brightness/contrast aug
AUGMENTATION_BLUR_SIGMA = (3, 9)   # GaussianBlur range
AUGMENTATION_BLUR_P = 0.6          # Probability of Gaussian blur
AUGMENTATION_MOTION_BLUR = 9       # MotionBlur kernel size
AUGMENTATION_MOTION_BLUR_P = 0.5   # Probability of motion blur
AUGMENTATION_JPEG_QUALITY_LOWER = 30  # JPEG compression lower bound
AUGMENTATION_JPEG_P = 0.6          # Probability of JPEG compression
AUGMENTATION_SHADOW_P = 0.4        # Probability of random shadow
AUGMENTATION_ROTATION = 20         # Rotation range in degrees
AUGMENTATION_ROTATION_P = 0.7      # Probability of rotation
AUGMENTATION_CLAHE_P = 1.0         # Probability of CLAHE (always apply)
AUGMENTATION_CLAHE_CLIP = 4.0      # CLAHE clip limit

# ============ CLASS-SPECIFIC AUGMENTATION ============
# Extra augmentation for Immature Cataract (reduces No ↔ Immature confusion)
IMMATURE_EXTRA_AUG = True          # Enable extra augmentation for Immature class
IMMATURE_COLOR_JITTER_BRIGHTNESS = 0.2  # Color jitter brightness for Immature
IMMATURE_COLOR_JITTER_CONTRAST = 0.2    # Color jitter contrast for Immature
IMMATURE_GAUSSIAN_BLUR_KERNEL = 3       # Gaussian blur kernel for Immature
IMMATURE_GAUSSIAN_BLUR_SIGMA = (0.1, 1.0)  # Gaussian blur sigma for Immature

# ============ BACKBONE FREEZING ============
FREEZE_BACKBONE_UNTIL_EPOCH = 3  # Freeze pretrained weights until this epoch (0-indexed)

# ============ CLASS WEIGHTING ============
CLASS_WEIGHT_POWER = 1.5     # Exponent for class weighting (1.5=aggressive boost for minorities)

# ============ LOGGING ============
PROGRESS_LOG_INTERVAL = 5    # How often to print batch progress (in batches)
SLOW_EPOCH_THRESHOLD = 600   # Warn if epoch exceeds this many seconds

# ============ PATHS ============
PARQUET_DATASET = '../dataset/cataract-training-dataset.parquet'
SUBMISSION_DIR = '../test_submission'

# ============ DEVICE ============
DEVICE = 'cpu'  # 'cpu' or 'cuda' (force CPU for safety)

# ============ MODEL ARCHITECTURE ============
BACKBONE_MODEL = 'efficientnet-b0'  # EfficientNet variant
NUM_CLASSES = 4  # Cataract classification: 4 classes
PRETRAINED = True  # Use pretrained ImageNet weights
