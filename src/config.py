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
NUM_EPOCHS = 20              # Reduced for faster training (was 20)
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
TRAIN_RESOLUTION = 512       # FIX-3: Train at 512 to match inference resolution
VAL_RESOLUTION = 512         # Match training resolution
INFERENCE_RESOLUTION = 512   # Inference stays 512×512 (consistent with training)

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
FREEZE_BACKBONE_UNTIL_EPOCH = 5   # Unfreeze backbone at epoch 5 (not later - head collapses by epoch 10)
UNFREEZE_LR = 5e-6           # When unfreezing, use 20x reduction from initial LR

# ============ LABEL SMOOTHING ============
LABEL_SMOOTHING = 0.05       # Mild label smoothing with CrossEntropyLoss (reduced from Focal config)

# ============ GRADIENT ACCUMULATION ============
GRADIENT_ACCUMULATION_STEPS = 4  # Accumulate gradients over this many batches (effective batch = BATCH_SIZE * GRAD_ACC_STEPS)

# ============ CLASS WEIGHTING ============
CLASS_WEIGHT_POWER = 1.0     # Exponent for class weighting (1.0=inverse frequency only, no aggressive boost)
CLASS_WEIGHT_MIN = 0.75      # CLAMP: Minimum class weight to prevent gradient starvation on No/Mature
CLASS_WEIGHT_MAX = 1.5       # CLAMP: Maximum class weight to prevent rare-class overfitting

# ============ MIXUP AUGMENTATION ============
MIXUP_PROB = 0.0             # DISABLED - Causes gradient starvation with Focal. Re-enable at macro-F1 > 0.75

# ============ IOL SAMPLING ============
IOL_SAMPLING_FRACTION = 0.33 # Include IOL 1 every ~3 batches instead of every batch (reduce pressure)

# ============ LOGGING ============
PROGRESS_LOG_INTERVAL = 5    # How often to print batch progress (in batches)
SLOW_EPOCH_THRESHOLD = 600   # Warn if epoch exceeds this many seconds

# ============ PATHS ============
PARQUET_DATASET = '../dataset/cataract-training-dataset.parquet'
SUBMISSION_DIR = '../test_submission'

# ============ DEVICE ============
DEVICE = 'cpu'  # 'cpu' or 'cuda' (force CPU for safety)

# ============ MODEL ARCHITECTURE ============
RANDOM_SEED = 42  # For ensemble diversity
BACKBONE_MODEL = 'efficientnet_b0'  # EfficientNet-B0 (original tested backbone)
NUM_CLASSES = 4  # Cataract classification: 4 classes
PRETRAINED = True  # Use pretrained ImageNet weights
