"""
Configuration file for DCGAN training
"""

import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model hyperparameters
LATENT_DIM = 100
IMAGE_SIZE = 256  # Updated for 256x256 images
IMAGE_CHANNELS = 3  # RGB images

# Training hyperparameters
BATCH_SIZE = 32  # Increased for 256x256 images
NUM_EPOCHS = 100
LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999

# Data hyperparameters
NUM_WORKERS = 2
DOWNLOAD_MNIST = True

# Save intervals
SAVE_SAMPLES_EVERY = 10  # epochs
SAVE_CHECKPOINT_EVERY = 25  # epochs

# Directories
SAMPLE_DIR = 'samples'
CHECKPOINT_DIR = 'checkpoints'
DATA_DIR = './data'

# Visualization
NUM_SAMPLES_TO_GENERATE = 16
SAMPLE_GRID_SIZE = 4  # 4x4 grid

# Model architecture
GENERATOR_FEATURES = [512, 256, 128, 64, 32, 16, 3]  # Number of features in each layer
DISCRIMINATOR_FEATURES = [16, 32, 64, 128, 256, 512, 1]  # Number of features in each layer

# Loss function
LOSS_FUNCTION = 'BCE'  # Binary Cross Entropy

# Optimizer
OPTIMIZER = 'Adam'

# Training stability
LABEL_SMOOTHING = False
LABEL_SMOOTHING_VALUE = 0.1

# Gradient clipping (optional)
USE_GRADIENT_CLIPPING = False
GRADIENT_CLIP_VALUE = 1.0

# Noise for visualization
FIXED_NOISE_SEED = 42 