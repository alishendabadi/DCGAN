# DCGAN for MNIST Dataset

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate handwritten digits from the MNIST dataset using PyTorch.

## Overview

DCGAN is a type of generative adversarial network that uses convolutional layers in both the generator and discriminator. This implementation is specifically designed for the MNIST dataset (28x28 grayscale images of handwritten digits).

## Features

- **Generator**: Uses transposed convolutions to upsample from 100-dimensional noise to 28x28 images
- **Discriminator**: Uses convolutional layers to classify images as real or fake
- **Training**: Implements the standard GAN training procedure with Adam optimizer
- **Visualization**: Automatically saves generated samples during training
- **Checkpointing**: Saves model checkpoints for resuming training
- **Sample Generation**: Utility script for generating samples from trained models
- **Interpolation**: Generate smooth transitions between different digits

## Architecture

### Generator
- Input: 100-dimensional random noise
- Output: 28x28 grayscale image
- Architecture: 4 transposed convolutional layers with batch normalization and ReLU activation

### Discriminator
- Input: 28x28 grayscale image
- Output: Single value (probability of being real)
- Architecture: 4 convolutional layers with batch normalization and LeakyReLU activation

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the DCGAN on MNIST:

```bash
python train_dcgan.py
```

The training script will:
- Download the MNIST dataset automatically
- Create `samples/` and `checkpoints/` directories
- Save generated samples every 10 epochs
- Save model checkpoints every 25 epochs
- Display training progress with loss values

### Generating Samples

To generate samples from a trained model:

```bash
python generate_samples.py --model_path checkpoints/final_model.pth --num_samples 16
```

### Generating Interpolation

To generate interpolation between two random points in latent space:

```bash
python generate_samples.py --model_path checkpoints/final_model.pth --interpolation --num_steps 10
```

## Hyperparameters

The default hyperparameters are:
- **Latent dimension**: 100
- **Batch size**: 64
- **Learning rate**: 0.0002
- **Beta1**: 0.5 (for Adam optimizer)
- **Beta2**: 0.999 (for Adam optimizer)
- **Number of epochs**: 100

## File Structure

```
dcgan_mnist/
├── train_dcgan.py          # Main training script
├── generate_samples.py     # Sample generation utility
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── samples/               # Generated samples during training
├── checkpoints/           # Model checkpoints
└── data/                  # MNIST dataset (downloaded automatically)
```

## Training Progress

During training, you'll see:
- Progress bars showing current epoch and batch progress
- Generator and Discriminator loss values
- Generated samples saved every 10 epochs
- Model checkpoints saved every 25 epochs

## Expected Results

After training for 100 epochs, you should see:
- Generated images that look like handwritten digits
- Stable training with balanced generator and discriminator losses
- Smooth interpolation between different digit styles

## Tips for Better Results

1. **Training Time**: Training typically takes 1-2 hours on a GPU, longer on CPU
2. **Loss Balance**: Monitor that neither generator nor discriminator loss becomes too low
3. **Sample Quality**: Generated samples improve significantly after 50+ epochs
4. **Hardware**: GPU acceleration is recommended for faster training

## Troubleshooting

- **CUDA out of memory**: Reduce batch size
- **Poor quality samples**: Increase number of training epochs
- **Training instability**: Try adjusting learning rate or beta parameters
- **Import errors**: Ensure all dependencies are installed correctly

## License

This project is open source and available under the MIT License.

## References

- [DCGAN Paper](https://arxiv.org/abs/1511.06434)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) 

## Downloading CASIA-WebFace Dataset

This project now uses the CASIA-WebFace dataset from Kaggle. To download it:

1. **Install Kaggle CLI** (already included in `install.sh`):
   ```bash
   pip install kaggle
   ```
2. **Get your Kaggle API key:**
   - Go to https://www.kaggle.com > Account > Create New API Token
   - This will download a file called `kaggle.json`.
   - Place it in `~/.kaggle/`:
     ```bash
     mkdir -p ~/.kaggle
     mv /path/to/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```
3. **Download the dataset:**
   ```bash
   kaggle datasets download -d hwalsuklee/casia-webface --unzip -p ./data
   ```
   This will extract the images to `./data`. 