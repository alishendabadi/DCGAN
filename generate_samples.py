import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Import the Generator class from the training script
from train_dcgan import Generator

def generate_samples(model_path, num_samples=16, save_path='generated_samples.png'):
    """
    Generate samples from a trained DCGAN model
    
    Args:
        model_path (str): Path to the saved model checkpoint
        num_samples (int): Number of samples to generate
        save_path (str): Path to save the generated samples image
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the generator
    generator = Generator().to(device)
    
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
    else:
        # If it's just the generator state dict
        generator.load_state_dict(checkpoint)
    
    generator.eval()
    
    # Generate random noise
    latent_dim = 100
    noise = torch.randn(num_samples, latent_dim, 1, 1, device=device)
    
    # Generate images
    with torch.no_grad():
        fake_images = generator(noise).detach().cpu()
    
    # Denormalize images from [-1, 1] to [0, 1]
    fake_images = (fake_images + 1) / 2.0
    
    # Create a grid of images
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    
    for i in range(num_samples):
        row = i // grid_size
        col = i % grid_size
        img = np.transpose(fake_images[i].numpy(), (1, 2, 0))  # Convert to HWC format
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(num_samples, grid_size * grid_size):
        row = i // grid_size
        col = i % grid_size
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Generated samples saved to {save_path}")

def generate_interpolation(model_path, num_steps=10, save_path='interpolation.png'):
    """
    Generate interpolation between two random points in latent space
    
    Args:
        model_path (str): Path to the saved model checkpoint
        num_steps (int): Number of interpolation steps
        save_path (str): Path to save the interpolation image
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the generator
    generator = Generator().to(device)
    
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
    else:
        generator.load_state_dict(checkpoint)
    
    generator.eval()
    
    # Generate two random points in latent space
    latent_dim = 100
    z1 = torch.randn(1, latent_dim, 1, 1, device=device)
    z2 = torch.randn(1, latent_dim, 1, 1, device=device)
    
    # Create interpolation
    alphas = torch.linspace(0, 1, num_steps, device=device)
    interpolated_images = []
    
    with torch.no_grad():
        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            fake_image = generator(z_interp).detach().cpu()
            interpolated_images.append(fake_image)
    
    # Stack images
    interpolated_images = torch.cat(interpolated_images, dim=0)
    interpolated_images = (interpolated_images + 1) / 2.0  # Denormalize
    
    # Create figure
    fig, axes = plt.subplots(1, num_steps, figsize=(3*num_steps, 3))
    
    for i in range(num_steps):
        img = np.transpose(interpolated_images[i].numpy(), (1, 2, 0))  # Convert to HWC format
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'α={i/(num_steps-1):.1f}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Interpolation saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate samples from trained DCGAN')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--num_samples', type=int, default=16,
                       help='Number of samples to generate')
    parser.add_argument('--save_path', type=str, default='generated_samples.png',
                       help='Path to save the generated samples')
    parser.add_argument('--interpolation', action='store_true',
                       help='Generate interpolation instead of random samples')
    parser.add_argument('--num_steps', type=int, default=10,
                       help='Number of interpolation steps')
    
    args = parser.parse_args()
    
    if args.interpolation:
        generate_interpolation(args.model_path, args.num_steps, args.save_path)
    else:
        generate_samples(args.model_path, args.num_samples, args.save_path) 