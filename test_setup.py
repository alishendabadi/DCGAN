#!/usr/bin/env python3
"""
Test script to verify DCGAN setup and basic functionality
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# Import our models
from train_dcgan import Generator, Discriminator, weights_init

def test_imports():
    """Test that all required packages are imported successfully"""
    print("✓ All imports successful")
    return True

def test_device():
    """Test device availability"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    return device

def test_model_creation(device):
    """Test that Generator and Discriminator can be created"""
    try:
        # Create models
        generator = Generator().to(device)
        discriminator = Discriminator().to(device)
        
        # Apply weight initialization
        generator.apply(weights_init)
        discriminator.apply(weights_init)
        
        print("✓ Generator and Discriminator created successfully")
        return generator, discriminator
    except Exception as e:
        print(f"✗ Error creating models: {e}")
        return None, None

def test_forward_pass(generator, discriminator, device):
    """Test forward pass through both models"""
    try:
        # Test generator
        batch_size = 4
        latent_dim = 100
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = generator(noise)
        
        # Test discriminator on fake images
        fake_outputs = discriminator(fake_images)
        
        # Test discriminator on random noise (should work but give low confidence)
        random_images = torch.randn(batch_size, 3, 512, 512, device=device)
        random_outputs = discriminator(random_images)
        
        print(f"✓ Forward pass successful")
        print(f"  - Generator output shape: {fake_images.shape}")
        print(f"  - Discriminator output shape: {fake_outputs.shape}")
        print(f"  - Fake image outputs range: [{fake_outputs.min():.3f}, {fake_outputs.max():.3f}]")
        
        return True
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        return False

def test_sample_generation(generator, device):
    """Test sample generation and visualization"""
    try:
        # Generate samples
        generator.eval()
        with torch.no_grad():
            noise = torch.randn(16, 100, 1, 1, device=device)
            samples = generator(noise).detach().cpu()
        # Denormalize
        samples = (samples + 1) / 2.0
        # Create visualization
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))  # Larger figure for 512x512 images
        for i in range(16):
            row, col = i // 4, i % 4
            img = np.transpose(samples[i].numpy(), (1, 2, 0))
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
        plt.tight_layout()
        plt.savefig('test_samples.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Sample generation and visualization successful")
        print("  - Test samples saved to 'test_samples.png'")
        return True
    except Exception as e:
        print(f"✗ Error in sample generation: {e}")
        return False

def main():
    """Run all tests"""
    print("Running DCGAN setup tests...\n")
    test_imports()
    device = test_device()
    generator, discriminator = test_model_creation(device)
    if generator is None or discriminator is None:
        print("\n✗ Setup failed - cannot create models")
        return False
    if not test_forward_pass(generator, discriminator, device):
        print("\n✗ Setup failed - forward pass error")
        return False
    if not test_sample_generation(generator, device):
        print("\n✗ Setup failed - sample generation error")
        return False
    print("\n🎉 All tests passed! DCGAN is ready to use with CASIA-WebFace.")
    print("\nNext steps:")
    print("1. Run: python train_dcgan.py")
    print("2. Wait for training to complete")
    print("3. Generate samples with: python generate_samples.py --model_path checkpoints/final_model.pth")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 