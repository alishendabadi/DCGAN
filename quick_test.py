#!/usr/bin/env python3
"""
Quick test script to verify DCGAN training works correctly
Runs for only 5 epochs to test the complete pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# Import our models and config
from train_dcgan import Generator, Discriminator, weights_init

# Quick test configuration
QUICK_TEST_EPOCHS = 5
QUICK_TEST_BATCH_SIZE = 32

def quick_train():
    """Run a quick training test"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Quick test using device: {device}")
    
    # Create directories
    os.makedirs('samples', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Apply weight initialization
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Loss function and optimizers
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Load data with smaller batch size for quick test
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = torchvision.datasets.ImageFolder(
        root='./data',
        transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=QUICK_TEST_BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(16, 100, 1, 1, device=device)
    
    print(f"Starting quick test training for {QUICK_TEST_EPOCHS} epochs...")
    
    # Training loop
    for epoch in range(QUICK_TEST_EPOCHS):
        generator.train()
        discriminator.train()
        
        g_losses = []
        d_losses = []
        
        progress_bar = tqdm(dataloader, desc=f'Quick Test Epoch {epoch+1}/{QUICK_TEST_EPOCHS}')
        
        for i, (real_images, _) in enumerate(progress_bar):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Labels
            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            # Real images
            real_outputs = discriminator(real_images)
            d_real_loss = criterion(real_outputs, real_labels)
            
            # Fake images
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_images = generator(noise)
            fake_outputs = discriminator(fake_images.detach())
            d_fake_loss = criterion(fake_outputs, fake_labels)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            
            fake_outputs = discriminator(fake_images)
            g_loss = criterion(fake_outputs, real_labels)
            
            g_loss.backward()
            g_optimizer.step()
            
            # Record losses
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'G_Loss': f'{g_loss.item():.4f}',
                'D_Loss': f'{d_loss.item():.4f}'
            })
            
            # Only process a few batches for quick test
            if i >= 10:  # Process only 10 batches per epoch for quick test
                break
        
        # Save samples every epoch for quick test
        generator.eval()
        with torch.no_grad():
            fake_images = generator(fixed_noise).detach().cpu()
        
        # Save a grid of images
        fake_images = (fake_images + 1) / 2.0  # Denormalize from [-1, 1] to [0, 1]
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i in range(16):
            row, col = i // 4, i % 4
            img = np.transpose(fake_images[i].numpy(), (1, 2, 0))
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
        plt.tight_layout()
        plt.savefig(f'samples/quick_test_epoch_{epoch+1:03d}.png')
        plt.close()
        
        # Print epoch summary
        avg_g_loss = np.mean(g_losses)
        avg_d_loss = np.mean(d_losses)
        print(f'Quick Test Epoch [{epoch+1}/{QUICK_TEST_EPOCHS}] - G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}')
    
    print("Quick test completed!")
    print("Check the 'samples/' directory for generated images.")
    print("If you see some digit-like shapes, the training is working correctly!")
    
    # Save final model from quick test
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
    }, 'checkpoints/quick_test_model.pth')
    
    print("Quick test model saved to 'checkpoints/quick_test_model.pth'")

if __name__ == "__main__":
    quick_train() 