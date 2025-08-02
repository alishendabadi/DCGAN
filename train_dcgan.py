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

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
latent_dim = 100
image_size = 256  # Updated for 256x256 images
batch_size = 32  # Increased batch size for 256x256 images
num_epochs = 100
lr = 0.0002
beta1 = 0.5
beta2 = 0.999

# Create directories
os.makedirs('samples', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),  # 4x4
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64 x 32 x 32
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),  # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 32 x 64 x 64
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),  # 128x128
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # 16 x 128 x 128
            nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=False),  # 256x256, 3 channels
            nn.Tanh()
        )
    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: 3 x 256 x 256
            nn.Conv2d(3, 16, 4, 2, 1, bias=False),  # 128x128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),  # 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),  # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),  # 1x1
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def load_data():
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = torchvision.datasets.ImageFolder(
        root='./data',
        transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    return dataloader

def save_samples(generator, epoch, fixed_noise):
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise).detach().cpu()
    fake_images = (fake_images + 1) / 2.0  # Denormalize from [-1, 1] to [0, 1]
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))  # Medium figure for 256x256 images
    for i in range(16):
        row, col = i // 4, i % 4
        img = np.transpose(fake_images[i].numpy(), (1, 2, 0))
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.savefig(f'samples/epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()

def train():
    # Initialize models
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)
    
    # Apply weight initialization
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Loss function and optimizers
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    
    # Load data
    dataloader = load_data()
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)
    
    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        g_losses = []
        d_losses = []
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
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
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
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
        
        # Save samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_samples(generator, epoch + 1, fixed_noise)
        
        # Save checkpoints every 25 epochs
        if (epoch + 1) % 25 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'g_loss': g_losses[-1],
                'd_loss': d_losses[-1],
            }, f'checkpoints/checkpoint_epoch_{epoch+1}.pth')
        
        # Print epoch summary
        avg_g_loss = np.mean(g_losses)
        avg_d_loss = np.mean(d_losses)
        print(f'Epoch [{epoch+1}/{num_epochs}] - G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}')
    
    print("Training completed!")
    
    # Save final model
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
    }, 'checkpoints/final_model.pth')

if __name__ == "__main__":
    train() 