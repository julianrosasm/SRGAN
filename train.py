import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from model import Generator, Discriminator
from dataset import get_dataloaders


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.DEFAULT)
        # Use features up to conv5_4
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:36]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.mse_loss = nn.MSELoss()
        
    def forward(self, sr, hr):
        sr_features = self.feature_extractor(sr)
        hr_features = self.feature_extractor(hr)
        return self.mse_loss(sr_features, hr_features)


def train_srgan(
    lr_dir='data/blurred',
    hr_dir='data/unblurred',
    num_epochs=100,
    batch_size=16,
    lr=1e-4,
    model_dir='models/'
):
    """Train SRGAN model"""
    
    # Setup device - use MPS for M1/M2 Macs
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Loss functions
    adversarial_loss = nn.BCELoss()
    content_loss = VGGPerceptualLoss().to(device)
    mse_loss = nn.MSELoss()
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.9, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.9, 0.999))
    
    # Data loaders
    train_loader, val_loader = get_dataloaders(lr_dir, hr_dir, batch_size)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Training loop
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for i, (lr_imgs, hr_imgs) in enumerate(pbar):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            batch_size_current = lr_imgs.size(0)
            
            # Labels for adversarial loss
            real_labels = torch.ones(batch_size_current, 1).to(device)
            fake_labels = torch.zeros(batch_size_current, 1).to(device)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Real images
            real_validity = discriminator(hr_imgs)
            d_loss_real = adversarial_loss(real_validity, real_labels)
            
            # Fake images
            sr_imgs = generator(lr_imgs)
            fake_validity = discriminator(sr_imgs.detach())
            d_loss_fake = adversarial_loss(fake_validity, fake_labels)
            
            # Total discriminator loss
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Generate images
            sr_imgs = generator(lr_imgs)
            
            # Adversarial loss
            gen_validity = discriminator(sr_imgs)
            g_loss_adv = adversarial_loss(gen_validity, real_labels)
            
            # Content loss (perceptual + MSE)
            g_loss_content = content_loss(sr_imgs, hr_imgs)
            g_loss_mse = mse_loss(sr_imgs, hr_imgs)
            
            # Total generator loss
            g_loss = g_loss_content + 1e-3 * g_loss_adv + g_loss_mse
            g_loss.backward()
            optimizer_G.step()
            
            # Update progress bar
            pbar.set_postfix({
                'D_loss': f'{d_loss.item():.4f}',
                'G_loss': f'{g_loss.item():.4f}',
                'G_adv': f'{g_loss_adv.item():.4f}'
            })
        
        # Validation
        if (epoch + 1) % 10 == 0:
            generator.eval()
            val_loss = 0
            with torch.no_grad():
                for lr_imgs, hr_imgs in val_loader:
                    lr_imgs = lr_imgs.to(device)
                    hr_imgs = hr_imgs.to(device)
                    sr_imgs = generator(lr_imgs)
                    val_loss += mse_loss(sr_imgs, hr_imgs).item()
            
            val_loss /= len(val_loader)
            print(f'\nValidation MSE Loss: {val_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }, os.path.join(model_dir, f'srgan_epoch_{epoch+1}.pth'))
            print(f'Checkpoint saved at epoch {epoch+1}')
    
    # Save final model
    torch.save(generator.state_dict(), os.path.join(model_dir, 'generator_final.pth'))
    print('Training complete!')


if __name__ == '__main__':
    train_srgan(
        lr_dir='data/blurred',
        hr_dir='data/unblurred',
        num_epochs=50,  # Reduced for faster training on M1
        batch_size=2,  # Reduced for MacBook Air M1 memory
        lr=1e-4
    )
