import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import gc

from model import Generator, Discriminator
from dataset import get_dataloaders


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        from torchvision.models import vgg19, VGG19_Weights
        
        try:
            vgg = vgg19(weights=VGG19_Weights.DEFAULT)
            print("✓ VGG19 weights downloaded successfully")
        except Exception as e:
            print(f"⚠ Could not download VGG19 weights: {e}")
            print("  Using randomly initialized VGG19 instead")
            vgg = vgg19(weights=None)
        
        # Use features up to conv5_4
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:36]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.mse_loss = nn.MSELoss()
        
    def forward(self, sr, hr):
        # Resize sr to match hr if sizes don't match
        if sr.size() != hr.size():
            sr = torch.nn.functional.interpolate(sr, size=hr.size()[-2:], mode='bilinear', align_corners=False)
        
        sr_features = self.feature_extractor(sr)
        hr_features = self.feature_extractor(hr)
        return self.mse_loss(sr_features, hr_features)

def train_srgan(
    lr_dir='data/blurred',
    hr_dir='data/unblurred',
    num_epochs=50,
    batch_size=16,  # REDUCED from 32 to avoid freezing
    lr=1e-4,
    checkpoint_dir='checkpoints',
    use_amp=False  # Disable for MPS stability
):
    """Train SRGAN model optimized for M2 MacBook Pro"""
    
    # Setup device for M2 (uses Metal Performance Shaders)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
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
    
    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=20, gamma=0.5)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=20, gamma=0.5)
    
        ## Data loaders
    train_loader, val_loader = get_dataloaders(
        lr_dir, hr_dir, batch_size, hr_size=128
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    # Training loop
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for i, (lr_imgs, hr_imgs) in enumerate(pbar):
            lr_imgs = lr_imgs.to(device, non_blocking=False)
            hr_imgs = hr_imgs.to(device, non_blocking=False)
            batch_size_current = lr_imgs.size(0)
            
            # Labels for adversarial loss
            real_labels = torch.ones(batch_size_current, 1, device=device)
            fake_labels = torch.zeros(batch_size_current, 1, device=device)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Real images
            real_validity = discriminator(hr_imgs)
            d_loss_real = adversarial_loss(real_validity, real_labels)
            
            # Fake images
            sr_imgs = generator(lr_imgs)
            
            # Resize sr_imgs to match hr_imgs if needed
            if sr_imgs.size() != hr_imgs.size():
                sr_imgs_resized = torch.nn.functional.interpolate(sr_imgs, size=hr_imgs.size()[-2:], mode='bilinear', align_corners=False)
            else:
                sr_imgs_resized = sr_imgs
            
            fake_validity = discriminator(sr_imgs_resized.detach())
            d_loss_fake = adversarial_loss(fake_validity, fake_labels)
            # Total discriminator loss
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_D.step()
                        # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Generate images
            sr_imgs = generator(lr_imgs)
            
            # Resize sr_imgs to match hr_imgs if needed
            if sr_imgs.size() != hr_imgs.size():
                sr_imgs = torch.nn.functional.interpolate(sr_imgs, size=hr_imgs.size()[-2:], mode='bilinear', align_corners=False)
            
            # Adversarial loss
            gen_validity = discriminator(sr_imgs)
            g_loss_adv = adversarial_loss(gen_validity, real_labels)
            
            # Content loss (perceptual + MSE)
            g_loss_content = content_loss(sr_imgs, hr_imgs)
            g_loss_mse = mse_loss(sr_imgs, hr_imgs)
            
            # Total generator loss
            g_loss = g_loss_content + 5e-3 * g_loss_adv + 0.1 * g_loss_mse
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_G.step()
            
            # Update progress bar
            pbar.set_postfix({
                'D_loss': f'{d_loss.item():.4f}',
                'G_loss': f'{g_loss.item():.4f}'
            })
            
            # Clear cache periodically
            if (i + 1) % 5 == 0:
                gc.collect()
                if device.type == 'mps':
                    torch.mps.empty_cache()
        
        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()
            # Validation
        if (epoch + 1) % 5 == 0:
            generator.eval()
            val_loss = 0
            with torch.no_grad():
                for lr_imgs, hr_imgs in val_loader:
                    lr_imgs = lr_imgs.to(device)
                    hr_imgs = hr_imgs.to(device)
                    sr_imgs = generator(lr_imgs)
                    
                    # Resize sr_imgs to match hr_imgs if needed
                    if sr_imgs.size() != hr_imgs.size():
                        sr_imgs = torch.nn.functional.interpolate(sr_imgs, size=hr_imgs.size()[-2:], mode='bilinear', align_corners=False)
                    
                    val_loss += mse_loss(sr_imgs, hr_imgs).item()
            
            val_loss /= len(val_loader)
            print(f'\nValidation MSE Loss: {val_loss:.4f}')
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
            }, os.path.join(checkpoint_dir, f'srgan_epoch_{epoch+1}.pth'))
            print(f'Checkpoint saved at epoch {epoch+1}')
    
    print('Training complete!')


if __name__ == '__main__':
    train_srgan(
        lr_dir='data/blurred',
        hr_dir='data/unblurred',
        num_epochs=70,
        batch_size=12,  # Start small
        lr=1e-4
    )