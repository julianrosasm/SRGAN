import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from model import Generator


class BlurredImageDataset(Dataset):
    """Load blurred images for evaluation"""
    def __init__(self, lr_dir, transform=None, max_images=15):
        self.lr_dir = Path(lr_dir)
        all_files = sorted([f for f in os.listdir(self.lr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.image_files = all_files[:max_images]  # Limit to max_images
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.lr_dir / self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.image_files[idx]

def evaluate_srgan(
    checkpoint_path='checkpoints/srgan_epoch_70.pth',
    lr_dir='data/blurred',
    output_dir='evaluation_results',
    batch_size=4,
    max_images=15
):
    """Compare SRGAN output with blurred images"""
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize generator
    generator = Generator().to(device)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return
    
    generator.eval()
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load dataset
    dataset = BlurredImageDataset(lr_dir, transform=transform, max_images=max_images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Evaluating {len(dataset)} images...")
    
    # Process images
    with torch.no_grad():
        for batch_idx, (blurred_imgs, filenames) in enumerate(dataloader):
            blurred_imgs = blurred_imgs.to(device)
            
            # Generate sharpened images
            sharpened_imgs = generator(blurred_imgs)
            
            # Denormalize for visualization
            blurred_display = blurred_imgs.cpu() * 0.5 + 0.5
            sharpened_display = sharpened_imgs.cpu() * 0.5 + 0.5
            
            # Save comparison images
            for i, filename in enumerate(filenames):
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Blurred image
                blurred_np = blurred_display[i].permute(1, 2, 0).numpy()
                axes[0].imshow(np.clip(blurred_np, 0, 1))
                axes[0].set_title('Blurred Input')
                axes[0].axis('off')
                
                # Sharpened image
                sharpened_np = sharpened_display[i].permute(1, 2, 0).numpy()
                axes[1].imshow(np.clip(sharpened_np, 0, 1))
                axes[1].set_title('SRGAN Sharpened')
                axes[1].axis('off')
                
                # Save figure
                output_filename = f"comparison_{filename}"
                output_path = os.path.join(output_dir, output_filename)
                plt.savefig(output_path, dpi=100, bbox_inches='tight')
                plt.close()
                
                print(f"✓ Saved: {output_filename}")
    
    print(f"\nEvaluation complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    evaluate_srgan(
        checkpoint_path='checkpoints/srgan_epoch_50.pth',
        lr_dir='data/blurred',
        output_dir='evaluation_results',
        batch_size=4,
        max_images=15
    )