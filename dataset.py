import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class SRGANDataset(Dataset):
    """Dataset for loading blurred (LR) and unblurred (HR) image pairs"""
    def __init__(self, lr_dir, hr_dir, hr_size=256):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        
        # Get all image filenames
        self.image_files = [f for f in os.listdir(lr_dir) if f.lower().endswith('.png')]
        
        # Transforms
        self.hr_transform = transforms.Compose([
            transforms.Resize((hr_size, hr_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.lr_transform = transforms.Compose([
            transforms.Resize((hr_size, hr_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        
        # Load low-resolution (blurred) image
        lr_path = os.path.join(self.lr_dir, filename)
        lr_img = Image.open(lr_path).convert('RGB')
        
        # Load high-resolution (unblurred) image
        hr_path = os.path.join(self.hr_dir, filename)
        hr_img = Image.open(hr_path).convert('RGB')
        
        lr_img = self.lr_transform(lr_img)
        hr_img = self.hr_transform(hr_img)
        
        return lr_img, hr_img


def get_dataloaders(lr_dir='data/blurred', hr_dir='data/unblurred', batch_size=16, hr_size=256):
    """Create train and validation dataloaders"""
    dataset = SRGANDataset(lr_dir, hr_dir, hr_size)
    
    # Split dataset (80% train, 20% val)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader
