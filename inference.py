import torch
from PIL import Image
import torchvision.transforms as transforms
import os
from model import Generator


def denormalize(tensor):
    """Convert tensor from [-1, 1] to [0, 1]"""
    return (tensor + 1) / 2


def super_resolve_image(image_path, output_path, model_path='models/generator_final.pth'):
    """Apply super-resolution to a single image"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    lr_img = transform(img).unsqueeze(0).to(device)
    
    # Generate super-resolved image
    with torch.no_grad():
        sr_img = generator(lr_img)
    
    # Convert back to PIL Image
    sr_img = denormalize(sr_img).squeeze(0).cpu()
    sr_img = transforms.ToPILImage()(sr_img)
    
    # Save result
    sr_img.save(output_path)
    print(f"Super-resolved image saved to {output_path}")
    return sr_img


def batch_super_resolve(input_dir, output_dir, model_path='checkpoints/generator_final.pth'):
    """Apply super-resolution to all images in a directory"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                super_resolve_image(input_path, output_path, model_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == '__main__':
    # Example: super-resolve a single image
    # super_resolve_image('test_images_blurred/test.png', 'test_results/test_sr.png')
    
    # Example: batch process all blurred images
    batch_super_resolve('test_images_blurred', 'test_results/super_resolved')
