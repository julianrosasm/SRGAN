import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from model import Generator


def compare_images(input_path, output_dir='test_results', model_path='models/generator_final.pth', generator=None, device=None):
    """Load image, deblur it, and show before/after comparison"""
    
    # Setup device (if not provided)
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    
    # Load model (if not provided)
    if generator is None:
        print(f"Loading model from {model_path}...")
        generator = Generator().to(device)
        generator.load_state_dict(torch.load(model_path, map_location=device))
        generator.eval()
        print("Model loaded successfully!")
    
    # Load and preprocess image
    print(f"Loading image: {input_path}")
    original_img = Image.open(input_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    lr_img = transform(original_img).unsqueeze(0).to(device)
    
    # Generate deblurred image
    print("Generating deblurred image...")
    with torch.no_grad():
        sr_img = generator(lr_img)
    
    # Convert back to PIL Image
    def denormalize(tensor):
        return (tensor + 1) / 2
    
    sr_img = denormalize(sr_img).squeeze(0).cpu()
    sr_img = transforms.ToPILImage()(sr_img)
    
    # Save output
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    
    output_path = os.path.join(output_dir, f'{name}_deblurred{ext}')
    sr_img.save(output_path)
    
    # Show comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_img)
    axes[0].set_title('Original (Blurry)', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(sr_img)
    axes[1].set_title('Deblurred by Model', fontsize=14)
    axes[1].axis('off')
    
    # Calculate difference first
    import numpy as np
    orig_array = np.array(original_img)
    deblur_array = np.array(sr_img)
    diff = np.abs(orig_array.astype(float) - deblur_array.astype(float)).mean()
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, f'{name}_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close instead of show
    
    print(f"✓ Deblurred: {output_path}")
    print(f"✓ Comparison: {comparison_path}")
    
    print(f"  Pixel difference: {diff:.2f}")
    
    return diff


def batch_compare(input_dir, output_dir='test_results/comparisons', model_path='models/generator_final.pth'):
    """Process all images in a directory"""
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load model once
    print(f"Loading model from {model_path}...")
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    print("Model loaded successfully!\n")
    
    # Get all images
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process\n")
    
    # Process each image
    differences = []
    for i, filename in enumerate(image_files, 1):
        input_path = os.path.join(input_dir, filename)
        print(f"[{i}/{len(image_files)}] Processing {filename}...")
        
        try:
            diff = compare_images(input_path, output_dir, model_path, generator, device)
            differences.append(diff)
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        print()
    
    # Summary
    if differences:
        avg_diff = sum(differences) / len(differences)
        print(f"\n{'='*50}")
        print(f"Processed {len(differences)} images successfully")
        print(f"Average pixel difference: {avg_diff:.2f}")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*50}")


if __name__ == '__main__':
    # Option 1: Process a single image
    # compare_images('image.png')
    
    # Option 2: Process all images in a folder
    input_folder = 'test_images_blurred'  # Change to your folder
    
    if not os.path.exists(input_folder):
        print(f"Error: {input_folder} not found!")
    else:
        batch_compare(input_folder, output_dir='test_results/comparisons')
