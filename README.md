# SRGAN Image Deblurring Model

A Super-Resolution Generative Adversarial Network (SRGAN) trained to deblur images. This model learns to remove Gaussian blur from images using paired training data.

## Overview

This implementation uses a GAN architecture with:

- **Generator**: 8 residual blocks that learn to deblur images
- **Discriminator**: Evaluates image quality to push the generator to produce realistic results
- **Perceptual Loss**: Uses VGG19 features to maintain visual quality
- **Training Data**: Pairs of blurred and unblurred images

## Requirements

- Python 3.11+
- PyTorch 2.0+
- torchvision
- PIL (Pillow)
- matplotlib
- numpy
- tqdm

## Installation

1. Clone this repository
2. Create a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

Organize your data in the following structure:

```
data/
  blurred/        # Low-quality/blurred images
  unblurred/      # High-quality/sharp images
```

Images in both folders should have matching filenames. The training script expects PNG files but can be modified to support other formats.

## Training

### Basic Training

Run with default settings:

```bash
python train.py
```

### Configurable Parameters

The main training parameters can be modified in `train.py` (lines 157-162):

```python
train_srgan(
    lr_dir='data/blurred',           # Path to blurred images
    hr_dir='data/unblurred',         # Path to sharp images
    num_epochs=50,                   # Number of training epochs
    batch_size=2,                    # Batch size
    lr=1e-4                          # Learning rate
)
```

### Hardware-Specific Settings

**For MacBook Air M1 (default settings):**

- `batch_size=2`
- `num_epochs=50`
- Image size: 128x128 (set in `dataset.py`)

**For More Powerful GPUs:**

- Increase `batch_size` to 8-16
- Increase `num_epochs` to 100-200
- Increase image size in `dataset.py` line 53: `hr_size=256`
- Increase residual blocks in `model.py` line 40: `num_residual_blocks=16`

**For CPU-Only Training:**

- Reduce `batch_size=1`
- Reduce `num_epochs=20`
- Consider reducing residual blocks to 4

### Device Selection

The training script automatically detects and uses:

1. MPS (Metal Performance Shaders) for M1/M2 Macs
2. CUDA for NVIDIA GPUs
3. CPU as fallback

To force CPU training, modify `train.py` lines 41-47.

## Inference (Using the Trained Model)

### Test a Single Image

```python
from inference import super_resolve_image

super_resolve_image('path/to/blurry.png', 'path/to/output.png')
```

### Batch Process Multiple Images

```python
from inference import batch_super_resolve

batch_super_resolve('input_folder', 'output_folder')
```

### Compare Results with test_comparison.py

Process all images in a folder and generate side-by-side comparisons:

```bash
python test_comparison.py
```

Modify the input folder on line 154:

```python
input_folder = 'test_images_blurred'  # Change to your folder
```

Results are saved to `test_results/comparisons/` with:

- Deblurred images
- Side-by-side comparison images

## Model Files

Trained models are saved in the `models/` directory:

- `generator_final.pth` - Final trained generator (use this for inference)
- `srgan_epoch_20.pth` - Checkpoint at epoch 20
- `srgan_epoch_40.pth` - Checkpoint at epoch 40

## Configuration Options

### Blur Strength (gaussian_blur.py)

The training data blur strength can be adjusted on line 13:

```python
blurred_img = img.filter(ImageFilter.GaussianBlur(radius=2))
```

Increase `radius` (e.g., 4-8) for stronger blur training.

### Image Size (dataset.py)

Change the training/inference image size on line 53:

```python
def get_dataloaders(lr_dir='data/blurred', hr_dir='data/unblurred', batch_size=16, hr_size=128):
```

Larger sizes (256, 512) require more memory but preserve more detail.

### Model Complexity (model.py)

Adjust the number of residual blocks on line 40:

```python
def __init__(self, num_residual_blocks=8):
```

More blocks (16, 32) = better quality but slower training and more memory.

### Training Schedule (train.py)

- **Validation frequency** (line 135): Change `if (epoch + 1) % 10 == 0` to validate more/less often
- **Checkpoint frequency** (line 145): Change `if (epoch + 1) % 20 == 0` to save checkpoints more/less often

## Improving Results

To get more noticeable deblurring:

1. **Train on stronger blur**: Increase blur radius to 4-8 in `gaussian_blur.py`
2. **Longer training**: Increase epochs to 100-200
3. **Larger images**: Use 256x256 or larger (requires more memory)
4. **More residual blocks**: Increase to 12-16 (requires more memory)
5. **Diverse blur types**: Add motion blur, lens blur, etc. to training data

## Troubleshooting

### Out of Memory Errors

- Reduce `batch_size` to 1
- Reduce image size to 64x64 or 96x96
- Reduce `num_residual_blocks` to 4
- Set `num_workers=0` in `dataset.py`

### Training Too Slow

- Reduce `num_epochs`
- Reduce image size
- Use a smaller model (fewer residual blocks)

### Poor Results

- Train longer (more epochs)
- Ensure blur in test images matches training blur
- Increase model capacity (more residual blocks)
- Use larger training dataset (800+ image pairs)

## Model Architecture

- **Input**: Blurred RGB image (3 channels)
- **Output**: Deblurred RGB image (same size as input)
- **Generator**: ResNet-style with skip connections
- **No upsampling**: Maintains input resolution (for deblurring, not super-resolution)
