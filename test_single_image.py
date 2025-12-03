from inference import super_resolve_image
import os

# Test on a single image
input_image = 'gau2.png'  # Change to any image filename
output_image = 'test_results/test_deblurred.png'

# Make sure output directory exists
os.makedirs('output', exist_ok=True)

# Deblur the image
super_resolve_image(input_image, output_image)
print(f"Done! Check {output_image}")
