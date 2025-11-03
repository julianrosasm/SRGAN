import os
from PIL import Image, ImageFilter, UnidentifiedImageError

input_folder = 'data/unblurred'
output_folder = 'data/blurred'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith('.png'):
        img_path = os.path.join(input_folder, filename)
        try:
            img = Image.open(img_path)
            blurred_img = img.filter(ImageFilter.GaussianBlur(radius=2))
            output_path = os.path.join(output_folder, filename)
            blurred_img.save(output_path)
        except UnidentifiedImageError:
            print(f"Skipping unreadable image: {filename}")