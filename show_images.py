import os
import matplotlib.pyplot as plt
from PIL import Image

data_path = "data/"

files = os.listdir(data_path)
print("Number of images:", len(files))

for i, f in enumerate(files[:5]):
    img_path = os.path.join(data_path, f)
    img = Image.open(img_path)

    plt.imshow(img)
    plt.title(f"Image {i+1}: {f}")
    plt.axis('off')
    plt.show()
