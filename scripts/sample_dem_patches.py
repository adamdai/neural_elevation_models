import numpy as np
from PIL import Image

heightmap_path = '../data/unreal_moon_heightmap.png'
image = Image.open(heightmap_path)

# Sample 256x256 patches from the heightmap
n_samples = 100
patch_size = 256

for i in range(n_samples):
    x = np.random.randint(0, image.width - patch_size)
    y = np.random.randint(0, image.height - patch_size)
    patch = image.crop((x, y, x + patch_size, y + patch_size))
    patch.save(f'../data/dem_patches/patch_{i}.png')
    print(f"Saved patch {i}")