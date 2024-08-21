from pathlib import Path
from PIL import Image

import warp as wp
import numpy as np

def is_image(path: Path) -> bool:
    return path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

def get_image_dims(path: Path) -> tuple[int, int]:
    assert is_image(path), f"{path} is not an image file"
    img = Image.open(path)
    return img.size

# loads an image to the GPU
def load_image(path: Path) -> np.ndarray:
    assert is_image(path), f"{path} is not an image file"
    img = Image.open(path)
    return np.array(img)

# save image from planar format (rrr, ggg, bbb, aaa)
def save_image(data: wp.array3d(dtype=wp.uint8), path: Path):
    img = np.array(data.to("cpu")).transpose((2, 1, 0))
    img = Image.fromarray(img)
    img.save(path)
