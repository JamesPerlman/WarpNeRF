from pathlib import Path
from PIL import Image

import warp as wp
import numpy as np

def is_image(path: Path) -> bool:
    return path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

def load_image(path: Path) -> wp.array(dtype=float):
    assert is_image(path), f"{path} is not an image file"
    img = Image.open(path)
    img = np.array(img)
    img = img.astype(np.float32) / 255.0
    return wp.from_numpy(img, dtype=wp.float32)

def save_image(data: wp.array(dtype=float), path: Path):
    img = np.array(data.to("cpu"))
    img = (img * 255.0).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)
