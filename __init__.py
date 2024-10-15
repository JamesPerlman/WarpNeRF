import time
import warp as wp
import numpy as np


from pathlib import Path

from warpnerf.models.batch import RayBatch, create_ray_batch
from warpnerf.server.visualization import VisualizationServer
from warpnerf.training.batcher import init_training_rays_and_pixels_kernel
from warpnerf.training.dataset import Dataset
from warpnerf.utils.image import save_image

wp.init()

dataset = Dataset(path=Path("/home/luks/james/nerfs/turb-small"))
dataset.load()

random_seed = 1336
rays, rgba = dataset.get_batch(n_rays=1000, random_seed=random_seed)

wp.synchronize()

print(rgba)

# server = VisualizationServer()
# server.set_dataset(dataset)

# while True:
#     time.sleep(0.01)
