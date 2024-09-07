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

ray_batch = create_ray_batch(count=1000)
rgba_batch = wp.empty((4, 1000), dtype=wp.uint8, device="cuda")

random_seed = 1337
wp.launch(
    init_training_rays_and_pixels_kernel,
    dim=1000,
    inputs=[random_seed, dataset.num_images, dataset.image_dims, dataset.camera_data, dataset.image_data],
    outputs=[ray_batch, rgba_batch],
)

wp.synchronize()

# server = VisualizationServer()
# server.set_dataset(dataset)

# while True:
#     time.sleep(0.01)
