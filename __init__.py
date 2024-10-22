import time
import warp as wp
import numpy as np
import torch


from pathlib import Path

from warpnerf.models.batch import RayBatch, create_ray_batch
from warpnerf.models.camera import CameraData, TrainingCamera
from warpnerf.models.trimiprf_model import TrimipRFModel
from warpnerf.server.visualization import VisualizationServer
from warpnerf.training.batcher import init_training_rays_and_pixels_kernel
from warpnerf.models.dataset import Dataset
from warpnerf.training.trainer import Trainer
from warpnerf.utils.image import save_image
from warpnerf.utils.ray import get_rays_for_camera_kernel

wp.init()

dataset = Dataset(path=Path("/home/luks/james/nerfs/turb-small"))
dataset.load()

model = TrimipRFModel()
optimizer = torch.optim.Adam(model.parameters())
trainer = Trainer(dataset, model, optimizer)

def render_camera(model, camera: TrainingCamera, idx: int):
    img_w, img_h = camera.get_image_dims()
    n_pixels = img_w * img_h
    ray_batch = create_ray_batch(n_pixels)
    wp.launch(
        kernel=get_rays_for_camera_kernel,
        dim=(img_w, img_h),
        inputs=[
            camera.camera_data,
            img_w,
            img_h,
        ],
        outputs=[
            ray_batch
        ],
    )
    rgb, depth, alpha = trainer.renderer(model, ray_batch)
    a = (alpha * 255).to(dtype=torch.uint8)
    rgb = (rgb * 255).to(dtype=torch.uint8)
    rgba = torch.cat([rgb, a.unsqueeze(-1)], dim=1).reshape(img_w, img_h, 4).permute(2, 0, 1)
    rgba = wp.from_torch(rgba)
    save_image(rgba, f"/home/luks/james/nerfs/test_images/01-turb-small-{idx:05d}.png")

for i in range(1000):
    trainer.step()

    if i % 100 == 0:
        render_camera(model, dataset.training_cameras[0], i)

wp.synchronize()

# server = VisualizationServer()
# server.set_dataset(dataset)

# while True:
#     time.sleep(0.01)
