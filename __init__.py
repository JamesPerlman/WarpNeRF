import time
import warp as wp
import numpy as np
import torch


from pathlib import Path

from warpnerf.models.batch import create_ray_batch
from warpnerf.models.camera import TrainingCamera
from warpnerf.models.gridrf_model import GridRFModel
from warpnerf.models.trimiprf_model import TrimipRFModel
from warpnerf.models.dataset import Dataset, DatasetType
from warpnerf.models.warpnerf_model import WarpNeRFModel
from warpnerf.rendering.nerf_renderer import generate_samples, render_samples, query_samples
from warpnerf.server.visualization import VisualizationServer
from warpnerf.training.trainer import Trainer
from warpnerf.utils.image import save_image
from warpnerf.utils.ray import get_rays_for_camera_kernel

wp.init()

# random_vec3s = wp.array(data=np.random.uniform(low=-100, high=100, size=(1000, 3)), dtype=wp.vec3f)
# contracted_vec3s = wp.empty(shape=(random_vec3s.shape[0],), dtype=wp.vec3f)
# wp.launch(
#     kernel=apply_merf_contraction_kernel,
#     dim=(random_vec3s.shape[0],),
#     inputs=[
#         random_vec3s
#     ],
#     outputs=[
#         contracted_vec3s
#     ]
# )
# print(random_vec3s)
# print(contracted_vec3s)
# exit()
# dataset = Dataset(path=Path("/home/luks/james/nerfs/turb-small"), type=DatasetType.BUNDLER)
dataset = Dataset(
    path=Path("/home/luks/james/nerfs/nerf_synthetic/lego/transforms_train.json"),
    type=DatasetType.TRANSFORMS_JSON,
)
dataset.load()
scene_extent = dataset.scene_bounding_box.max - dataset.scene_bounding_box.min
aabb_scale = 4 #max(scene_extent.x, scene_extent.y, scene_extent.z)
dataset.resize_and_center(aabb_scale=aabb_scale)

model = WarpNeRFModel(aabb_scale=aabb_scale)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
trainer = Trainer(dataset, model, optimizer)

def update_optimizer(model):
    global optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    global trainer
    trainer.opt = optimizer

model.on_subdivide_grid = update_optimizer

server = VisualizationServer()
server.set_dataset(dataset)

def render_camera(model, camera: TrainingCamera, img_w=None, img_h=None):
    if img_w is None:
        img_w, img_h = camera.image_dims
    
    n_pixels_per_chunk = 65536
    n_pixels_in_image = img_w * img_h
    n_chunks = (n_pixels_in_image + n_pixels_per_chunk - 1) // n_pixels_per_chunk

    rays = create_ray_batch(n_pixels_per_chunk)

    # Preallocate tensors for RGB and alpha
    rgb = torch.empty((n_pixels_in_image, 3), dtype=torch.float32)
    alpha = torch.empty((n_pixels_in_image,), dtype=torch.float32)

    for i in range(n_chunks):
        pixel_offset = i * n_pixels_per_chunk
        n_pixels_in_this_chunk = min(n_pixels_per_chunk, n_pixels_in_image - pixel_offset)
        
        if rays.count != n_pixels_in_this_chunk:
            
            rays = create_ray_batch(n_pixels_in_this_chunk)
        
        wp.launch(
            kernel=get_rays_for_camera_kernel,
            dim=rays.count,
            inputs=[
                camera.camera_data,
                img_w,
                img_h,
                pixel_offset,
            ],
            outputs=[
                rays
            ],
        )

        samples = generate_samples(model, rays)
        
        if samples.count == 0:
            continue

        samples = query_samples(model, samples)
        
        rgb_chunk, depth_chunk, alpha_chunk = render_samples(samples)

        # Copy chunks into preallocated tensors
        rgb[pixel_offset:pixel_offset + n_pixels_in_this_chunk] = rgb_chunk
        alpha[pixel_offset:pixel_offset + n_pixels_in_this_chunk] = alpha_chunk

    return rgb, alpha


def save_img(model, camera: TrainingCamera, idx: int):

    img_w, img_h = camera.image_dims
    img_w = img_w // 4
    img_h = img_h // 4

    rgb, alpha = render_camera(model, camera, img_w, img_h)
    a = (alpha * 255).to(dtype=torch.uint8)
    rgb = (rgb * 255).to(dtype=torch.uint8)

    rgba = torch.cat([rgb, a.unsqueeze(-1)], dim=1).reshape(img_w, img_h, 4)
    rgba = rgba.permute(2, 0, 1)
    rgba = wp.from_torch(rgba)
    save_image(rgba, f"/home/luks/james/nerfs/test_images/01-turb-small-{idx:05d}.png")


def server_render():
    viewport_cam = server.get_viewport_camera(aabb_scale)
    if viewport_cam is None:
        return
    rgb, alpha = render_camera(model, viewport_cam)
    w, h = viewport_cam.image_dims
    rgb = np.array(rgb.reshape(w, h, 3).permute(1,0,2).detach().cpu().numpy(), dtype=np.float32)
    server.set_background_image(rgb)

for i in range(2000):
    trainer.step()

    if i % 25 == 0:
        server_render()

    if i % 100 == 0:
        save_img(model, dataset.training_cameras[0], i)

wp.synchronize()

while True:
    time.sleep(0.1)
    server_render()
