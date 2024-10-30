import time
import warp as wp
import numpy as np
import torch


from pathlib import Path

from warpnerf.models.batch import create_ray_batch
from warpnerf.models.camera import TrainingCamera
from warpnerf.models.trimiprf_model import TrimipRFModel
from warpnerf.models.dataset import Dataset
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
dataset = Dataset(path=Path("/home/luks/james/nerfs/turb-small"))
dataset.load()
aabb_scale = 8.0
dataset.resize_and_center(aabb_scale=aabb_scale)
print(dataset.scene_bounding_box.max - dataset.scene_bounding_box.min)
model = TrimipRFModel(aabb_scale=aabb_scale)
optimizer = torch.optim.Adam(model.parameters())
trainer = Trainer(dataset, model, optimizer)

server = VisualizationServer()
server.set_dataset(dataset)


def render_camera(model, camera: TrainingCamera, img_w = None, img_h = None):
    if img_w is None:
        img_w, img_h = camera.image_dims
    
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
    return rgb, alpha

def save_img(model, camera: TrainingCamera, idx: int):

    img_w, img_h = camera.image_dims
    img_w = img_w // 10
    img_h = img_h // 10

    rgb, alpha = render_camera(model, camera, img_w, img_h)
    a = (alpha * 255).to(dtype=torch.uint8)
    rgb = (rgb * 255).to(dtype=torch.uint8)

    rgba = torch.cat([rgb, a.unsqueeze(-1)], dim=1).reshape(img_w, img_h, 4)
    rgba = rgba.permute(2, 0, 1)
    rgba = wp.from_torch(rgba)
    save_image(rgba, f"/home/luks/james/nerfs/test_images/01-turb-small-{idx:05d}.png")



def server_render():
    viewport_cam = server.get_viewport_camera()
    rgb, alpha = render_camera(model, viewport_cam)
    w, h = viewport_cam.image_dims
    rgb = np.array(rgb.reshape(w, h, 3).detach().cpu().numpy(), dtype=np.float32)
    print("rgb dtype:", rgb.dtype)
    print("rgb shape:", rgb.shape)
    print("rgb max/min values:", rgb.max(), rgb.min())
    server.set_background_image(rgb)

for i in range(200):
    trainer.step()

    if i % 100 == 0:
        save_img(model, dataset.training_cameras[0], i)
        server_render()

wp.synchronize()

while True:
    time.sleep(0.1)
    server_render()
