import time
import warp as wp
import numpy as np
import torch


from pathlib import Path

from warpnerf.models.batch import create_ray_batch
from warpnerf.models.camera import CameraData, TrainingCamera
from warpnerf.models.dataset import Dataset, DatasetType
from warpnerf.models.warpnerf_model import WarpNeRFModel
from warpnerf.rendering.nerf_renderer import generate_samples, render_samples, query_samples
from warpnerf.server.visualization import VisualizationServer
from warpnerf.training.trainer import Trainer
from warpnerf.utils.image import save_image
from warpnerf.utils.cameras import get_rays_for_camera_kernel

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
# dataset = Dataset(
#     path=Path("/home/luks/james/nerfs/nerf_synthetic/lego/transforms_train.json"),
#     type=DatasetType.TRANSFORMS_JSON,
# )
# 0179-0184
proj_path = Path("/home/luks/james/nerfs/summertarget/NeRF_0180")
# proj_path = Path("/home/luks/james/nerfs/pipe-thingy-makawao/")
test_render_frames_path = proj_path / "test_render_frames"
test_render_frames_path.mkdir(exist_ok=True)

dataset = Dataset(
    path=proj_path / "transforms.json",
    type=DatasetType.TRANSFORMS_JSON,
)
# dataset = Dataset(
#     path=Path("/home/luks/james/nerfs/pipe-thingy-makawao/transforms.json"),
#     type=DatasetType.TRANSFORMS_JSON,
# )
dataset.load()


num_cams = dataset.num_cameras

cam_idx_a = num_cams // 5
cam_idx_b = 4 * num_cams // 5

aabb_size = dataset.scene_bounding_box.max - dataset.scene_bounding_box.min
aabb_scale = max(aabb_size.x, aabb_size.y, aabb_size.z)
aabb_scale = 8.0
dataset.resize_and_center(aabb_scale=aabb_scale)

model = WarpNeRFModel(aabb_scale=aabb_scale, n_appearance_embeddings=dataset.num_images)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)
trainer = Trainer(dataset, model, optimizer)

server = VisualizationServer()
server.set_dataset(dataset)

def render_camera(model, camera: TrainingCamera, img_w=None, img_h=None):
    with torch.no_grad():
        if img_w is None:
            img_w, img_h = camera.image_dims
        
        n_pixels_per_chunk = 65536
        n_pixels_in_image = img_w * img_h
        n_chunks = (n_pixels_in_image + n_pixels_per_chunk - 1) // n_pixels_per_chunk

        rays = create_ray_batch(n_pixels_per_chunk)

        # Preallocate tensors for RGB and alpha
        rgb = torch.zeros((n_pixels_in_image, 3), dtype=torch.float32)
        alpha = torch.zeros((n_pixels_in_image,), dtype=torch.float32)

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
    with torch.no_grad():
        img_w, img_h = camera.image_dims

        rgb, alpha = render_camera(model, camera, img_w, img_h)
        a = (alpha * 255).to(dtype=torch.uint8)
        rgb = (rgb * 255).to(dtype=torch.uint8)

        rgba = torch.cat([rgb, a.unsqueeze(-1)], dim=1).reshape(img_w, img_h, 4)
        rgba = rgba.permute(2, 0, 1)
        rgba = wp.from_torch(rgba)
        save_image(rgba, test_render_frames_path / f"step-{idx:05d}.png")


def server_render():
    viewport_cam = server.get_viewport_camera(aabb_scale)
    if viewport_cam is None:
        return
    rgb, alpha = render_camera(model, viewport_cam)
    w, h = viewport_cam.image_dims
    rgb = np.array(rgb.reshape(w, h, 3).permute(1,0,2).detach().cpu().numpy(), dtype=np.float32)
    server.set_background_image(rgb)

max_step = 9000
n_frames = 300
exp_frames = []
for j in range(0, n_frames, 2):
    # exponential from 0 to 10,000
    val = ((max_step + 1) ** (j / (n_frames - 1))) - 1
    i_val = int(round(val))
    exp_frames.append(i_val)

exp_frames_set = set(exp_frames)
frame_num = 0

def interpolated_cam(cam_a: TrainingCamera, cam_b: TrainingCamera, t: float) -> TrainingCamera:
    cam_data_a = cam_a.camera_data
    cam_data_b = cam_b.camera_data
    cam_data_c = CameraData()
    cam_data_c.f = cam_data_a.f + t * (cam_data_b.f - cam_data_a.f)
    cam_data_c.sx = cam_data_a.k1 + t * (cam_data_b.k1 - cam_data_a.k1)
    cam_data_c.k2 = cam_data_a.k2 + t * (cam_data_b.k2 - cam_data_a.k2)
    cam_data_c.p1 = cam_data_a.p1 + t * (cam_data_b.p1 - cam_data_a.p1)
    cam_data_c.p2 = cam_data_a.p2 + t * (cam_data_b.p2 - cam_data_a.p2)
    
    quat_a = wp.quat_from_matrix(cam_data_a.R)
    quat_b = wp.quat_from_matrix(cam_data_b.R)
    quat_c = wp.quat_slerp(quat_a, quat_b, t)
    cam_data_c.R = wp.quat_to_matrix(quat_c)
    cam_data_c.t = cam_data_a.t + t * (cam_data_b.t - cam_data_a.t)
    
    return TrainingCamera(cam_data_c, cam_a.image_path, cam_a.image_dims)

for i in range(max_step + 1):
    trainer.step()
    scheduler.step()

    if i % 100 == 0 and i > 0:
        server_render()

    if i % (max_step // n_frames) == 0:
        cam = interpolated_cam(dataset.training_cameras[cam_idx_a], dataset.training_cameras[cam_idx_b], i / max_step)
        cam.image_dims = (1024, 1024)
        cam.camera_data.sx = 1024
        cam.camera_data.sy = 1024
        cam.camera_data.cx = 512
        cam.camera_data.cy = 512
        cam.camera_data.f = 400
        save_img(model, cam, frame_num)
        frame_num += 1
wp.synchronize()

while True:
    time.sleep(0.1)
    server_render()
