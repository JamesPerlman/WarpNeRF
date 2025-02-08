
import numpy as np
import torch
import warp as wp
from PIL import Image

from warpnerf.models.batch import create_ray_batch
from warpnerf.models.camera import TrainingCamera
from warpnerf.models.nerf_model import NeRFModel
from warpnerf.rendering.nerf_renderer import generate_samples, query_samples, render_samples
from warpnerf.utils.cameras import get_rays_for_camera_kernel

def get_rendered_image(model: NeRFModel, camera: TrainingCamera, img_w=None, img_h=None) -> Image.Image:
    with torch.no_grad():
        if img_w is None:
            img_w, img_h = camera.image_dims
        
        n_pixels_per_chunk = 65536
        n_pixels_in_image = img_w * img_h
        n_chunks = (n_pixels_in_image + n_pixels_per_chunk - 1) // n_pixels_per_chunk

        rays = create_ray_batch(n_pixels_per_chunk)

        # Preallocate tensors for RGB and alpha
        rgb = torch.zeros((n_pixels_in_image, 3), dtype=torch.float32)
        # alpha = torch.zeros((n_pixels_in_image,), dtype=torch.float32)

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
            # alpha[pixel_offset:pixel_offset + n_pixels_in_this_chunk] = alpha_chunk

        rgb = np.array(
            rgb.reshape(img_w, img_h, 3)
                .permute(1,0,2)
                .detach()
                .cpu()
                .numpy(),
            dtype=np.float32
        )
        
        img = Image.fromarray((rgb * 255).astype(np.uint8))

        return img
