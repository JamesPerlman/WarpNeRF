from enum import Enum
import json
import math
import numpy as np
import warp as wp

from pathlib import Path
from typing import List

from warpnerf.models.batch import RayBatch, create_ray_batch
from warpnerf.models.bounding_box import BoundingBox
from warpnerf.models.camera import CameraData, TrainingCamera, create_camera_data_from_bundler, get_scene_bounding_box
from warpnerf.training.batcher import init_training_rays_and_pixels_kernel
from warpnerf.utils.bundler_sfm import BundlerSFMData, create_bundler_sfm_data_from_path
from warpnerf.utils.image import get_image_dims

def get_image_paths_from_lst(path: Path) -> List[Path]:
    with path.open("r") as f:
        return [path.parent / Path(line.strip()) for line in f.readlines()]

@wp.kernel
def copy_and_transpose_image_data_kernel(
    rgba_in: wp.array3d(dtype=wp.uint8),
    img_index: wp.int32,
    rgba_out: wp.array4d(dtype=wp.uint8),
):
    i, j = wp.tid()
    
    rgba_out[img_index][0][i][j] = rgba_in[j][i][0]
    rgba_out[img_index][1][i][j] = rgba_in[j][i][1]
    rgba_out[img_index][2][i][j] = rgba_in[j][i][2]
    rgba_out[img_index][3][i][j] = rgba_in[j][i][3]

# dataset types string union
class DatasetType(Enum):
    BUNDLER = "bundler"
    TRANSFORMS_JSON = "transforms_json"

class Dataset:
    
    path: Path
    is_loaded: bool = False
    training_cameras: List[TrainingCamera] = []
    type: DatasetType

    def __init__(self, path: Path, type: DatasetType): 
        self.path = path
        self.type = type
    
    @property
    def num_cameras(self) -> int:
        assert self.is_loaded, "Dataset not loaded"
        return len(self.training_cameras)
    
    @property
    def num_images(self) -> int:
        return self.num_cameras
    
    @property
    def image_dims(self) -> tuple[int, int]:
        assert self.is_loaded, "Dataset not loaded"
        
        if not hasattr(self, "_image_dims"):
            self._image_dims = self.training_cameras[0].image_dims
        
        return self._image_dims

    @property
    def num_pixels_total(self) -> int:
        assert self.is_loaded, "Dataset not loaded"

        img_w, img_h = self.image_dims
        return img_w * img_h * self.num_images
    
    @property
    def camera_data(self) -> wp.array1d(dtype=CameraData):
        assert self.is_loaded, "Dataset not loaded"

        if hasattr(self, "_camera_data"):
            return self._camera_data
        
        cpu_cams = [camera.camera_data for camera in self.training_cameras]
        self._camera_data = wp.array(cpu_cams, dtype=CameraData, device="cuda")
        return self._camera_data
    
    @property
    def image_data(self) -> wp.array4d(dtype=wp.float32):
        assert self.is_loaded, "Dataset not loaded"

        if hasattr(self, "_image_data"):
            return self._image_data

        img_w, img_h = self.image_dims
        
        self._image_data = wp.array(
            shape=(self.num_images, 4, img_w, img_h),
            dtype=wp.uint8,
            device="cuda",
        )

        for i, camera in enumerate(self.training_cameras):
            image = wp.from_numpy(camera.get_image(), dtype=wp.uint8, device="cuda")
            wp.launch(
                copy_and_transpose_image_data_kernel,
                dim=(img_w, img_h),
                inputs=[image, i],
                outputs=[self._image_data],
            )
            print(f"Loaded image {i+1}/{len(self.training_cameras)}")

        return self._image_data

    @property
    def scene_bounding_box(self) -> BoundingBox:
        assert self.is_loaded, "Dataset not loaded"
        
        return get_scene_bounding_box(self.training_cameras)
    
    @property
    def scene_center(self) -> wp.vec3f:
        assert self.is_loaded, "Dataset not loaded"
        
        bbox = self.scene_bounding_box
        return 0.5 * (bbox.min + bbox.max)

    def load(self):

        if self.is_loaded:
            return

        if self.type == DatasetType.BUNDLER:
            bundler_data = create_bundler_sfm_data_from_path(self.path / "registration.out", read_points=False)

            # images are named "00000.png", "00001.png", etc.
            self.image_paths = [self.path / f"{i:05d}.png" for i in range(len(bundler_data.cameras))]

            self.training_cameras = []
            for bundler_camera_data, image_path in zip(bundler_data.cameras, self.image_paths):
                image_dims = get_image_dims(image_path)
                camera_data = create_camera_data_from_bundler(bundler_camera_data, image_dims)
                training_camera = TrainingCamera(camera_data, image_path, image_dims)
                self.training_cameras.append(training_camera)

        if self.type == DatasetType.TRANSFORMS_JSON:
            json_data: json = None
            with open(self.path, "r") as f:
                json_data = json.load(f)
            
            frames = json_data["frames"]
            self.image_paths = [self.path.parent / f["file_path"] for f in frames]
            self.image_paths = [p.parent.parent / "images" / p.name for p in self.image_paths]

            img_w, img_h = get_image_dims(self.image_paths[0])
            cam_w = json_data.get("w", img_w)
            cam_h = json_data.get("h", img_h)
            if "camera_angle_x" in json_data:
                camera_angle_x = json_data["camera_angle_x"]
                camera_fx = 0.5 * cam_w / math.tan(0.5 * camera_angle_x)
            if "fl_x" in json_data:
                camera_fx = json_data["fl_x"]
            
            camera_fy = json_data.get("fl_y", camera_fx)
            camera_cx = json_data.get("cx", cam_w / 2)
            camera_cy = json_data.get("cy", cam_h / 2)
            
            camera_k1 = json_data["k1"] if "k1" in json_data else 0.0
            camera_k2 = json_data["k2"] if "k2" in json_data else 0.0
            camera_p1 = json_data["p1"] if "p1" in json_data else 0.0
            camera_p2 = json_data["p2"] if "p2" in json_data else 0.0

            self.training_cameras = []
            for i, frame in enumerate(frames):
                M = np.array(frame["transform_matrix"])
                
                cam_data = CameraData()
                cam_data.f = camera_fx
                cam_data.sx = cam_w
                cam_data.sy = cam_h
                cam_data.k1 = camera_k1
                cam_data.k2 = camera_k2
                cam_data.p1 = camera_p1
                cam_data.p2 = camera_p2
                cam_data.cx = camera_cx
                cam_data.cy = camera_cy
                
                R = wp.mat33f(M[:3, :3])
                FLIP_MAT = wp.mat33f(
                    [[1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]]
                )
                cam_data.R = wp.mul(R, FLIP_MAT)
                cam_data.t = wp.vec3f(M[:3, 3])

                training_camera = TrainingCamera(cam_data, self.image_paths[i], (img_w, img_h))
                self.training_cameras.append(training_camera)
            
        self.is_loaded = True
    
    def resize_and_center(self, aabb_scale: float):
        assert self.is_loaded, "Dataset not loaded"

        scene_bbox = self.scene_bounding_box
        scene_center = 0.5 * (scene_bbox.min + scene_bbox.max)
        scene_extent = scene_bbox.max - scene_bbox.min
        scene_max_extent = max(scene_extent.x, scene_extent.y, scene_extent.z)
        scale = 1.0 / scene_max_extent * aabb_scale

        for camera in self.training_cameras:
            camera.camera_data.t = scale * (camera.camera_data.t - scene_center)
            camera.camera_data.f *= scale
            camera.camera_data.sx *= scale
            camera.camera_data.sy *= scale
            camera.camera_data.cx *= scale
            camera.camera_data.cy *= scale


    def get_batch(self, n_rays: int, random_seed: wp.int32, device: str = "cuda") -> tuple[RayBatch, wp.array(dtype=wp.vec4f)]:
        assert self.is_loaded, "Dataset not loaded"

        batch = create_ray_batch(n_rays, device=device)
        rgba = wp.empty(shape=(4, n_rays), dtype=wp.uint8, device=device)

        pixel_indices = wp.array(np.random.choice(self.num_pixels_total, n_rays), dtype=wp.uint64, device=device)
        random_xy_noise = wp.array(np.random.ranf((n_rays, 2)), dtype=wp.float32, device=device)

        wp.launch(
            init_training_rays_and_pixels_kernel,
            dim=n_rays,
            inputs=[
                self.image_dims,
                self.camera_data,
                self.image_data,
                pixel_indices,
                random_xy_noise,
            ],
            outputs=[
                batch,
                rgba
            ],
            device=device,
        )

        return batch, rgba
