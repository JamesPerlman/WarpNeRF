import warp as wp

from pathlib import Path
from typing import List

from warpnerf.kernels.image import copy_image_rgba_to_planar_kernel
from warpnerf.models.camera import CameraData, TrainingCamera, create_camera_data_from_bundler
from warpnerf.utils.bundler_sfm import BundlerSFMData, create_bundler_sfm_data_from_path
from warpnerf.utils.image import save_image

def get_image_paths_from_lst(path: Path) -> List[Path]:
    with path.open("r") as f:
        return [path.parent / Path(line.strip()) for line in f.readlines()]

class Dataset:
    
    path: Path
    bundler_data: BundlerSFMData = None
    training_cameras: List[TrainingCamera] = []

    def __init__(self, path: Path):
        self.path = path

    @property
    def is_loaded(self) -> bool:
        return self.bundler_data is not None
    
    @property
    def num_cameras(self) -> int:
        return len(self.training_cameras)
    
    @property
    def camera_data(self) -> wp.array1d(dtype=CameraData):
        if hasattr(self, "_camera_data"):
            return self._camera_data
        
        cpu_cams = [camera.camera_data for camera in self.training_cameras]
        self._camera_data = wp.array(cpu_cams, dtype=CameraData, device="cuda")
        return self._camera_data
    
    @property
    def image_data(self) -> wp.array2d(dtype=wp.float32):
        if hasattr(self, "_image_data"):
            return self._image_data
        
        first_image = self.training_cameras[0].get_image()
        img_w, img_h = first_image.shape[1:]
        n_images = len(self.training_cameras)

        self._image_data = wp.array(
            shape=(n_images, 4, img_w, img_h),
            dtype=wp.uint8,
            device="cuda",
        )

        for i, camera in enumerate(self.training_cameras):
            if i >= n_images:
                break
            image = camera.get_image()
            wp.copy(self._image_data, image, i * 4 * img_w * img_h)
            print(f"Loaded image {i+1}/{len(self.training_cameras)}")

        wp.synchronize()

        return self._image_data

    def load(self):

        if self.is_loaded:
            return

        self.bundler_data = create_bundler_sfm_data_from_path(self.path / "registration.out", read_points=False)

        # images are named "00000.png", "00001.png", etc.
        self.image_paths = [self.path / f"{i:05d}.png" for i in range(len(self.bundler_data.cameras))]

        self.training_cameras = []
        for bundler_camera_data, image_path in zip(self.bundler_data.cameras, self.image_paths):
            camera_data = create_camera_data_from_bundler(bundler_camera_data)
            training_camera = TrainingCamera(camera_data, image_path)
            self.training_cameras.append(training_camera)
            
    