import warp as wp

from pathlib import Path
from warpnerf.utils.bundler_sfm import BundlerSFMCameraData
from warpnerf.utils.image import load_image

@wp.struct
class CameraData:
    f: wp.float32
    k1: wp.float32
    k2: wp.float32
    R: wp.array(dtype=wp.mat33f)
    t: wp.array(dtype=wp.vec3f)

def create_camera_data_from_bundler(data: BundlerSFMCameraData) -> CameraData:
    cam = CameraData()
    cam.f = data.f
    cam.k1 = data.k1
    cam.k2 = data.k2
    cam.R = wp.from_numpy(data.R, dtype=wp.mat33f)
    cam.t = wp.from_numpy(data.t, dtype=wp.vec3f)

class TrainingCamera:
    camera_data: CameraData

    image_path: Path

    def __init__(self, camera_data: CameraData, image_path: Path):
        self.camera_data = camera_data
        self.image_path = image_path

    @property
    def image(self) -> wp.array(dtype=wp.float32):
        if not hasattr(self, "_image"):
            self._image = load_image(self.image_path)
        return self._image
