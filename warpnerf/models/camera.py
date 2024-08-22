import numpy as np
import warp as wp

from pathlib import Path
from warpnerf.utils.bundler_sfm import BundlerSFMCameraData
from warpnerf.utils.image import get_image_dims, load_image

@wp.struct
class CameraData:
    f: wp.float32
    k1: wp.float32
    k2: wp.float32
    R: wp.mat33f
    t: wp.vec3f

def create_camera_data_from_bundler(data: BundlerSFMCameraData) -> CameraData:
    cam = CameraData()
    cam.f = data.f
    cam.k1 = data.k1
    cam.k2 = data.k2
    R = wp.transpose(wp.mat33f(data.R))
    t = wp.vec3f(data.t)
    # adjustment matrix to convert from bundler to warpnerf coordinate system
    M = wp.mat33f(
        [[1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]]
    )
    cam.R = wp.mul(R, M)
    cam.t = wp.mul(wp.neg(R), t)
    return cam

class TrainingCamera:

    camera_data: CameraData
    image_path: Path

    def __init__(self, camera_data: CameraData, image_path: Path):
        self.camera_data = camera_data
        self.image_path = image_path

    def get_image(self) -> wp.array3d(dtype=wp.uint8):
        return  load_image(self.image_path)

    def get_image_dims(self) -> tuple[int, int]:
        return get_image_dims(self.image_path)
