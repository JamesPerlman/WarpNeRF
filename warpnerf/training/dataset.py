from pathlib import Path
from typing import List

from warpnerf.models.camera import TrainingCamera, create_camera_data_from_bundler
from warpnerf.utils.bundler_sfm import BundlerSFMData, create_bundler_sfm_data_from_path

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

    def load(self):

        if self.is_loaded:
            return

        self.bundler_data = create_bundler_sfm_data_from_path(self.path / "registration.out", read_points=False)
        self.image_paths = get_image_paths_from_lst(self.path / "images.lst")

        assert(len(self.bundler_data.cameras) == len(self.image_paths), "Number of cameras and images do not match")

        self.training_cameras = []
        for bundler_camera_data, image_path in zip(self.bundler_data.cameras, self.image_paths):
            camera_data = create_camera_data_from_bundler(bundler_camera_data)
            training_camera = TrainingCamera(camera_data, image_path)
            self.training_cameras.append(training_camera)

    