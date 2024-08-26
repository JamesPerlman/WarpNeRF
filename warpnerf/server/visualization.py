import math
import numpy as np
import warp as wp

from viser import ViserServer

from warpnerf.training.dataset import Dataset

class VisualizationServer:
    def __init__(self):
        self.viser = ViserServer()

    def set_dataset(self, dataset: Dataset):
        self.viser.scene.reset()

        images = [t.get_image() for t in dataset.training_cameras]
        dims = dataset.image_dims
        w, h = dims[0], dims[1]

        for i, training_cam in enumerate(dataset.training_cameras):
            image = images[i][::8, ::8, :]
            
            q = wp.quat_from_matrix(training_cam.camera_data.R)
            cam_rot = np.array([q[3], q[0], q[1], q[2]])
            cam_pos = np.array(training_cam.camera_data.t)
            fov = 2.0 * math.atan2(h / 2, training_cam.camera_data.f)
            self.viser.scene.add_camera_frustum(
                name=f"cameras/training_camera_{i}",
                fov=fov,
                aspect=w / h,
                image=image,
                format="png",
                wxyz=cam_rot,
                position=cam_pos
            )
    
    def set_background_image(self, image: np.ndarray):
        self.viser.scene.set_background_image(image)
