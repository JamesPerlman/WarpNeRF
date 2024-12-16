import math
import numpy as np
import warp as wp

from viser import ViserServer

from warpnerf.models.camera import CameraData, TrainingCamera
from warpnerf.models.dataset import Dataset

class VisualizationServer:
    def __init__(self):
        self.viser = ViserServer()

    def set_dataset(self, dataset: Dataset):
        self.viser.scene.reset()

        images = [t.get_image() for t in dataset.training_cameras]

        for i, training_cam in enumerate(dataset.training_cameras):
            image = images[i][::16, ::16, :]
            w = training_cam.camera_data.sx
            h = training_cam.camera_data.sy
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
    
    def get_viewport_camera(self, aabb_scale: float = 1.0) -> TrainingCamera:
        clients = self.viser.get_clients()
        if len(clients) == 0:
            return None
        client = clients[0]
        viser_cam = client.camera
        T = viser_cam.position
        q = viser_cam.wxyz
        R = wp.quat_to_matrix(wp.quat([q[1], q[2], q[3], q[0]]))
        cam_data = CameraData()
        fov = viser_cam.fov
        focal_len = 0.5 / math.tan(fov / 2)
        H = 1024
        W = H * viser_cam.aspect
        cam_data.f = focal_len
        cam_data.sx = viser_cam.aspect
        cam_data.sy = 1.0
        cam_data.cx = 0.5 * cam_data.sx
        cam_data.cy = 0.5 * cam_data.sy
        cam_data.t = T
        cam_data.R = R

        train_cam = TrainingCamera(camera_data=cam_data, image_path=None, image_dims=(int(W), int(H)))

        return train_cam
