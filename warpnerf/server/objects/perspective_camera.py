import numpy as np

from typing import Tuple

from warpnerf.server.objects.object_identifiers import WN_OTYPE_PERSPECTIVE_CAMERA
from warpnerf.server.protocols.dict_compatible import DictCompatible

class PerspectiveCamera(DictCompatible):

    focal_length: float
    image_dims: Tuple[int, int]
    sensor_dims: Tuple[float, float]
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray

    @classmethod
    def type(cls) -> str:
        return WN_OTYPE_PERSPECTIVE_CAMERA

    def to_dict(self) -> dict:
        return {
            'f': self.focal_length,
            'img_w': self.image_dims[0],
            'img_h': self.image_dims[1],
            'sx': self.sensor_dims[0],
            'sy': self.sensor_dims[1],
            'R': self.rotation_matrix.tolist(),
            't': self.translation_vector.tolist()
        }

    @classmethod
    def from_dict(cls, data: dict):
        obj = cls()

        obj.focal_length = data['f']
        obj.image_dims = (data['img_w'], data['img_h'])
        obj.sensor_dims = (data['sx'], data['sy'])
        obj.rotation_matrix = np.array(data['R'], dtype=np.float32)
        obj.translation_vector = np.array(data['t'], dtype=np.float32)

        return obj
