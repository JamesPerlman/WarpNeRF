from typing import Tuple
import numpy as np
from warpnerf.server.objects.object_identifiers import WN_OTYPE_RADIANCE_FIELD
from warpnerf.server.protocols.dict_compatible import DictCompatible

class RadianceField(DictCompatible):
    
    id: int = 0
    rf_type: str = 'nerf'
    bbox_size: float = 1.0
    transform: np.ndarray = np.eye(4, dtype=np.float32)
    is_trainable: bool = True
    is_training_enabled: bool = False
    limit_training: bool = True
    n_steps_max: int = 10000
    n_steps_trained: int = 0
    n_images_loaded: int = 0
    n_images_total: int = 0

    @classmethod
    def type(cls) -> str:
        return WN_OTYPE_RADIANCE_FIELD
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'type': self.type,
            'rf_type': self.rf_type,
            'bbox_size': self.bbox_size,
            'transform': self.transform.tolist(),
            'is_trainable': self.is_trainable,
            'is_training_enabled': self.is_training_enabled,
            'limit_training': self.limit_training,
            'n_steps_max': self.n_steps_max,
            'n_steps_trained': self.n_steps_trained,
            'n_images_loaded': self.n_images_loaded,
            'n_images_total': self.n_images_total
        }

    @classmethod
    def from_dict(cls, data: dict):
        obj = cls()

        obj.id = data['id']
        obj.rf_type = data['rf_type']
        obj.bbox_size = data['bbox_size']
        obj.transform = np.array(data['transform'], dtype=np.float32)
        obj.is_trainable = data['is_trainable']
        obj.is_training_enabled = data['is_training_enabled']
        obj.limit_training = data['limit_training']
        obj.n_steps_max = data['n_steps_max']
        obj.n_steps_trained = data['n_steps_trained']
        obj.n_images_loaded = data['n_images_loaded']
        obj.n_images_total = data['n_images_total']

        return obj