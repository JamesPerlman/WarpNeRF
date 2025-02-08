from typing import Tuple
from warpnerf.server.objects.perspective_camera import PerspectiveCamera
from warpnerf.server.objects.scene import Scene
from warpnerf.server.protocols.dict_compatible import DictDeserializable

class RenderRequest(DictDeserializable):
    id: any
    scene: Scene
    camera: PerspectiveCamera
    size: Tuple[int, int]

    @classmethod
    def from_dict(self, data: dict):
        obj = self()

        obj.id = data['id']
        obj.scene = Scene.from_dict(data['scene'])
        obj.camera = PerspectiveCamera.from_dict(data['camera'])
        obj.size = (data['size'][0], data['size'][1])

        return obj
