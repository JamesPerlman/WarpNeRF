from typing import List, Union, get_args
from warpnerf.server.objects.radiance_field import RadianceField
from warpnerf.server.protocols.dict_compatible import DictDeserializable

SceneObject = Union[RadianceField]

def obj_from_dict(data: dict) -> SceneObject:
    otype = data['type']

    for t in get_args(SceneObject):
        if t.type() == otype:
            return t.from_dict(data)

    return None

class Scene(DictDeserializable):
    objects: List[SceneObject]

    @classmethod
    def from_dict(cls, data: dict):
        obj = cls()
        
        obj.objects = [obj_from_dict(o) for o in data['objects']]
        obj.objects = [o for o in obj.objects if o is not None] # strip Nones

        return obj
