import warp as wp

from warpnerf.utils.math import vec3f_gte, vec3f_lte

@wp.struct
class BoundingBox:
    min: wp.vec3
    max: wp.vec3

def create_bounding_box(min: wp.vec3, max: wp.vec3) -> BoundingBox:
    return BoundingBox(min=min, max=max)

@wp.func
def bbox_contains_point(bbox: BoundingBox, point: wp.vec3) -> wp.bool:
    return vec3f_lte(bbox.min, point) and vec3f_gte(bbox.max, point)
