import warp as wp

@wp.func
def vec3f_lte(a: wp.vec3f, b: wp.vec3f) -> wp.bool:
    return a[0] <= b[0] and a[1] <= b[1] and a[2] <= b[2]

@wp.func
def vec3f_gte(a: wp.vec3f, b: wp.vec3f) -> wp.bool:
    return a[0] >= b[0] and a[1] >= b[1] and a[2] >= b[2]

@wp.func
def vec3f_cwise_min(a: wp.vec3f, b: wp.vec3f) -> wp.vec3f:
    return wp.vec3f(wp.min(a[0], b[0]), wp.min(a[1], b[1]), wp.min(a[2], b[2]))

@wp.func
def vec3f_cwise_max(a: wp.vec3f, b: wp.vec3f) -> wp.vec3f:
    return wp.vec3f(wp.max(a[0], b[0]), wp.max(a[1], b[1]), wp.max(a[2], b[2]))
