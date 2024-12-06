import warp as wp

@wp.func
def sigma_to_alpha(sigma: wp.float32, dt: wp.float32) -> wp.float32:
    return 1.0 - wp.exp(-sigma * dt)
