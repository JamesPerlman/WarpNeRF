import warp as wp

@wp.struct
class OccupancyGrid:
    n_levels: wp.int32
    resolution: wp.int32
    volume: wp.int32

    density: wp.array3d(dtype=wp.float32)
    bitfield: wp.array3d(dtype=wp.uint8)


def create_occupancy_grid(n_levels: int, resolution: int) -> OccupancyGrid:
    volume = n_levels * (resolution ** 3)
    return OccupancyGrid(
        n_levels=n_levels,
        resolution=resolution,
        volume=volume,
        density=wp.zeros((volume, 1), dtype=wp.float32),
        bitfield=wp.zeros((volume, 1), dtype=wp.uint8)
    )
