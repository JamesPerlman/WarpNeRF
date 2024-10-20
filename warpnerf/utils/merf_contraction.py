import torch
import warp as wp

from warpnerf.models.bounding_box import BoundingBox

# adapted from page 4 of https://arxiv.org/pdf/2302.12249
# transforms (-∞, ∞) to (-2, 2), preserving the range [-1, 1] along each axis

@wp.func
def apply_merf_contraction(xyz: wp.vec3f) -> wp.vec3f:
    max_extent_xyz = wp.max(xyz)

    if max_extent_xyz <= 1.0:
        return xyz
    
    res: wp.vec3f = xyz

    if xyz.x != max_extent_xyz:
        res.x = xyz.x / max_extent_xyz
    else:
        res.x = (2.0 - 1.0 / wp.abs(xyz.x)) * wp.sign(xyz.x)

    if xyz.y != max_extent_xyz:
        res.y = xyz.y / max_extent_xyz
    else:
        res.y = (2.0 - 1.0 / wp.abs(xyz.y)) * wp.sign(xyz.y)

    if xyz.z != max_extent_xyz:
        res.z = xyz.z / max_extent_xyz
    else:
        res.z = (2.0 - 1.0 / wp.abs(xyz.z)) * wp.sign(xyz.z)
        
    return res


# this one normalizes the points after applying the MERF contraction
@wp.kernel
def apply_merf_contraction_kernel(
    xyz: wp.array1d(dtype=wp.vec3f),
    out: wp.array1d(dtype=wp.vec3f)
) -> None:
    i = wp.tid()
    out[i] = wp.cw_mul(0.5, apply_merf_contraction(xyz[i]))


class MERFContractionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz):
        ctx.xyz = wp.from_torch(xyz)
        ctx.out = wp.empty(
            shape=xyz.shape,
            dtype=wp.vec3f,
            device=ctx.xyz.device,
            requires_grad=True
        )
        
        wp.launch(
            kernel=apply_merf_contraction_kernel,
            dim=xyz.shape[0],
            inputs=[ctx.xyz],
            outputs=[ctx.out],
        )

        return wp.to_torch(ctx.out)

    @staticmethod
    def backward(ctx, adj_out):
        ctx.out.grad = wp.from_torch(adj_out)
        
        wp.launch(
            kernel=apply_merf_contraction_kernel,
            dim=ctx.xyz.shape[0],
            inputs=[ctx.xyz],
            outputs=[ctx.out],
            adj_inputs=[ctx.xyz.grad],
            adj_outputs=[ctx.out.grad],
            adjoint=True,
        )

        return wp.to_torch(ctx.out.grad)
