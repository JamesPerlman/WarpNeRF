import torch
import warp as wp

# adapted from page 4 of https://arxiv.org/pdf/2302.12249
# transforms (-∞, ∞) to (-2, 2), preserving the range [-1, 1] along each axis

@wp.func
def apply_merf_contraction(xyz: wp.vec3f) -> wp.vec3f:
    max_extent_xyz = wp.max(xyz)

    if max_extent_xyz <= 1.0:
        return xyz
    
    x, y, z = xyz.x, xyz.y, xyz.z

    if x != max_extent_xyz:
        x = x / max_extent_xyz
    else:
        x = (2.0 - 1.0 / wp.abs(x)) * wp.sign(x)

    if y != max_extent_xyz:
        y = y / max_extent_xyz
    else:
        y = (2.0 - 1.0 / wp.abs(y)) * wp.sign(y)

    if z != max_extent_xyz:
        z = z / max_extent_xyz
    else:
        z = (2.0 - 1.0 / wp.abs(z)) * wp.sign(z)
        
    return wp.vec3f(x, y, z)


# this one normalizes the points after applying the MERF contraction
@wp.kernel
def apply_merf_contraction_kernel(
    xyz: wp.array1d(dtype=wp.vec3f),
    out: wp.array1d(dtype=wp.vec3f)
):
    i = wp.tid()

    contracted_xyz = apply_merf_contraction(xyz[i])

    # normalize the contracted point to (-1, 1)
    out[i] = 0.5 * contracted_xyz


class MERFContraction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz):
        ctx.xyz = wp.from_torch(xyz, dtype=wp.vec3f)
        ctx.out = wp.empty(
            shape=(xyz.shape[0]),
            dtype=wp.vec3f,
            device=ctx.xyz.device,
            requires_grad=True
        )

        wp.launch(
            kernel=apply_merf_contraction_kernel,
            dim=ctx.xyz.shape[0],
            inputs=[ctx.xyz],
            outputs=[ctx.out]
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
