import torch
import warp as wp

@wp.func
def encode_sh_deg4(dir: wp.vec3f, res: wp.array1d(dtype=wp.float32)):

    x, y, z = dir[0], dir[1], dir[2]
    xy, xz, yz = x * y, x * z, y * z
    x2, y2, z2 = x**2.0, y**2.0, z**2.0
    
    res[0] = 0.2820947917738781
    res[1] = -0.4886025119029199 * y
    res[2] = 0.4886025119029199 * z
    res[3] = -0.4886025119029199 * x
    res[4] = 1.0925484305920792 * xy
    res[5] = -1.0925484305920792 * yz
    res[6] = 0.31539156525252005 * (3.0 * z2 - 1.0)
    res[7] = -1.0925484305920792 * xz
    res[8] = 0.5462742152960396 * (x2 - y2)
    res[9] = 0.5462742152960396 * x * (x2 - 3.0 * y2)
    res[10] = -0.5462742152960396 * y * (3.0 * x2 - y2)
    res[11] = 0.5462742152960396 * z * (2.0 * z2 - 3.0 * x2 - 3.0 * y2)
    res[12] = 0.5900435899266435 * x * (x2 - 3.0 * y2) * z
    res[13] = -0.5900435899266435 * y * (3.0 * x2 - y2) * z
    res[14] = 1.445305721320277 * x * y * (x2 - y2)
    res[15] = -0.5900435899266435 * (x2 - y2) * (3.0 * z2 - 1.0)

@wp.kernel
def encode_sh_deg4_kernel(
    dirs: wp.array1d(dtype=wp.vec3f),
    sh_out: wp.array2d(dtype=wp.float32)
):
    i = wp.tid()

    encode_sh_deg4(dirs[i], sh_out[i])

class SHDeg4Encode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dirs):
        ctx.dirs = wp.from_torch(dirs)
        ctx.sh = wp.zeros((dirs.shape[0], 16), dtype=wp.float32, device="cuda", requires_grad=True)
        
        wp.launch(
            kernel=encode_sh_deg4_kernel,
            dim=dirs.shape[0],
            inputs=[ctx.dirs],
            outputs=[ctx.sh],
        )

        return wp.to_torch(ctx.sh)

    @staticmethod
    def backward(ctx, adj_sh):
        ctx.sh.grad = wp.from_torch(adj_sh)

        wp.launch(
            kernel=encode_sh_deg4_kernel,
            dim=ctx.dirs.shape[0],
            inputs=[ctx.dirs],
            outputs=[ctx.sh],
            adj_inputs=[ctx.dirs.grad],
            adj_outputs=[ctx.sh.grad],
            adjoint=True,
        )

        return wp.to_torch(ctx.sh.grad)

class SHDeg4Encoding(torch.nn.Module):
    def forward(self, dirs):
        return SHDeg4Encode.apply(dirs)
    
    @property
    def input_dim(self):
        return 3

    @property
    def output_dim(self):
        return 16
