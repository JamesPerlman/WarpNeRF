import warp as wp
import numpy as np

from warpnerf.training.mlp import MLP, mlp_kernel
from warpnerf.training.trainer import Trainer

from pathlib import Path

wp.init()

mlp = MLP(input_dim=3, hidden_dim=8, n_hidden_layers=3, output_dim=3)
mlp.init_params()

xs = wp.array([0.0, 1.0, 2.0, 1.0, 3.0, 0.0], dtype=wp.float32, requires_grad=True)
ys = wp.zeros_like(xs)
targets = wp.array([1.0, 2.0, 3.0, 2.0, 4.0, 1.0], dtype=wp.float32)

loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

@wp.kernel
def mse_loss_kernel(
    pred: wp.array(dtype=float),
    targ: wp.array(dtype=float),
    scale: float,
    loss: wp.array(dtype=float)
):
    i = wp.tid()
    diff = pred[i] - targ[i]
    wp.atomic_add(loss, 0, scale * diff * diff)

tape = wp.Tape()
with tape:
    wp.launch(mlp_kernel, dim=6, inputs=[mlp.weights, mlp.biases, xs, ys], device="cuda")
    wp.launch(mse_loss_kernel, dim=6, inputs=[ys, targets, 1.0 / len(ys), loss], device="cuda")

tape.backward(loss)
wp.synchronize()
print("xs: ", xs)
print("ys: ", ys)
print("xs.grad: ", ys.grad)
print("loss: ", loss)