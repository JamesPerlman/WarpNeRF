import warp as wp
import numpy as np

from pathlib import Path

# Set the cache directory
wp.config.kernel_cache_dir = Path(__file__).parent.parent / '.warpcache'
wp.init()

@wp.func
def fn(x: float):
    return x * x

@wp.func_grad(fn)
def adj_fn(x: float, adj_ret: float):
    if x > 0.0:
        wp.adjoint[x] += 2.0 * x * adj_ret

@wp.kernel
def run_fn(xs: wp.array(dtype=float), output: wp.array(dtype=float)):
    i = wp.tid()
    output[i] = fn(xs[i])

@wp.kernel
def test_add(
    counter: wp.array(dtype=int),
    input: wp.array(dtype=float),
    output: wp.array(dtype=float)
):
    i = wp.atomic_add(counter, 1)
    output[i] = wp.sqrt(input[i])

xs = wp.array([0.0, 1.0, 2.0, 3.0, 0.0], dtype=wp.float32, requires_grad=True)
ys = wp.zeros_like(xs)

tape = wp.Tape()
with tape:
    wp.launch(run_fn, dim=len(xs), inputs=[xs], outputs=[ys])

tape.backward(grads={ys: wp.ones_like(ys)})

print("xs: ", xs)
print("ys: ", ys)
print("xs.grad: ", xs.grad)