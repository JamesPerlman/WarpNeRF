import warp as wp
from warpnerf.training.mlp import MLP

class Trainer:

    mlp: MLP

    def __init__(self, mlp: MLP):
        self.mlp = mlp

    def loss(self, batch: wp.array2d(dtype=float), target: wp.array2d(dtype=float)):
        wp.launch(
            self.mlp.forward,
            dim=1024,
            inputs=[batch, target],
            device="cuda"
        )
