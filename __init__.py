import time
import warp as wp
import numpy as np
import torch


from pathlib import Path

from warpnerf.models.batch import RayBatch, create_ray_batch
from warpnerf.models.trimiprf_model import TrimipRFModel
from warpnerf.server.visualization import VisualizationServer
from warpnerf.training.batcher import init_training_rays_and_pixels_kernel
from warpnerf.models.dataset import Dataset
from warpnerf.training.trainer import Trainer
from warpnerf.utils.image import save_image

wp.init()

dataset = Dataset(path=Path("/home/luks/james/nerfs/turb-small"))
dataset.load()

model = TrimipRFModel()
optimizer = torch.optim.Adam(model.parameters())
trainer = Trainer(dataset, model, optimizer)

for i in range(1000):
    trainer.step()

wp.synchronize()

# server = VisualizationServer()
# server.set_dataset(dataset)

# while True:
#     time.sleep(0.01)
