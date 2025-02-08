import asyncio
from collections import defaultdict
from pathlib import Path
import msgpack
import numpy as np
import torch
import websockets
from typing import Callable, Type

from warpnerf.models.camera import TrainingCamera, create_camera_data_from_perspective_camera
from warpnerf.models.dataset import Dataset, DatasetType
from warpnerf.models.warpnerf_model import WarpNeRFModel
from warpnerf.server.objects.radiance_field import RadianceField
from warpnerf.server.objects.render_request import RenderRequest
from warpnerf.server.objects.render_result import RenderResult
from warpnerf.training.trainer import Trainer
from warpnerf.utils.rendering import get_rendered_image

class WarpNeRFServer:
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.subscriptions = defaultdict(list)
        self.queue = asyncio.Queue()  # Single queue for all requests
        self.connections = set()  # Initialize connections set

        # Single NeRF for now, needs to be revised to allow multiple NeRFs
        self.dataset = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.trainer: Trainer = None
        self.is_training = False
        self.training_step = 0

        # Register message handlers
        self.register_handlers()
    
    def __del__(self):

        # close all connections
        for connection in self.connections:
            connection.close()
        
        self.connections = set()

        # clear all subscriptions
        self.subscriptions = defaultdict(list)

    def subscribe(self, topic: str, handler: Callable) -> Callable:
        """Subscribe to a topic."""
        self.subscriptions[topic].append(handler)

    async def send(self, topic: str, payload: dict):
        """Broadcast a message to all connected clients."""
        message = msgpack.packb({"topic": topic, "payload": payload})
        to_remove = set()
        for websocket in self.connections:
            try:
                await websocket.send(message)
            except websockets.ConnectionClosed:
                print(f"Connection closed: removing {websocket}")
                to_remove.add(websocket)
        self.connections -= to_remove

    async def websocket_handler(self, websocket, path):
        print(f"Connection established: {path}")
        self.connections.add(websocket)
        try:
            async for message in websocket:
                try:
                    # Decode the message from msgpack
                    data = msgpack.unpackb(message)
                    topic = data.get("topic")
                    payload = data.get("payload")

                    print(f"Received topic: {topic}, payload: {payload}")

                    # Enqueue the message for processing
                    await self.queue.put((topic, payload))

                except Exception as e:
                    print(f"Error decoding message: {e}")
        except websockets.ConnectionClosed as e:
            print(f"Connection closed: {e}")
        finally:
            self.connections.remove(websocket)

    async def process_websocket_queue(self):
        """Process messages from the queue serially."""
        while True:
            # Always await for the next task in the queue
            topic, payload = await self.queue.get()
            try:
                if topic in self.subscriptions:
                    for handler in self.subscriptions[topic]:
                        await handler(payload)
                else:
                    print(f"No handler registered for topic: {topic}")
            except Exception as e:
                print(f"Error processing topic {topic}: {e}")
            finally:
                self.queue.task_done()

    async def async_runloop(self):
        loop = asyncio.get_running_loop()
        while True:
            if self.is_training:
                await loop.run_in_executor(None, self.trainer.step)
                await loop.run_in_executor(None, self.scheduler.step)
                await self.send("training_step", {"step": self.training_step})
                self.training_step += 1

            else:
                await asyncio.sleep(1.0)

    def register_handlers(self):
        """Register all message handlers."""
        self.subscribe("load_dataset", self.on_load_dataset)
        self.subscribe("request_render", self.on_request_render)
    
    async def on_load_dataset(self, payload):
        print(f"Received payload: {payload}")
        dataset_path = Path(payload["path"])
        self.dataset = Dataset(path=dataset_path, type=DatasetType.TRANSFORMS_JSON)
        self.dataset.load()
        self.dataset.resize_and_center(aabb_scale=8.0)
        self.model = WarpNeRFModel(
            aabb_scale=self.dataset.aabb_scale * 2,
            n_appearance_embeddings=self.dataset.num_images
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9995)
        self.trainer = Trainer(
            dataset=self.dataset,
            model=self.model,
            optimizer=self.optimizer
        )
        self.is_training = True
        self.training_step = 0
        rf_data = {
            "id": 0,
            "rf_type": "warpnerf",
            "bbox_size": self.dataset.aabb_scale,
            "transform": np.eye(4).tolist(),
            "is_trainable": True,
            "is_training_enabled": True,
            "limit_training": False,
            "n_steps_max": 10000,
            "n_steps_trained": 0,
            "n_images_loaded": self.dataset.num_images,
            "n_images_total": self.dataset.num_images,
        }
        await self.send("add_radiance_field", rf_data)
        print(f"Adding radiance field", rf_data)
    
    async def on_request_render(self, payload):
        print(f"Received payload: {payload}")
        render_request = RenderRequest.from_dict(payload)
        print(render_request)
        # for now it just renders the only radiance field
        cam_data = create_camera_data_from_perspective_camera(render_request.camera)
        training_cam = TrainingCamera(cam_data, None, render_request.camera.image_dims)
        tcam0 = self.dataset.training_cameras[0]
        img = get_rendered_image(self.model, training_cam, *render_request.size)
        render_result = RenderResult(request_id=render_request.id, image=img)
        await self.send("render_result", render_result.to_dict())

    async def start(self):
        print(f"Starting WarpNeRF server at {self.host}:{self.port}")
        await asyncio.gather(
            websockets.serve(self.websocket_handler, self.host, self.port),
            self.process_websocket_queue(),
            self.async_runloop()
        )


def run_warpnerf_server():
    server = WarpNeRFServer()
    asyncio.run(server.start())
