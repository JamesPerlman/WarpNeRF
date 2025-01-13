import asyncio
from pathlib import Path
import msgpack
import torch
import websockets
from typing import Callable, Type

from warpnerf.models.dataset import Dataset, DatasetType
from warpnerf.models.warpnerf_model import WarpNeRFModel
from warpnerf.training.trainer import Trainer

class WarpNeRFServer:
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.handlers = {}
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

    def recv(self, topic: str, payload_type: Type):
        """Decorator to register a handler for a specific topic."""
        def decorator(func: Callable):
            async def wrapper(payload):
                try:
                    payload = payload_type(**payload) if payload_type else payload
                    await func(payload)
                except Exception as e:
                    print(f"Error handling topic {topic}: {e}")
            self.handlers[topic] = wrapper
            return func

        return decorator

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

    async def process_queue(self):
        """Process messages from the queue serially."""
        while True:
            # Always await for the next task in the queue
            topic, payload = await self.queue.get()
            try:
                if topic in self.handlers:
                    await self.handlers[topic](payload)
                else:
                    print(f"No handler registered for topic: {topic}")
            except Exception as e:
                print(f"Error processing topic {topic}: {e}")
            finally:
                self.queue.task_done()

    def register_handlers(self):
        """Register all message handlers."""

        @self.recv("load_dataset", dict)
        async def on_load_dataset(payload):
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

    async def start(self):
        print(f"Starting WarpNeRF server at {self.host}:{self.port}")
        await asyncio.gather(
            websockets.serve(self.websocket_handler, self.host, self.port),
            self.process_queue()
        )


def run_warpnerf_server():
    server = WarpNeRFServer()
    asyncio.run(server.start())
