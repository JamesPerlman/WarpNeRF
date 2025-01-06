import asyncio
import msgpack
import websockets
from typing import Callable, Type

class WarpNeRFServer:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.handlers = {}
        self.queue = asyncio.Queue()  # Single queue for all requests
        self.connections = set()  # Initialize connections set

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

    async def handler(self, websocket, path):
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
            topic, payload = await self.queue.get()
            if topic in self.handlers:
                await self.handlers[topic](payload)
            else:
                print(f"No handler registered for topic: {topic}")

    async def start(self):
        print(f"Starting WarpNeRF server at {self.host}:{self.port}")
        await websockets.serve(self.handler, self.host, self.port)

def run_warpnerf_server():
    server = WarpNeRFServer()

    @server.recv("load_dataset", dict)
    async def on_load_dataset(payload):
        print(f"Received payload: {payload}")
        await server.send("some_response", {"topic": "some_response", "payload": "test"})

    server.start()

    print("hello")
