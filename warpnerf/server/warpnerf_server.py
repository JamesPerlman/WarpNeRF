import asyncio
import websockets

class WarpNeRFServer:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port

    async def handler(self, websocket, path):
        print(f"Connection established: {path}")
        try:
            async for message in websocket:
                print(f"Received: {message}")
                # Echo back the message
                await websocket.send(f"Echo: {message}")
        except websockets.ConnectionClosed as e:
            print(f"Connection closed: {e}")

    def start(self):
        print(f"Starting WarpNeRFServer at {self.host}:{self.port}")
        start_server = websockets.serve(self.handler, self.host, self.port)

        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

def run_warpnerf_server():
    server = WarpNeRFServer()
    server.start()
