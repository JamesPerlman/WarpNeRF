import logging
logging.basicConfig(level=logging.DEBUG)

import asyncio
import websockets
import msgpack

# WebSocket server URL
WS_URL = "ws://127.0.0.1:8765"  # Replace with your WebSocket server URL

# The message to send
message = {
    "topic": "load_dataset",
    "payload": "hello"
}

async def websocket_client():
    try:
        # Connect to the WebSocket server
        async with websockets.connect(WS_URL) as websocket:
            print("WebSocket connection opened.")

            # Serialize the message with msgpack
            packed_message = msgpack.packb(message)
            
            # Send the packed message
            await websocket.send(packed_message)
            print(f"Message sent: {message}")

            # Wait for a response from the server
            response = await websocket.recv()
            
            # Deserialize the response if it's in msgpack format
            try:
                unpacked_response = msgpack.unpackb(response)
                print(f"Received message: {unpacked_response}")
            except msgpack.exceptions.ExtraData:
                print(f"Received non-msgpack message: {response}")

    except Exception as e:
        print(f"Error: {e}")

# Run the client
asyncio.run(websocket_client())
