from dataclasses import dataclass
from io import BytesIO
from PIL import Image

from warpnerf.server.protocols.dict_compatible import DictSerializable

@dataclass
class RenderResult(DictSerializable):
    request_id: int
    image: Image.Image

    def to_dict(self) -> dict:
        with BytesIO() as output:
            self.image.save(output, format="PNG")
            image_bytes = output.getvalue()
        
        return {
            "request_id": self.request_id,
            "image": image_bytes
        }
