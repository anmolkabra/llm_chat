import base64
import io

from PIL import Image


def pil_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def base64_to_pil(base64_image: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(base64_image)))
