from PIL import Image
from pydantic import BaseModel


class ContentTextMessage(BaseModel):
    type: str = "text"
    text: str


class ContentImageMessage(BaseModel):
    type: str = "image"
    image: Image.Image

    class Config:
        # Allow arbitrary types so that Pydantic doesn't complain about Image.Image type
        arbitrary_types_allowed = True


class Message(BaseModel):
    role: str
    content: list[ContentTextMessage | ContentImageMessage]


class Conversation(BaseModel):
    messages: list[Message]
