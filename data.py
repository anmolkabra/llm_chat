from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, model_serializer

import files


class ContentTextMessage(BaseModel):
    type: str = "text"
    text: str


class ContentImageMessage(BaseModel):
    type: str = "image"
    image: Image.Image

    class Config:
        # Allow arbitrary types so that Pydantic doesn't complain about Image.Image type
        arbitrary_types_allowed = True

    @model_serializer
    def serialize_model(self) -> dict[str, Any]:
        # Serialize the image to a base64 string
        image_base64 = files.pil_to_base64(self.image)
        return {"type": self.type, "image": image_base64}


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: list[ContentTextMessage | ContentImageMessage]
    created_at: datetime


class Conversation(BaseModel):
    messages: list[Message]

    @staticmethod
    def load_from_path(file_path: str) -> Conversation:
        """
        Load a conversation from a JSON file.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            Conversation: The conversation object.
        """
        with open(Path(file_path).expanduser(), "r") as f:
            data = json.load(f)

        # Loop through the messages and deserialize the content
        for i, message in enumerate(data["messages"]):
            for j, content in enumerate(message["content"]):
                if content["type"] == "image":
                    data["messages"][i]["content"][j]["image"] = files.base64_to_pil(content["image"])

        return Conversation(**data)

    def save_to_path(self, file_path: str) -> None:
        """
        Save the conversation to a JSON file.

        Args:
            file_path (str): Path to the JSON file.
        """
        with open(Path(file_path).expanduser(), "w") as f:
            f.write(self.model_dump_json(indent=2))
