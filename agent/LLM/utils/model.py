from pydantic import BaseModel, Field
from typing import Optional


class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(..., description="The setup of the joke")
    punchline: str = Field(..., description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )


class ImageRecognition(BaseModel):
    """I have labeled a bright numeric ID atthe center for each visual object in the image."
    "Please tell me the unique id for my requested itmes in this picture."""

    id: int = Field(
        ..., description="the unique id for my requested itmes in this picture."
    )
