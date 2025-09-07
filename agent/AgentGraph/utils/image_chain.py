from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv(override=True)


# structured output schema
class ImageRecognition(BaseModel):
    """I have labeled a bright numeric ID atthe center for each visual object in the image."
    "Please tell me the unique id for my requested itmes in this picture, "
    "also give me the unique id of the palce to put the item if asked."""

    object_number: str = Field(
        ..., description="the unique id for my requested items in this picture."
    )
    place_number: str = Field(
        ...,
        description="the unique id of the place to place my requested items in this picture",
    )


SYSTEM_PROMPT = """\
# introduction
I have labeled a bright numeric ID at the center for each visual object in the image.\
Please tell me the unique id for my requested items in this picture, 
also give me the unique id of the place to put the item if asked.
# example
## question
I have labeled a bright numeric ID at the center for each visual object in the image.\
Please tell me the IDs for: The curved cable, and i also want to put it on the straight cable, also tell me the corresponding id.
## your answer
the two corresponding numbers for my requested item and the place to place the item.
"""

# Image input format
IMAGE_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            [
                {"type": "text", "text": "{question}"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpg;base64, {encoded_image}"},
                },
            ],
        ),
    ]
)

# Chatting with image structured output
LLM_Structured_Chatter = ChatOpenAI(
    model="gpt-4o", temperature=0.6
).with_structured_output(ImageRecognition)

# image recognition chain
# input(dict): question & base64 encoded image
# output(dict): object_no(str) & place_no(str)
image_chain = (
    IMAGE_TEMPLATE
    | LLM_Structured_Chatter
    | {
        "object_number": RunnableLambda(lambda x: x.object_number),
        "place_number": RunnableLambda(lambda x: x.place_number),
    }
)

if __name__ == "__main__":
    import base64

    with open("/home/lwc/python_script/URgrasp_agent/UR/URgrasp_agent/agent/AgentGraph/annotated_image.png", "rb") as f:
        image = base64.b64encode(f.read()).decode()

    print(
        image_chain.invoke(
            {
                "question": "Please tell me the IDs for: The curved cable, and i also want to put it on the straight calbe, also tell me the corresponding id",
                "encoded_image": image,
            }
        )
    )
