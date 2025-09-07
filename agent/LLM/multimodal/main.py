# https://python.langchain.com/docs/how_to/multimodal_inputs
# https://platform.openai.com/docs/api-reference/introduction

from dotenv import load_dotenv
import base64

load_dotenv(override=True)

USE_OPENAI_OR_LANGCHAIN = True

with open("./man.jpg", "rb") as f:
    image = base64.b64encode(f.read()).decode()

MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's the picture about?"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpg;base64, {image}"},
            },
        ],
    }
]


if USE_OPENAI_OR_LANGCHAIN:
    from openai import OpenAI

    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1")

    completion = client.chat.completions.create(
        model="meta/llama-4-maverick-17b-128e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's the picture about?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpg;base64, {image}"},
                    }
                ],
            }
        ],
        temperature=0.6,
        top_p=0.7,
        max_tokens=1024,
        stream=True,
    )

    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
else:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    client = ChatNVIDIA(
        model="meta/llama-4-maverick-17b-128e-instruct",
        temperature=0.6,
        top_p=0.7,
        max_tokens=1024,
    )

    for chunk in client.stream(MESSAGES):
        print(chunk.content, end="")
