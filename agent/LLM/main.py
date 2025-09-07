from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from agent.LLM.utils.model import Joke, ImageRecognition
from agent.LLM.utils.prompt_template import generate_image_prompt, IMAGE_TEMPLATE

load_dotenv(override=True)

# Chatting
LLM_Chatter = ChatNVIDIA(
    model="meta/llama-3.3-70b-instruct",
    temperature=0.6,
    top_p=0.7,
    max_tokens=1024,
)

# Chatting with image structured output
LLM_Structured_Chatter = ChatOpenAI(
    model="gpt-4o", temperature=0.6
).with_structured_output(ImageRecognition)

# Reasoning
# LLM_Reasoner = ChatNVIDIA(
#     model="deepseek-ai/deepseek-r1-distill-llama-8b",
#     temperature=0.6,
#     top_p=0.7,
#     max_tokens=1024,
# )


# LLM Singleton
# class LLMFactory:
#     _instance = None

#     def __new__(cls, *args, **kwargs):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#         return cls._instance

#     def __init__(self, name: str = None):
#         pass


if __name__ == "__main__":
    import base64

    with open("./pictures/cable.png", "rb") as f:
        image = base64.b64encode(f.read()).decode()

    image_chain = IMAGE_TEMPLATE | LLM_Structured_Chatter

    print(
        image_chain.invoke(
            {
                "question": "I have labeled a bright numeric ID atthe center for each visual object in the image.Please tell me the IDs for: The curved cable.",
                "encoded_image": image,
            }
        )
    )
