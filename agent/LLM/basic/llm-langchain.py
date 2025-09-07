from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv

load_dotenv(override=True)

client = ChatNVIDIA(
    # model="deepseek-ai/deepseek-r1-distill-llama-8b",
    model="meta/llama-3.3-70b-instruct",
    temperature=0.6,
    top_p=0.7,
    max_tokens=1024,
)


def extract_thinking_process():
    extracted = ""
    left = ""

    def inner(accessed_message: str):
        nonlocal extracted, left
        if "</think>" not in accessed_message:
            extracted = accessed_message[len("<think>") :]
        elif "</think>" in accessed_message:
            left = accessed_message[
                accessed_message.index("</think>") + len("</think>") :
            ]
        return f"Thinking: {extracted}\nOutput: {left}\n"

    return inner


message = ""
extractor = extract_thinking_process()


for chunk in client.stream([{"role": "user", "content": "鲁迅为什么暴打周树人？"}]):
    message += chunk.content
    print(extractor(message))
