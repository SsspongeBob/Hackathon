from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI(base_url="https://integrate.api.nvidia.com/v1")

completion = client.chat.completions.create(
    model="meta/llama-3.3-70b-instruct",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.6,
    top_p=0.7,
    max_tokens=1024,
    stream=True,
)

for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
