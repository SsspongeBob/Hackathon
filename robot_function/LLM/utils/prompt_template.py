from langchain_core.prompts import ChatPromptTemplate

# Image input format
IMAGE_TEMPLATE = ChatPromptTemplate.from_messages(
    [
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


def generate_image_prompt(question: str, encoded_image: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpg;base64, {encoded_image}"},
                },
            ],
        }
    ]


if __name__ == "__main__":

    print(IMAGE_TEMPLATE.format_messages(question="Hello", encoded_image="World"))
