# import sys
# from pathlib import Path

# sys.path.append(str(Path(__file__).parent.parent))

from langgraph.prebuilt import create_react_agent
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import BaseMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv

load_dotenv(override=True)

client = MultiServerMCPClient(
    {
        "robot": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        },
    }
)

LLM_Chatter = ChatNVIDIA(
    model="meta/llama-3.3-70b-instruct",
    temperature=0.6,
    top_p=0.7,
    max_tokens=1024,
)

Agent_Prompt = (
    "* Background **\n"
    "Imagine we are working with a manipulator robot.\n"
    "This is a robotic arm with 6 degrees of freedom that has a gripper attached to its end effector\n"
    "** Your Goal **\n"
    "I would like you to assist me in sending commands to this robot given a scene and a task. \n"
    "I will give you two numbers, first one is the number of the item in the scene, and the second one is the number of the place you should place the item to.\n"
    "Your task is to move the given item(the item number) to the specific place(the place number).\n"
    "** Your ability **\n"
    "At any point, you have access to the following functions:\n"
    "You are allowed to create new functions using these, but you are not allowed to use any other hypothetical functions.\n"
    "* robot_pick(object_no): Given a number of an object, it moves the gripper to a given position of the object No.\n"
    "* robot_place(place_no): Given a number of a place, it moves the gripper to a given position of the place.\n"
    "Keep the solutions simple and clear. "
    "** Example **\n"
    "Here is an example scenario that illustrates how you can ask clarification questions.\n"
    "Let's assume a scene containing one bag and two plate.\n\n"
    "Me: pick up the bag and pick it to the left plate.\n"
    "You: (use the function **pick** to pick the bag and after you pick up the bag, inform me and ask for the next instruction.)"
    "i have picked the bag to {{XYZ position}}, what's next ?\n"
    "Me: put the bag to the plate.\n\n"
    "You: (use the function **place** to plcae the bag to the plate, and then inform me and ask for the next instruction.)"
    "i have place the bag to the plate in {{XYZ position}}, what's next ?\n"
    "Use python code to express your solution or output questions in {language}.\n\n"
    "** Creteria **\n"
    "You must operate step by step, for instance, you should first use **pick** function to pick the item, "
    "and then inform me that you have picked the item, next, i will ask you to place the item in the next turn, "
    "then you should use the **place** fucntion to place the item, "
    "after that, tell me you have placed the item in the pointed place."
)


async def reAct_agent():
    sub_graph_agent = create_react_agent(
        model=LLM_Chatter, tools=await client.get_tools(), prompt=Agent_Prompt
    )

    messages: list[BaseMessage] = [{"role": "user", "content": input("input>> ")}]
    while True:
        res = await sub_graph_agent.ainvoke({"messages": messages}, debug=True)
        messages = res["messages"]

        messages.append({"role": "user", "content": input("input>> ")})


if __name__ == "__main__":
    import asyncio

    asyncio.run((reAct_agent()))
