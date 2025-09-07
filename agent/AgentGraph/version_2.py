from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.chat_agent_executor import create_react_agent, AgentState
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from utils.image_chain import image_chain
from typing import TypedDict, Annotated, Sequence, Literal
from dotenv import load_dotenv

load_dotenv(override=True)

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


# Graph state
class RobotState(TypedDict):
    """The state of the robot"""

    question: str

    encoded_image: str

    object_number: int

    place_number: int

    image_task_done: bool
    """if processed by agent, then set to True"""

    task_done: bool
    """if task done, then set to True"""

    messages: Annotated[Sequence[BaseMessage], add_messages]


class RobotOutputState(TypedDict):
    """The output state of the robot"""

    object_number: int
    place_number: int
    messages: Annotated[Sequence[BaseMessage], add_messages]


# MCP client
client = MultiServerMCPClient(
    {
        "robot": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        },
    }
)


# Image Recognition Node
def image_recognition_node(
        state: RobotState, config: RunnableConfig, store: BaseStore
) -> RobotState:
    res = image_chain.invoke(
        {"question": state["question"], "encoded_image": state["encoded_image"]}
    )
    print("Image Recognition Finished!")
    return {"object_number": res["object_number"], "place_number": res["place_number"]}


# human process node, for forward integration
def human_node(state: RobotState) -> RobotState:
    if state["image_task_done"]:
        prompt = input("Question: ")
        if prompt == "Done":
            return {"task_done": True}
        return {"messages": [HumanMessage(content=prompt)]}
    else:
        prompt = f"图像已经处理完成了，代表物体的数字为{state['object_number']}， 代表放置物体的地方的数字为：{state['place_number']}"
        return {"messages": [HumanMessage(content=prompt)]}


# agent node
async def agent_node(state: RobotState) -> RobotState:
    node = create_react_agent(
        model=ChatNVIDIA(
            model="meta/llama-3.3-70b-instruct",
            temperature=0.6,
            top_p=0.7,
            max_tokens=1024,
        ),
        tools=await client.get_tools(),
        prompt=Agent_Prompt,
    )
    res = await node.ainvoke(state, debug=True)
    return {"messages": res["messages"], "image_task_done": True}


def route_to_agent(state: RobotState) -> Literal["agent_node", "__end__"]:
    if state["task_done"]:
        return END
    else:
        return "agent_node"


# Build workflow
graph = StateGraph(state_schema=RobotState, output=RobotOutputState)

# Add nodes
graph.add_node("image_recognition_node", image_recognition_node)
graph.add_node("human_node", human_node)
graph.add_node("agent_node", agent_node)

# Add edges to connect nodes
graph.add_edge(START, "image_recognition_node")
graph.add_edge("image_recognition_node", "human_node")
graph.add_conditional_edges("human_node", route_to_agent)
graph.add_edge("agent_node", "human_node")

robot_agent = graph.compile()

if __name__ == "__main__":
    import asyncio
    import base64


    async def main():
        async with client.session(server_name="robot") as session:
            print("Task started.")
            # encoded_image = (await session.call_tool("base64image")).content[0].text
            with open("/home/lwc/python_script/URgrasp_agent/UR/URgrasp_agent/agent/AgentGraph/annotated_image.png",
                      "rb") as f:
                encoded_image = base64.b64encode(f.read()).decode()
            print(encoded_image)
            print("Encode image generated!")
            async for chunk in robot_agent.astream(
                    {
                        "question": input("Question: "),
                        # question: 请把绿色的零食放在蓝色的盘子里，给出相对应的两个数字
                        "encoded_image": encoded_image,
                        "messages": [],
                        "image_task_done": False,
                        "task_done": False,
                    },
                    stream_mode="updates",
                    subgraphs=True
            ):
                print(chunk)


    asyncio.run(main())
