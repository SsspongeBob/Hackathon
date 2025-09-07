from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage
from agent.AgentGraph.utils.image_chain import image_chain
from typing import TypedDict, Union, Annotated, Sequence
from robot_function.operator import Operator

# operator initialization
operator = Operator()
operator.display()
operator.init_sam()


# Graph state
class RobotState(AgentState):
    """The state of the robot"""

    question: str
    encoded_image: str
    object_number: int
    place_number: int


class RobotOutputState(TypedDict):
    """The output state of the robot"""

    object_number: int
    place_number: int
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Image Recognition Node
def image_recognition_node(
    state: RobotState, config: RunnableConfig, store: BaseStore
) -> RobotState:
    res = image_chain.invoke(
        {"question": state["question"], "encoded_image": state["encoded_image"]}
    )
    return {"object_number": res["object_number"], "place_number": res["place_number"]}


# pick node
def pick_node(state: RobotState):
    operator.pick(state["object_number"])


# place node
def place_node(state: RobotState):
    operator.place(state["place_number"])


# Build workflow
graph = StateGraph(state_schema=RobotState, output=RobotOutputState)

# Add nodes
graph.add_node("image_recognition_node", image_recognition_node)
graph.add_node("pick_node", pick_node)
graph.add_node("place_node", place_node)

# Add edges to connect nodes
graph.add_edge(START, "image_recognition_node")
graph.add_edge("image_recognition_node", "pick_node")
graph.add_edge("pick_node", "place_node")
graph.add_edge("place_node", END)

robot_agent = graph.compile()

if __name__ == "__main__":
    print("stream started.")
    while True:
        for chunk in robot_agent.stream(
            {
                "question": input("Question: "),
                "encoded_image": operator.get_annotated_image(),
            },
            stream_mode="updates",
        ):
            print(chunk)

        operator.display()
        operator.init_sam()
