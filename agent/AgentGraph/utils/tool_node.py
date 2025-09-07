from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
from langgraph.graph.message import add_messages
from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, ToolCall, ToolMessage, BaseMessage
from langchain_mcp_adapters.client import MultiServerMCPClient


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def initial_node(state: State) -> State:
    """Initial node to set up the state."""
    state["messages"] = [
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(name="divide", args={"a": 329993, "b": 13662}, id="123"),
                ToolCall(name="divide", args={"a": 323223, "b": 32133}, id="256"),
            ],
        )
    ]
    return state


async def tool_node(state: State):
    client = MultiServerMCPClient(
        {
            "robot": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            },
        }
    )
    async with client.session("robot") as session:
        return ToolNode(await client.get_tools())


if __name__ == "__main__":
    import asyncio

    graph_builder = StateGraph(State)

    graph_builder.add_node("initial_node", initial_node)
    graph_builder.add_node("tools", tool_node)

    # graph_builder.add_conditional_edges("initial_node", tools_condition)
    graph_builder.add_edge("initial_node", "tools")

    graph_builder.set_entry_point("initial_node")
    graph_builder.set_finish_point("tools")

    graph = graph_builder.compile()

    async def main():
        res = await graph.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": "What's 329993 divided by 13662?"}
                ]
            }
        )

        print(res)

    asyncio.run(main())
