from mcp.server.fastmcp import FastMCP, Context
from robot_function.operator import Operator
from contextlib import asynccontextmanager
from typing import Any, TypedDict
import json

operator = Operator()
operator.display()
operator.init_sam()


@asynccontextmanager
async def lifespan(server: FastMCP):
    print("Welcome to Robot MCP!")
    try:
        yield {"operator": operator}
    finally:
        # 服务器关闭时执行清理逻辑
        print("Robot MCP closed.")


class ContextType(TypedDict):
    operator: Operator


mcp = FastMCP(
    name="Robot MCP", instructions="Robot Action MCP server."
)


@mcp.tool(name="robot_pick")
def robot_pick(object_no: int) -> str:
    """我会给你一个物体的代号，也就是一个数字，它对应物体的掩码，该函数会通过这个掩码获得物体的具体位置并实现物体的抓取，\
       如果函数执行成功，物体会被夹取，并等待放置到另一个位置，并且返回机械臂的执行成功后的空间坐标。

    Args:
        object_no: 物体的代号（标签）。
    Return:
        返回字符串，包含机械臂的空间坐标。
    """
    position = operator.pick(int(object_no)).tolist()
    return f"夹取成功，机械臂空间坐标：{json.dumps(position)}"


@mcp.tool(name="robot_place")
def robot_place(place_no: int) -> int:
    """我会给你一个要放置物体的位置的代号，也就是一个数字，它对应这个位置的XY坐标，该函数会通过这个XY坐标获得具体位置，并实现物体的放置。\
       如果函数执行成功，物体会被放置在指定位置，并且返回机械臂的执行成功后的空间坐标。

    Args:
        place_no: 放置位置的代号（标签）.
    Return:
        返回字符串，包含机械臂的空间坐标。
    """
    position = operator.place(int(place_no)).tolist()
    return f"放置成功，机械臂空间坐标：{json.dumps(position)}"


@mcp.tool(name="base64image")
def base64image() -> str:
    """Echo a message as a resource"""
    operator.display()
    operator.init_sam()
    return operator.get_annotated_image()


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
