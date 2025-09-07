from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(server: FastMCP):
    print("Welcome to Robot MCP!")
    try:
        yield {"Hello": "World"}
    finally:
        # 服务器关闭时执行清理逻辑
        print("Robot MCP closed.")


mcp = FastMCP(
    name="Robot MCP", instructions="Robot Action MCP server.", lifespan=lifespan
)


@mcp.tool(name="echo")
def echo_tool(echo: str, ctx: Context) -> int:
    """Echo a message as a tool

    Args:
        echo: message to echo.
    """
    print(ctx.request_context.lifespan_context["Hello"])
    return echo


@mcp.resource("robot://{message}")
def echo_resource(message: str) -> str:
    """Echo a message as a resource"""
    return f"Resource echo: {message}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
