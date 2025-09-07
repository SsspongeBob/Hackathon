# main.py
import contextlib
from fastapi import FastAPI

from mcp.server.fastmcp import FastMCP

echo = FastMCP(name="EchoServer", stateless_http=True)


@echo.tool(description="A simple echo tool")
def echo_msg(message: str) -> str:
    return f"Echo: {message}"


math = FastMCP(name="MathServer", stateless_http=True)


@math.tool(description="A simple add tool")
def add_two(n: int) -> int:
    return n + 2


# Create a combined lifespan to manage both session managers
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(echo.session_manager.run())
        await stack.enter_async_context(math.session_manager.run())
        yield


app = FastAPI(lifespan=lifespan)
app.mount("/echo", echo.streamable_http_app())
app.mount("/math", math.streamable_http_app())

if __name__ == "__main":
    import uvicorn

    uvicorn.run(app=app)
