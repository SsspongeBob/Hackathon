from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
    {
        "robot": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        },
    }
)

if __name__ == "__main__":
    import asyncio

    async def main():
        print(await client.get_tools(server_name="robot"))

        async with client.session(server_name="robot") as session:
            print((await session.call_tool("base64image")).content[0].text)

            # print(await session.call_tool("robot_pick", {"object_no": 4}))
            #
            # await session.call_tool("robot_place", {"place_no": 1})

    asyncio.run(main=main())
