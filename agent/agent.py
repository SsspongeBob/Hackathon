from MCP import MCPManager
from dotenv import load_dotenv
import logging
from typing import Optional

load_dotenv(override=True)

logger = logging.getLogger(__name__)


class RobotAgent:
    def __init__(self):
        self.mcp_manager: MCPManager = MCPManager()
        self.compiled_graph = None
        self.initialize()

    def initialize(self):
        """resources initialization."""
        self.mcp_manager.initialize()
        logger.info("MCP Serever started successfully!")

    def close(self):
        """resources cleanup"""
        self.mcp_manager.close()
        logger.info("MCP Serever closed.")


if __name__ == "__main__":

    print(globals(), "\n",locals())
    import logging

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    agent = RobotAgent()

    while input("") != "exit":
        continue

    agent.close()
