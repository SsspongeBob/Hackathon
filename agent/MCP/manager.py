import subprocess
import signal
import time
import os
from pathlib import Path
import logging
from threading import Thread
import queue
import psutil
import socket

logger = logging.getLogger(__name__)


class MCPManager(object):
    """Model Context Protocol Lifespan Manager."""

    _initialized = False
    _session_manager = None

    def __init__(
        self, server_path: str = str(Path(__file__).parent.joinpath("server.py"))
    ):
        logger.info(f"MCP Server Path: {server_path}")
        self.server_path: str = server_path
        self.output_queue: queue.Queue = queue.Queue()
        self.logger_thread: Thread | None = None

    def _check_and_kill_port_8000(self):
        """Check if port 8000 is occupied and kill the process if needed."""
        try:
            # First try to bind to the port to check availability
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            result = s.connect_ex(("127.0.0.1", 8000))
            s.close()

            if result == 0:  # Port is in use
                logger.warning("Port 8000 is in use, attempting to kill the process...")

                for proc in psutil.process_iter(["pid", "name", "connections"]):
                    try:
                        for conn in proc.connections():
                            if conn.laddr.port == 8000:
                                logger.info(
                                    f"Found process using port 8000: PID {proc.pid}"
                                )
                                proc.kill()
                                proc.wait(timeout=5)
                                logger.info(f"Successfully killed process {proc.pid}")
                                return
                    except (
                        psutil.NoSuchProcess,
                        psutil.AccessDenied,
                        psutil.ZombieProcess,
                    ):
                        continue

                logger.warning("Could not find or kill process using port 8000")
        except Exception as e:
            logger.error(f"Error checking/killing port 8000: {e}")

    def initialize(self):
        """MCP Initialization"""
        self._check_and_kill_port_8000()

        MCP_SERVER = self.server_path + ":mcp"
        try:
            self._session_manager = subprocess.Popen(
                ["mcp", "run", MCP_SERVER, "-t", "streamable-http"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                ),
            )
            logger.info(f"MCP server is starting, PID: {self._session_manager.pid}")
            self._initialized = True

            self.logger_thread = Thread(
                target=self._read_output_continuously,
                name="Thread-MCP-Logger",
                daemon=True,
            )
            self.logger_thread.start()

            while (
                self.output_queue.qsize() < 6
            ):  # TODO: need to take starting failure into consideration.
                time.sleep(0.5)
            logger.info(f"MCP server started, PID: {self._session_manager.pid}")

            if self._session_manager.poll() is not None:
                logger.warning(
                    f"MCP server process exited abnormally, exit code: {self._session_manager.returncode}"
                )

        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e.stderr}")
        except FileNotFoundError:
            logger.error("mcp command not found, please confirm MCP is installed")

    def _read_output_continuously(self):
        """read output from MCP sub process."""
        # TODO: add output record to file.
        for line in self._session_manager.stderr:
            self.output_queue.put(line)

    def info(self):
        """get info from subprocess."""
        if self._session_manager is None or self._session_manager.poll() is not None:
            logger.warning("MCP Server not running")
            return

        try:
            line = self.output_queue.get_nowait()
            logger.info(f"MCP output: {line}")
        except queue.Empty:
            logger.debug("No new output")

    def close(self):
        """Close MCP server and release port."""
        if self._session_manager is None:
            return

        try:
            if self._session_manager.poll() is None:
                logger.info(f"Closing MCP server (PID: {self._session_manager.pid})...")

                if os.name == "nt":
                    self._session_manager.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self._session_manager.terminate()

                try:
                    self._session_manager.wait(timeout=5)
                    logger.info("MCP server closed gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "Graceful shutdown timeout, force terminating process..."
                    )
                    self._session_manager.kill()
                    self._session_manager.wait()
                    logger.info("MCP server force terminated")
            else:
                logger.info(
                    f"MCP server already stopped (exit code: {self._session_manager.returncode})"
                )

        except Exception as e:
            logger.error(f"Error closing MCP server: {e}")
            try:
                self._session_manager.kill()
                self._session_manager.wait()
            except:
                pass
        finally:
            self._session_manager = None
            self._initialized = False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    mcp_manager = MCPManager()

    mcp_manager.initialize()

    while input("") != "exit":
        mcp_manager.info()

    mcp_manager.close()
