import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, Any
import asyncio

from app.core.config import Settings, get_settings # Import Settings
from fastapi import Depends # Import Depends
from app.mcp_client.client import MCPClient, MCPError

logger = logging.getLogger(__name__)

async def log_stderr(stream: asyncio.StreamReader, prefix: str):
    """Helper async function to log stderr lines."""
    while True:
        try:
            line = await stream.readline()
            if not line: break
            logger.info(f"[{prefix}] {line.decode().strip()}")
        except Exception as e: logger.error(f"Error reading {prefix}: {e}"); break
    logger.debug(f"Stopped logging {prefix}")

class MCPService:
    def __init__(self, settings: Settings = Depends(get_settings)):
        self.settings = settings
        self.config_path = Path(settings.MCP_CONFIG_PATH)
        self._mcp_processes: Dict[str, asyncio.subprocess.Process] = {}
        self._mcp_clients: Dict[str, MCPClient] = {}
        self._start_stop_lock = asyncio.Lock()

    async def start_servers(self):
        """Starts MCP servers based on the configuration file."""
        async with self._start_stop_lock:
            if self._mcp_processes: # Prevent starting if already running
                logger.warning("MCP servers seem to be already running or starting.")
                return
                
            logger.info(f"Loading MCP configuration from {self.config_path}...")
            if not self.config_path.exists():
                logger.warning(f"MCP configuration file not found at {self.config_path}. Skipping MCP server startup.")
                return

            try:
                with open(self.config_path, 'r') as f:
                    mcp_config = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read or parse MCP config: {e}")
                return

            mcp_servers = mcp_config.get("mcpServers", {})
            if not mcp_servers:
                logger.info("No MCP servers defined in the configuration.")
                return

            logger.info(f"Found {len(mcp_servers)} MCP server(s) to start.")
            start_tasks = []
            for name, config in mcp_servers.items():
                command = config.get("command")
                args = config.get("args", [])
                if not command:
                    logger.warning(f"Skipping MCP server '{name}': 'command' not specified.")
                    continue
                # Create a task for each server start
                start_tasks.append(asyncio.create_task(self._start_single_server(name, command, args)))

            # Wait for all startup tasks to complete
            if start_tasks:
                await asyncio.gather(*start_tasks, return_exceptions=True) # Handle exceptions during startup
            logger.info("MCP server startup process completed.")

    async def _start_single_server(self, name: str, command: str, args: list):
        """Starts a single MCP server process and its client."""
        try:
            full_command = [command] + args
            logger.info(f"Starting MCP server '{name}' with command: {' '.join(full_command)}")
            process = await asyncio.create_subprocess_exec(
                *full_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            self._mcp_processes[name] = process
            try:
                self._mcp_clients[name] = MCPClient(name, process)
                logger.info(f"MCP server '{name}' started (PID {process.pid}) and client created.")
                # Start logging stderr in the background
                if process.stderr:
                    asyncio.create_task(log_stderr(process.stderr, f"{name}-stderr"))
            except ValueError as e:
                logger.error(f"Failed to create MCPClient for '{name}': {e}. Process terminated.")
                if process.returncode is None: process.terminate()
                del self._mcp_processes[name]

        except FileNotFoundError:
            logger.error(f"Failed to start MCP server '{name}': Command '{command}' not found.")
        except Exception as e:
            logger.error(f"Failed to start MCP server '{name}': {e}", exc_info=True)
        # Ensure process is removed if client creation failed or other exceptions occurred
        if name in self._mcp_processes and name not in self._mcp_clients:
             del self._mcp_processes[name]

    async def stop_servers(self):
        """Stops all managed MCP server processes and clients gracefully."""
        async with self._start_stop_lock:
            if not self._mcp_processes and not self._mcp_clients:
                 logger.info("No MCP servers or clients to stop.")
                 return
                 
            logger.info("Shutting down MCP clients...")
            close_tasks = [asyncio.create_task(client.close()) for client in self._mcp_clients.values()]
            if close_tasks: await asyncio.gather(*close_tasks, return_exceptions=True)
            self._mcp_clients.clear() # Clear clients after closing
            logger.info("MCP clients closed.")

            logger.info("Stopping MCP server processes...")
            stop_tasks = [asyncio.create_task(self._stop_single_server(name, process)) 
                          for name, process in self._mcp_processes.items()]
            if stop_tasks: await asyncio.gather(*stop_tasks, return_exceptions=True)
            self._mcp_processes.clear() # Clear processes after stopping
            logger.info("MCP server processes stopped.")

    async def _stop_single_server(self, name: str, process: asyncio.subprocess.Process):
        """Stops a single MCP server process."""
        logger.info(f"Stopping MCP server '{name}' (PID: {process.pid})...")
        try:
            if process.returncode is None:
                if process.stdin and not process.stdin.is_closing():
                    process.stdin.close()
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                    logger.info(f"MCP server '{name}' terminated gracefully (code: {process.returncode}).")
                except asyncio.TimeoutError:
                    logger.warning(f"MCP server '{name}' termination timed out, killing.")
                    process.kill()
                    await process.wait()
                    logger.info(f"MCP server '{name}' killed (code: {process.returncode}).")
            else:
                logger.info(f"MCP server '{name}' was already terminated (code: {process.returncode}).")
        except ProcessLookupError:
            logger.warning(f"Process for '{name}' not found during shutdown.")
        except Exception as e:
            logger.error(f"Error stopping MCP server '{name}': {e}", exc_info=True)

    def get_process_status(self, server_name: str) -> Dict[str, Any]:
         """Checks the status of a specific MCP server process."""
         process = self._mcp_processes.get(server_name)
         client = self._mcp_clients.get(server_name)
         status = {"server_name": server_name, "status": "not_found", "pid": None, "return_code": None, "client_active": False}
         if process:
             if process.returncode is None:
                 status["status"] = "running"
                 status["pid"] = process.pid
             else:
                 status["status"] = "terminated"
                 status["pid"] = process.pid
                 status["return_code"] = process.returncode
         return status

    def get_running_servers(self) -> list[str]:
         """Returns a list of names of currently running MCP servers."""
         return [name for name, process in self._mcp_processes.items() if process.returncode is None]

    async def call_mcp_tool(self, server_name: str, tool_name: str, arguments: Optional[Dict[str, Any]] = None, timeout: float = 10.0) -> Any:
        """Calls a specific tool on a managed MCP server."""
        client = self._mcp_clients.get(server_name)
        if client is None:
            # Check process status for more context
            proc_status = self.get_process_status(server_name)
            raise ValueError(f"No active MCP client or running process found for server '{server_name}' (Status: {proc_status['status']}).")

        try:
            logger.info(f"Calling tool '{tool_name}' on MCP server '{server_name}' with args: {arguments}")
            result = await client.call_tool(tool_name, arguments, timeout=timeout)
            logger.info(f"Received result from '{server_name}' tool '{tool_name}'.")
            return result
        except ConnectionError as e:
            logger.error(f"Connection error calling tool '{tool_name}' on '{server_name}': {e}")
            raise
        except TimeoutError as e:
            logger.error(f"Timeout calling tool '{tool_name}' on '{server_name}': {e}")
            raise
        except MCPError as e:
            logger.error(f"MCP error calling tool '{tool_name}' on '{server_name}': {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling tool '{tool_name}' on '{server_name}': {e}", exc_info=True)
            raise

# Dependency function (optional, but can be useful for clarity or testing)
def get_mcp_service(settings: Settings = Depends(get_settings)) -> MCPService:
     # This could potentially return a singleton managed elsewhere,
     # but for now, it relies on FastAPI's handling or overrides.
     # Creating it directly here might lead to multiple instances if not overridden.
     # The lifespan manager in main.py is the primary way to manage the singleton.
     # This function signature matches what FastAPI expects for dependency injection.
     # We will override this in main.py to return the instance from app_state.
     raise NotImplementedError("This function should be overridden in main.py to provide the singleton MCPService instance.") 