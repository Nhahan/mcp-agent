import asyncio
import json
import logging
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class MCPClient:
    """Handles JSON-RPC 2.0 communication with a single MCP server over stdio."""

    def __init__(self, name: str, process: asyncio.subprocess.Process):
        self.name = name
        self.process = process
        self._requests: Dict[str, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._is_running = False

        if not self.process.stdin or not self.process.stdout:
            raise ValueError(f"Process for MCP server '{self.name}' must have stdin and stdout piped.")

        self._start_reader()

    def _start_reader(self):
        """Starts the background task to read and process messages from stdout."""
        if self._reader_task is None or self._reader_task.done():
            self._is_running = True
            self._reader_task = asyncio.create_task(self._read_loop())
            logger.info(f"Started stdout reader task for MCP server '{self.name}'.")

    async def _read_loop(self):
        """Continuously reads lines from stdout and handles incoming messages."""
        if not self.process.stdout:
             logger.error(f"Stdout not available for MCP server '{self.name}', stopping reader.")
             self._is_running = False
             return

        while self._is_running and self.process.returncode is None:
            try:
                line_bytes = await self.process.stdout.readline()
                if not line_bytes:
                    if self.process.returncode is not None:
                         logger.info(f"EOF reached for '{self.name}' stdout (process terminated).")
                    else:
                         logger.warning(f"Empty line read from '{self.name}' stdout, but process still running? Waiting briefly.")
                         await asyncio.sleep(0.1)
                    break # Exit loop if EOF or process terminated

                line = line_bytes.decode().strip()
                if not line:
                    continue # Skip empty lines

                logger.debug(f"Received from '{self.name}': {line}")
                try:
                    message = json.loads(line)
                    self._handle_message(message)
                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON line from '{self.name}': {line}")
                except Exception as e:
                     logger.error(f"Error handling message from '{self.name}': {e}", exc_info=True)

            except asyncio.CancelledError:
                 logger.info(f"Reader task for '{self.name}' cancelled.")
                 break
            except Exception as e:
                logger.error(f"Error reading from '{self.name}' stdout: {e}", exc_info=True)
                # Avoid tight loop on persistent error
                await asyncio.sleep(0.5)
                # Check process status again
                if self.process.returncode is not None:
                    logger.warning(f"Process '{self.name}' terminated during read error.")
                    break

        self._is_running = False
        logger.info(f"Stopped stdout reader task for MCP server '{self.name}'.")
        # Clean up pending requests on exit
        for future in self._requests.values():
            if not future.done():
                future.set_exception(RuntimeError(f"MCP client '{self.name}' connection closed."))

    def _handle_message(self, message: Dict[str, Any]):
        """Processes a received JSON-RPC message (response or notification)."""
        if "id" in message and message["id"] is not None:
            # This is likely a response to a request
            request_id = str(message["id"])
            if request_id in self._requests:
                future = self._requests.pop(request_id)
                if "result" in message:
                    future.set_result(message["result"])
                elif "error" in message:
                    future.set_exception(MCPError(message["error"], request_id=request_id))
                else:
                    future.set_exception(MCPError({"code": -32603, "message": "Invalid response format"}, request_id=request_id))
            else:
                logger.warning(f"Received response for unknown request ID '{request_id}' from '{self.name}'.")
        else:
            # This is likely a notification
            # TODO: Implement notification handling if needed (e.g., callbacks)
            logger.info(f"Received notification from '{self.name}': {message.get('method')}")
            pass

    async def call(self, method: str, params: Optional[Dict[str, Any]] = None, timeout: float = 10.0) -> Any:
        """Sends a JSON-RPC request and waits for the response."""
        if not self._is_running or not self.process.stdin or self.process.stdin.is_closing():
            raise ConnectionError(f"MCP Client '{self.name}' is not running or stdin is closed.")

        request_id = str(uuid.uuid4())
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            request["params"] = params

        future = asyncio.Future()
        self._requests[request_id] = future

        try:
            request_str = json.dumps(request) + '\n' # Add newline for stdio transport
            logger.debug(f"Sending to '{self.name}': {request_str.strip()}")
            self.process.stdin.write(request_str.encode())
            await self.process.stdin.drain()
        except Exception as e:
            # If sending fails, remove the pending future
            if request_id in self._requests:
                 del self._requests[request_id]
            future.set_exception(ConnectionError(f"Failed to send request to '{self.name}': {e}"))
            # Re-raise or handle connection error appropriately
            raise ConnectionError(f"Failed to send request to '{self.name}': {e}") from e

        try:
            # Wait for the response future to be set by the reader task
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            # Timeout occurred, remove the pending future
            if request_id in self._requests:
                del self._requests[request_id]
            raise TimeoutError(f"Request '{request_id}' to '{self.name}' timed out after {timeout}s.")
        except Exception as e:
             # Future might have been set with an exception by the reader loop
             if not future.done(): # Check if the exception came from the wait_for itself
                 # If future not done but we got here, something else went wrong
                 if request_id in self._requests:
                     del self._requests[request_id]
                 future.set_exception(e)
             # Re-raise the exception (either from future or wait_for)
             raise e

    async def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None, timeout: float = 10.0) -> Any:
        """Convenience method to call an MCP tool."""
        params = {"name": tool_name}
        if arguments is not None:
            params["arguments"] = arguments
        return await self.call("tools/call", params, timeout=timeout)

    async def close(self):
        """Stops the reader task and cleans up."""
        logger.info(f"Closing MCP client for '{self.name}'.")
        self._is_running = False
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                logger.debug(f"Reader task for '{self.name}' cancellation confirmed.")
        # Note: Process closing (stdin/terminate) should be handled by mcp_service.stop_mcp_servers

class MCPError(Exception):
    """Represents an error received from an MCP server."""
    def __init__(self, error_obj: Dict[str, Any], request_id: Optional[str] = None):
        self.code = error_obj.get("code")
        self.message = error_obj.get("message")
        self.data = error_obj.get("data")
        self.request_id = request_id
        super().__init__(f"MCP Error (Request ID: {request_id}): Code={self.code}, Message={self.message}") 