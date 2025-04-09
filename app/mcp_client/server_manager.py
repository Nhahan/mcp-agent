import asyncio
import logging
import platform
import shlex
import signal
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

from app.mcp_client.client import MCPClient, MCPError

logger = logging.getLogger(__name__)

class MCPServerManager:
    """Manages multiple MCP server processes and their clients."""

    def __init__(self):
        self.servers: Dict[str, Dict[str, Any]] = {}
        self.clients: Dict[str, MCPClient] = {}
        self._tool_schema_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
        self._signal_handlers_setup = False
        self._shutdown_in_progress = False
        self._server_tasks: Set[asyncio.Task] = set()

    async def setup_signal_handlers(self):
        """Sets up signal handlers for graceful process termination."""
        if self._signal_handlers_setup:
            return
        
        loop = asyncio.get_running_loop()
        
        # Signal handling is platform specific
        if platform.system() != "Windows":
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(
                    sig, lambda s=sig: asyncio.create_task(self.shutdown(s))
                )
            logger.info("Signal handlers set up for SIGINT and SIGTERM")
        else:
            # Windows doesn't support signal handlers as cleanly in asyncio
            logger.info("Signal handlers not set up (Windows platform)")
        
        self._signal_handlers_setup = True

    async def shutdown(self, sig=None):
        """Closes all MCP clients and processes."""
        # 여러 시그널 핸들러 호출 방지용 락
        async with self._lock:
            if self._shutdown_in_progress:
                logger.info("Shutdown already in progress, ignoring duplicate call")
                return
            self._shutdown_in_progress = True
        
        if sig:
            logger.info(f"Received signal {sig.name}, shutting down MCP server manager...")
        else:
            logger.info("Shutting down MCP server manager...")
        
        # Cancel any server start tasks
        for task in self._server_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._server_tasks.clear()
        
        # Close all clients
        client_names = list(self.clients.keys())
        for name in client_names:
            client = self.clients.get(name)
            if client:
                logger.info(f"Closing MCP client '{name}'")
                await client.close()
        
        # Reference to processes for termination
        server_processes = {name: info.get('process') for name, info in self.servers.items()}
        
        # Clear collections
        self.clients.clear()
        self.servers.clear()
        self._tool_schema_cache.clear()
        
        # Terminate processes if they're still running
        for name, process in server_processes.items():
            if process and process.returncode is None:
                try:
                    logger.info(f"Terminating MCP server process '{name}'")
                    if platform.system() == "Windows":
                        process.terminate()
                    else:
                        process.send_signal(signal.SIGTERM)
                    
                    # Give it a moment to terminate gracefully
                    try:
                        await asyncio.wait_for(process.wait(), timeout=2.0)
                        logger.info(f"MCP server '{name}' terminated gracefully")
                    except asyncio.TimeoutError:
                        logger.warning(f"MCP server '{name}' did not terminate within timeout, force killing")
                        process.kill()
                        await process.wait()
                except Exception as e:
                    logger.error(f"Error terminating MCP server '{name}': {e}")
        
        logger.info("MCP server manager shutdown completed")
        # self._shutdown_in_progress = False # 락 외부에서 재설정 불필요
        
        # 시그널 핸들러 제거 및 시그널 다시 보내기 (uvicorn 종료 유도)
        if sig and platform.system() != "Windows":
             logger.info(f"Restoring default handler for {sig.name} and re-raising signal...")
             loop = asyncio.get_running_loop()
             loop.remove_signal_handler(sig)
             # 현재 프로세스에 시그널 다시 보내기
             os.kill(os.getpid(), sig)

    async def start_server(self, name: str, command: str, args: Optional[List[str]] = None) -> bool:
        """Starts an MCP server process with the specified command and arguments."""
        async with self._lock:
            if name in self.servers:
                logger.warning(f"MCP server '{name}' is already running or being started")
                return False
            
            # Add placeholder to indicate server is being started
            self.servers[name] = {'status': 'starting'}
        
        # Create task and track it
        start_task = asyncio.create_task(self._start_server_internal(name, command, args))
        self._server_tasks.add(start_task)
        start_task.add_done_callback(lambda t: self._server_tasks.discard(t))
        
        try:
            return await start_task
        except Exception as e:
            logger.error(f"Error starting MCP server '{name}': {e}", exc_info=True)
            async with self._lock:
                if name in self.servers and self.servers[name].get('status') == 'starting':
                    del self.servers[name]
            return False

    async def _start_server_internal(self, name: str, command: str, args: Optional[List[str]] = None) -> bool:
        """Internal method to start an MCP server and set up its client."""
        args = args or []
        full_command = [command] + args
        
        # Build full command details for logging
        cmd_str = command
        if args:
            cmd_str += " " + " ".join(shlex.quote(arg) for arg in args)
        logger.info(f"Starting MCP server '{name}' with command: {cmd_str}")
        
        try:
            # Start process with stdio pipes
            process = await asyncio.create_subprocess_exec(
                command, *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE  # Capture stderr but handle it separately
            )
            
            # Create stdout line reader for error output (async)
            stderr_task = asyncio.create_task(self._read_stderr(name, process))
            
            # Update server info
            async with self._lock:
                self.servers[name] = {
                    'command': command,
                    'args': args,
                    'process': process,
                    'status': 'started',
                    'stderr_task': stderr_task
                }
            
            # Create client
            client = MCPClient(name, process)
            
            # Initialize client (synchronous start of reader)
            # client.start_reader_task_sync() # Let connect handle reader start implicitly
            
            # Try to connect to verify the server is working
            connected = await client.connect(timeout=10.0)
            if not connected:
                logger.error(f"Failed to connect to MCP server '{name}'")
                # Clean up resources 
                await client.close()
                if process.returncode is None:
                    process.terminate()
                    await process.wait()
                stderr_task.cancel()
                try:
                    await stderr_task
                except asyncio.CancelledError:
                    pass  # Expected
                async with self._lock:
                    if name in self.servers:
                        del self.servers[name]
                return False
            
            # Successfully connected, store client
            async with self._lock:
                self.clients[name] = client
                self.servers[name]['status'] = 'connected'
                # Pre-cache tool schemas upon successful connection
                await self._update_tool_cache(name, client)
            
            logger.info(f"Successfully started and connected to MCP server '{name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to start MCP server '{name}': {e}", exc_info=True)
            async with self._lock:
                if name in self.servers:
                    server_info = self.servers[name]
                    process = server_info.get('process')
                    stderr_task = server_info.get('stderr_task')
                    
                    # Clean up if needed
                    if stderr_task and not stderr_task.done():
                        stderr_task.cancel()
                    if process and process.returncode is None:
                        process.terminate()
                        await process.wait()
                    
                    del self.servers[name]
            return False

    async def _read_stderr(self, name: str, process: asyncio.subprocess.Process):
        """Reads and logs stderr output from the server process."""
        try:
            while True:
                line = await process.stderr.readline()
                if not line:
                    break  # EOF
                try:
                    stderr_line = line.decode('utf-8', errors='replace').rstrip()
                    if stderr_line:
                        logger.warning(f"MCP '{name}' stderr: {stderr_line}")
                except Exception as decode_err:
                    logger.error(f"Error decoding stderr from MCP '{name}': {decode_err}")
            
            logger.info(f"Stderr stream closed for MCP '{name}'")
        except asyncio.CancelledError:
            logger.debug(f"Stderr reader for MCP '{name}' cancelled")
            raise
        except Exception as e:
            logger.error(f"Error reading stderr from MCP '{name}': {e}")

    async def list_servers(self) -> Dict[str, Dict[str, Any]]:
        """Returns a dictionary of server status information."""
        async with self._lock:
            # Create a safe copy with process objects removed (not serializable)
            result = {}
            for name, info in self.servers.items():
                # Exclude sensitive or non-serializable info
                status_info = {
                    k: v for k, v in info.items() 
                    if k not in ['process', 'stderr_task'] 
                }
                # Add client connection status if available
                client = self.clients.get(name)
                status_info['client_connected'] = client.is_connected() if client else False
                status_info['process_returncode'] = info['process'].returncode if info.get('process') else None
                result[name] = status_info
            return result

    async def stop_server(self, name: str) -> bool:
        """Stops an MCP server and its client."""
        async with self._lock:
            if name not in self.servers:
                logger.warning(f"Cannot stop MCP server '{name}' as it is not running")
                return False
            
            server_info = self.servers[name]
            client = self.clients.get(name)
            process = server_info.get('process')
            stderr_task = server_info.get('stderr_task')
        
        logger.info(f"Stopping MCP server '{name}'")
        
        # Close client first if available
        if client:
            logger.debug(f"Closing client for MCP server '{name}'")
            await client.close()
            async with self._lock:
                if name in self.clients:
                    del self.clients[name]
        
        # Terminate process if still running
        if process and process.returncode is None:
            logger.debug(f"Terminating process for MCP server '{name}'")
            try:
                process.terminate()
                try:
                    # Give it a moment to terminate gracefully
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning(f"MCP server process '{name}' did not terminate, force killing")
                    process.kill()
                    await process.wait()
            except Exception as e:
                logger.error(f"Error terminating MCP server process '{name}': {e}")
        
        # Cancel stderr reader task if needed
        if stderr_task and not stderr_task.done():
            stderr_task.cancel()
            try:
                await stderr_task
            except asyncio.CancelledError:
                pass  # Expected
        
        # Clean up server entry
        async with self._lock:
            if name in self.servers:
                del self.servers[name]
            if name in self._tool_schema_cache:
                del self._tool_schema_cache[name]
        
        logger.info(f"MCP server '{name}' stopped")
        return True

    async def reload_server(self, name: str, command: str, args: Optional[List[str]] = None) -> bool:
        """Stops and restarts an MCP server."""
        logger.info(f"Reloading MCP server '{name}'")
        
        # Stop the server
        await self.stop_server(name)
        
        # Start it again
        return await self.start_server(name, command, args)

    async def get_active_servers(self) -> list:
        """Returns a list of names of actively connected MCP servers."""
        async with self._lock:
            active_list = []
            for name, client in self.clients.items():
                # Check connection status if possible (e.g., via ping or stored state)
                if client.is_connected(): # Assuming MCPClient has an is_connected method
                    active_list.append(name)
                else:
                    # Attempt to ping or reconnect if needed?
                    # For now, just rely on the stored state
                    server_info = self.servers.get(name)
                    if server_info and server_info.get('status') == 'connected':
                        # Check if process is still alive
                        if server_info.get('process') and server_info['process'].returncode is None:
                            active_list.append(name)
                        else:
                            logger.warning(f"Server '{name}' marked connected, but process seems dead.")
                    else:
                         logger.warning(f"Client for server '{name}' exists but is not connected.")
            return active_list

    async def _update_tool_cache(self, server_name: str, client: MCPClient):
        """Fetches tools using client.get_tools() and updates the cache.
           Handles both list (standard) and dict (workaround) return types.
           Assumes the lock is already held by the caller.
        """
        try:
            logger.info(f"Fetching tools for '{server_name}' using client.get_tools()...")
            server_tools_result = await client.get_tools(timeout=10.0)

            processed_tools_list: List[Dict[str, Any]] = []

            if isinstance(server_tools_result, list):
                processed_tools_list = server_tools_result
                logger.debug(f"Received standard list of {len(processed_tools_list)} tools from '{server_name}'.")
            elif isinstance(server_tools_result, dict):
                logger.warning(f"Received tools as a dictionary from '{server_name}' via get_tools(). Applying workaround.")
                for tool_name_key, tool_info_value in server_tools_result.items():
                    if isinstance(tool_info_value, dict):
                        if 'name' not in tool_info_value:
                            tool_info_value['name'] = tool_name_key
                        processed_tools_list.append(tool_info_value)
                    else:
                        logger.warning(f"Skipping non-dictionary value in tools dictionary from '{server_name}': key={tool_name_key}")
            else:
                logger.warning(f"client.get_tools() for '{server_name}' returned unexpected type ({type(server_tools_result)}). Clearing cache.")
                # Clear cache (caller holds the lock)
                if server_name in self._tool_schema_cache:
                    self._tool_schema_cache[server_name].clear()
                return

            # Update cache with the processed list (caller holds the lock)
            server_cache = self._tool_schema_cache.setdefault(server_name, {})
            server_cache.clear() # Clear old cache for this server
            valid_tool_count = 0
            for tool_info in processed_tools_list:
                if isinstance(tool_info, dict) and 'name' in tool_info:
                    tool_name = tool_info['name']
                    server_cache[tool_name] = tool_info 
                    valid_tool_count += 1
                else:
                    logger.warning(f"Skipping invalid tool entry during cache update for '{server_name}': {tool_info}")
            logger.info(f"Updated tool cache for '{server_name}' with {valid_tool_count} tools.")

        except (TimeoutError, MCPError) as e:
            logger.error(f"Failed to fetch tools for '{server_name}' using get_tools(): {e}. Clearing cache.")
            # Clear cache (caller holds the lock)
            if server_name in self._tool_schema_cache:
                self._tool_schema_cache[server_name].clear()
        except Exception as e:
            logger.error(f"Unexpected error during get_tools() or cache update for '{server_name}': {e}", exc_info=True)

    async def get_available_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """Returns a dictionary of available tools grouped by server.
           Refreshes cache based on currently connected clients before returning.
        """
        async with self._lock: # 캐시 업데이트 및 조회를 위해 락 사용
            current_clients = list(self.clients.items()) # 현재 클라이언트 목록 복사
            
            # 연결된 클라이언트 기준으로 캐시 업데이트 시도
            for server_name, client in current_clients:
                if client.is_connected():
                    # 캐시 업데이트 (내부적으로 capabilities 가져오기 시도)
                    await self._update_tool_cache(server_name, client)
                else:
                    # 연결 끊긴 클라이언트면 캐시에서 제거 (선택적)
                    if server_name in self._tool_schema_cache:
                        logger.info(f"Removing disconnected server '{server_name}' from tool cache.")
                        del self._tool_schema_cache[server_name]
            
            # 최종적으로 업데이트된 캐시 내용 반환 (연결된 서버만 포함)
            available_tools = {}
            for server_name, server_cache in self._tool_schema_cache.items():
                 # 다시 한번 클라이언트 연결 상태 확인 (업데이트 중 끊겼을 수 있음)
                 client = self.clients.get(server_name)
                 if client and client.is_connected():
                      available_tools[server_name] = list(server_cache.values())
                      
        if not available_tools:
             logger.warning("get_available_tools: No tools found from any connected server.")
             
        return available_tools

    def get_tool_schema(self, server_name: str, tool_name: str) -> Optional[Dict[str, Any]]:
        """Retrieves the input schema for a specific tool from the cache."""
        # No lock needed for read access if update is atomic enough
        # but using lock for safety during potential updates
        # async with self._lock: # Read doesn't strictly need lock if _update is safe
        server_cache = self._tool_schema_cache.get(server_name)
        if server_cache:
            tool_info = server_cache.get(tool_name)
            if tool_info:
                # Return the inputSchema part
                return tool_info.get('inputSchema')
        
        logger.warning(f"Schema not found in cache for tool '{server_name}.{tool_name}'. Cache might be outdated or tool unavailable.")
        return None

    async def call_tool(self, server_name: str, tool_name: str, arguments: Optional[Dict[str, Any]] = None, timeout: float = 30.0) -> Any:
        """Finds the correct client and calls the specified tool."""
        async with self._lock:
            client = self.clients.get(server_name)

        if not client:
            raise ValueError(f"MCP server '{server_name}' not found or not connected.")
        if not client.is_connected():
             raise ConnectionError(f"MCP client for '{server_name}' is not connected.")

        try:
            # Use the client's call_tool method
            result = await client.call_tool(tool_name, arguments, timeout=timeout)
            return result
        except (TimeoutError, ConnectionError, MCPError) as e:
            logger.error(f"Error calling tool '{server_name}.{tool_name}': {e}")
            # Re-raise specific errors for handling in InferenceService
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling tool '{server_name}.{tool_name}': {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error during tool call: {e}") from e 