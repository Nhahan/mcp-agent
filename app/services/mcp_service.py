import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import asyncio
from jsonschema import validate, ValidationError

from app.core.config import settings
from app.mcp_client.client import MCPClient, MCPError

logger = logging.getLogger(__name__)

class MCPService:
    def __init__(self):
        self.config_path = Path(settings.mcp_config_path)
        self._mcp_processes: Dict[str, asyncio.subprocess.Process] = {}
        self._mcp_clients: Dict[str, MCPClient] = {}
        self._start_stop_lock = asyncio.Lock()
        self._mcp_config: Dict[str, Any] = {} # Store loaded config
        self._load_config() # Load config during initialization

    def _load_config(self):
        """Loads the MCP configuration file."""
        if not self.config_path.exists():
            logger.warning(f"MCP configuration file not found at {self.config_path}. No MCP servers available.")
            self._mcp_config = {"mcpServers": {}}
            return
        try:
            with open(self.config_path, 'r') as f:
                self._mcp_config = json.load(f)
            logger.info(f"MCP configuration loaded from {self.config_path}.")
        except Exception as e:
            logger.error(f"Failed to read or parse MCP config: {e}")
            self._mcp_config = {"mcpServers": {}}

    def get_available_server_names(self) -> List[str]:
        """Returns a list of all server names defined in the MCP config."""
        # Ensure config is loaded if not already
        if not self._mcp_config: 
            self._load_config()
        return list(self._mcp_config.get("mcpServers", {}).keys())

    async def start_servers(self):
        """MCP 서버들을 시작합니다."""
        logger.info("Attempting to start configured MCP server(s)...")
        
        if not self._mcp_config:
            self._load_config()  # 설정이 없으면 다시 로드 시도
            
        if not self._mcp_config:
            logger.warning("MCP 서버 설정이 없습니다. 서버를 시작할 수 없습니다.")
            return
            
        servers_to_start = self._mcp_config.get("mcpServers", {})
        if not servers_to_start:
            logger.warning("시작할 MCP 서버가 없습니다.")
            return
            
        # 시작되지 않은 서버만 시작
        servers_to_start = {name: config for name, config in servers_to_start.items() 
                           if name not in self._mcp_processes or self._mcp_processes[name].returncode is not None}
        
        if not servers_to_start:
            logger.info("모든 MCP 서버가 이미 실행 중입니다.")
            return
            
        logger.info(f"Starting MCP servers: {', '.join(servers_to_start.keys())}")
            
        # 서버 시작
        for server_name, server_config in servers_to_start.items():
            logger.info(f"Starting MCP server '{server_name}'...")
            
            try:
                # 서버 프로세스 시작
                process = await self._start_mcp_server_process(server_name, server_config)
                self._mcp_processes[server_name] = process # 프로세스 먼저 등록
                
                # 클라이언트 생성 및 시작
                client = MCPClient(server_name, process)
                self._mcp_clients[server_name] = client # 클라이언트 등록
                client.start_reader_task_sync()  # 리더 시작
                logger.info(f"MCP server '{server_name}' started (PID {process.pid}) and client created.")
                
                # 연결 및 도구 목록 가져오기 시도
                logger.info(f"Attempting connection and tool fetch for '{server_name}'")
                is_connected = await client.connect(timeout=10.0) 

                if is_connected:
                    logger.info(f"MCP 서버 '{server_name}' 연결 및 초기화 성공")
                    await asyncio.sleep(0.5) # 지연 유지
                    
                    # 도구 정보 가져오기
                    tools = await self._fetch_server_tools(server_name) 
                else:
                    logger.error(f"MCP 서버 '{server_name}' 연결/초기화 실패.")

            except Exception as e:
                logger.error(f"MCP server '{server_name}' 시작 또는 초기화 중 오류 발생: {e}", exc_info=True)
                # 실패 시 정리
                if server_name in self._mcp_clients:
                     await self._mcp_clients[server_name].close() # 클라이언트 닫기
                     del self._mcp_clients[server_name]
                if server_name in self._mcp_processes:
                    proc = self._mcp_processes.pop(server_name)
                    if proc.returncode is None: 
                        try: 
                            proc.terminate()
                            await asyncio.wait_for(proc.wait(), timeout=1.0)
                        except: pass # 정리 중 오류는 무시
                
        logger.info("MCP server startup process completed.")

    async def stop_servers(self):
        """Stops all managed MCP server processes and clients gracefully."""
        async with self._start_stop_lock:
            if not self._mcp_processes and not self._mcp_clients:
                 logger.info("No MCP servers or clients to stop.")
                 return
            logger.info("Shutting down MCP clients...")
            close_tasks = [asyncio.create_task(client.close()) for client in self._mcp_clients.values()]
            if close_tasks: await asyncio.gather(*close_tasks, return_exceptions=True)
            self._mcp_clients.clear()
            logger.info("MCP clients closed.")
            logger.info("Stopping MCP server processes...")
            stop_tasks = [asyncio.create_task(self._stop_single_server(name, process)) 
                          for name, process in self._mcp_processes.items()]
            if stop_tasks: await asyncio.gather(*stop_tasks, return_exceptions=True)
            self._mcp_processes.clear()
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
         status = {"server_name": server_name, "status": "not_found", "pid": None, "return_code": None}
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

    def list_servers(self) -> list[str]:
        """Returns a list of all available server names.
        This combines both configured servers from mcp.json and actively running servers.
        """
        # Combine configured and running servers (might be redundant)
        configured_servers = set(self.get_available_server_names())
        running_servers = set(self.get_running_servers())
        return list(configured_servers.union(running_servers))

    def get_server_tools(self, server_name: str) -> Dict[str, Any]:
        """
        특정 MCP 서버의 도구 목록을 가져옵니다.
        
        Args:
            server_name: MCP 서버 이름
            
        Returns:
            Dict[str, Any]: 도구 이름을 키로, 도구 정보를 값으로 가지는 딕셔너리
        """
        client = self._mcp_clients.get(server_name)
        if not client:
            logger.warning(f"Client for MCP server '{server_name}' not found")
            return {}
        
        tools = client.get_tools_info()
        if not tools:
            logger.warning(f"No tools found for MCP server '{server_name}'")
            return {}
            
        return tools
    
    def get_tool_info(self, server_name: str, tool_name: str) -> Dict[str, Any]:
        """
        특정 MCP 서버의 특정 도구 정보를 가져옵니다.
        
        Args:
            server_name: MCP 서버 이름
            tool_name: 도구 이름
            
        Returns:
            Dict[str, Any]: 도구 정보
        """
        server_tools = self.get_server_tools(server_name)
        if not server_tools or tool_name not in server_tools:
            logger.warning(f"Tool '{tool_name}' not found in server '{server_name}'")
            return {}
            
        tool_info = server_tools.get(tool_name, {})
        return {
            'description': tool_info.get('description', 'No description available'),
            'input_schema': tool_info.get('inputSchema', {}),
        }
        
    async def get_tool_schema(self, server_name: str, tool_name: str) -> Optional[Dict]:
        """
        특정 MCP 서버의 특정 도구 스키마를 가져옵니다.
        
        Args:
            server_name: MCP 서버 이름
            tool_name: 도구 이름
            
        Returns:
            Optional[Dict]: 도구 스키마 (입력 스키마)
        """
        tool_info = self.get_tool_info(server_name, tool_name)
        if not tool_info:
            return None
            
        return tool_info.get('input_schema')

    async def call_mcp_tool(self, server_name: str, tool_name: str, args: dict) -> Any:
        """MCP 서버의 도구를 호출합니다.
        
        Args:
            server_name: MCP 서버 이름
            tool_name: 호출할 도구 이름
            args: 도구에 전달할 인수
            
        Returns:
            도구 호출 결과
        """        
        # 서버가 등록되었는지 확인
        if server_name not in self._mcp_clients:
            error_msg = f"MCP 서버 '{server_name}'가 등록되지 않았습니다."
            logger.error(error_msg)
            return {"error": error_msg}
            
        client = self._mcp_clients[server_name]
        
        # 서버가 연결되었는지 확인 (is_connected 속성 사용)
        if client is None:
            error_msg = f"MCP 클라이언트 '{server_name}' 객체가 None입니다. (호출 불가)"
            logger.error(error_msg)
            return {"error": error_msg}
            
        try:
            is_conn = client.is_connected
        except Exception as conn_e:
            logger.error(f"Error checking is_connected for '{server_name}': {conn_e}", exc_info=True)
            return {"error": f"Failed to check connection status for {server_name}: {str(conn_e)}"}
            
        if not is_conn:
            error_msg = f"MCP 서버 '{server_name}'가 연결되어 있지 않습니다."
            logger.error(error_msg)
            return {"error": error_msg}
            
        try:
            result = await client.call_tool(tool_name, args)
            return result
        except MCPError as e:
            logger.error(f"MCP 도구 '{server_name}/{tool_name}' 호출 실패: {e}", exc_info=True)
            return {"error": str(e)}
        except asyncio.TimeoutError:
            error_msg = f"MCP 도구 '{server_name}/{tool_name}' 호출 시간 초과"
            logger.error(error_msg)
            return {"error": error_msg}
        except Exception as e:
            logger.error(f"MCP 도구 '{server_name}/{tool_name}' 호출 중 예상치 못한 오류 발생: {e}", exc_info=True)
            return {"error": f"예상치 못한 오류: {str(e)}"}

    async def _fetch_server_tools(self, server_name: str) -> dict:
        """Fetches and caches the list of tools for a specific server."""
        client = self._mcp_clients.get(server_name)
        if not client:
            logger.warning(f"Client '{server_name}' not found for fetching tools.")
            return {}
        
        try:
            # Attempt to get tools with a timeout
            tools = await asyncio.wait_for(client.get_tools(), timeout=10.0)
            if tools:
                logger.info(f"Successfully fetched {len(tools)} tools for '{server_name}'. Caching...")
                client.cached_tools = tools # Cache the fetched tools
                return tools
            else:
                logger.warning(f"No tools received from server '{server_name}'. Returning empty dict.")
                client.cached_tools = {} # Cache empty result
                return {}
        except asyncio.TimeoutError:
            logger.error(f"Timeout while fetching tools for '{server_name}'.")
            client.cached_tools = {} # Cache failure (empty)
            return {}
        except MCPError as e:
            logger.error(f"MCPError fetching tools for '{server_name}': {e}")
            client.cached_tools = {} # Cache failure (empty)
            return {}
        except Exception as e:
            logger.error(f"Unexpected error fetching tools for '{server_name}': {e}", exc_info=True)
            client.cached_tools = {} # Cache failure (empty)
            return {}
    
    def get_all_tool_details(self) -> Dict[str, Dict]:
        """
        모든 MCP 서버의 도구 세부 정보를 가져옵니다.
        
        Returns:
            Dict[str, Dict]: 서버 이름을 키로, 도구 세부 정보를 값으로 가지는 딕셔너리
        """
        tool_details = {}
        for server_name in self._mcp_clients.keys():
            tools = self.get_server_tools(server_name)
            if tools:
                tool_details[server_name] = {}
                for tool_name, tool_info in tools.items():
                    tool_details[server_name][tool_name] = {
                        'description': tool_info.get('description', 'No description available')
                    }
        return tool_details
    
    def get_tool_details(self, server_name: str, tool_name: str) -> Dict:
        """
        특정 MCP 서버의 특정 도구 세부 정보를 가져옵니다.
        
        Args:
            server_name: MCP 서버 이름
            tool_name: 도구 이름
            
        Returns:
            Dict: 도구 세부 정보 (설명 등을 포함)
        """
        tools = self.get_server_tools(server_name)
        if not tools or tool_name not in tools:
            logger.warning(f"Tool '{tool_name}' not found in server '{server_name}' or server not available")
            return {'description': 'Tool not found or server not available'}
        
        tool_info = tools.get(tool_name, {})
        return {
            'description': tool_info.get('description', 'No description available'),
            # 추가적인 도구 메타데이터가 있다면 여기에 포함
            'parameters': tool_info.get('parameters', {}),
            'returnType': tool_info.get('returnType', {}),
            'inputSchema': tool_info.get('inputSchema', {})
        }
    
    async def execute_tool(self, server_name: str, tool_name: str, arguments: Dict) -> Any:
        """
        MCP 서버의 도구를 실행합니다.
        
        Args:
            server_name: MCP 서버 이름
            tool_name: 실행할 도구 이름
            arguments: 도구에 전달할 인수
            
        Returns:
            Any: 도구 실행 결과
        """
        return await self.call_mcp_tool(server_name, tool_name, arguments)

    def validate_tool_arguments(self, server_name: str, tool_name: str, arguments: Dict) -> Tuple[bool, str, Dict]:
        """Validates the provided arguments against the tool's inputSchema."""
        logger.debug(f"Validating arguments for tool '{server_name}.{tool_name}'")
        tool_details = self.get_tool_details(server_name, tool_name)

        schema = tool_details.get('inputSchema')

        if not schema or not isinstance(schema, dict):
            # If no schema is defined, log a warning and return True (no validation possible)
            logger.warning(f"No valid inputSchema found for tool '{server_name}.{tool_name}'. Skipping validation.")
            return True, "Schema not found, validation skipped.", arguments

        try:
            # Ensure arguments is a dict (it should be, but double-check)
            if not isinstance(arguments, dict):
                 raise TypeError("Arguments must be a dictionary.")

            # Perform validation using jsonschema
            # The `validate` function will raise ValidationError if invalid
            validate(instance=arguments, schema=schema)

            logger.debug(f"Arguments validated successfully for '{server_name}.{tool_name}' using jsonschema.")
            return True, "Arguments are valid.", arguments

        except ValidationError as e:
            # Validation failed
            # Provide a more informative error message from the validation exception
            # e.g., "'param_name' is a required property" or "123 is not of type 'string'"
            error_path = " -> ".join(map(str, e.path)) if e.path else "root"
            message = f"Validation failed for parameter '{error_path}': {e.message}"
            logger.warning(f"Validation failed for '{server_name}.{tool_name}': {message}")
            return False, message, arguments

        except TypeError as e:
             # Handle cases where arguments themselves are not a dict
             message = f"Invalid arguments format: {str(e)}"
             logger.warning(f"Validation failed for '{server_name}.{tool_name}': {message}")
             return False, message, arguments

        except Exception as e:
            # Catch other potential errors during validation
            message = f"An unexpected error occurred during validation: {str(e)}"
            logger.error(f"Unexpected validation error for '{server_name}.{tool_name}': {e}", exc_info=True)
            return False, message, arguments

    async def _start_mcp_server_process(self, server_name: str, server_config: dict) -> asyncio.subprocess.Process:
        """Starts the MCP server process based on configuration."""
        command = server_config.get('command')
        args = server_config.get('args', [])

        if not command:
            raise ValueError(f"'command' not specified for MCP server '{server_name}'")

        try:
            full_command = [command] + args
            logger.info(f"Executing command for MCP server '{server_name}': {' '.join(full_command)}")
            process = await asyncio.create_subprocess_exec(
                *full_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            logger.info(f"MCP server '{server_name}' process started with PID: {process.pid}")
            # Start the stderr reader task
            self._start_stderr_reader(server_name, process)
            return process
        except FileNotFoundError:
            logger.error(f"Command '{command}' not found for MCP server '{server_name}'. Check your PATH or command spelling.")
            raise
        except Exception as e:
            logger.error(f"Failed to execute command for MCP server '{server_name}': {e}", exc_info=True)
            raise

    def _start_stderr_reader(self, server_name: str, process: asyncio.subprocess.Process):
        async def read_stderr():
            prefix = f"{server_name}-stderr"
            while process.returncode is None:
                try:
                    if process.stderr:
                        line = await process.stderr.readline()
                        if not line:
                            break
                        logger.info(f"[{prefix}] {line.decode(errors='ignore').strip()}")
                    else:
                        await asyncio.sleep(0.1) # stderr이 없으면 잠시 대기
                except Exception as e:
                    if process.returncode is None: # 프로세스가 아직 실행 중일 때만 오류 로깅
                        logger.error(f"Error reading {prefix}: {e}", exc_info=True)
                    break
        
        # 오류 핸들러가 종료될 때까지 기다리지 않고 백그라운드에서 실행
        asyncio.create_task(read_stderr(), name=f"{server_name}_stderr_reader")