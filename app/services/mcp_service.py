import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, Any, List
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
        """Starts MCP servers based on the loaded configuration."""
        async with self._start_stop_lock:
            if self._mcp_processes: # Prevent starting if already running
                logger.warning("MCP servers seem to be already running or starting.")
                return
            
            mcp_servers = self._mcp_config.get("mcpServers", {})
            if not mcp_servers:
                logger.info("No MCP servers defined in the configuration.")
                return

            logger.info(f"Attempting to start {len(mcp_servers)} configured MCP server(s)...")
            start_tasks = []
            for name, config in mcp_servers.items():
                command = config.get("command")
                args = config.get("args", [])
                if not command:
                    logger.warning(f"Skipping MCP server '{name}': 'command' not specified.")
                    continue
                start_tasks.append(asyncio.create_task(self._start_single_server(name, command, args)))

            if start_tasks:
                await asyncio.gather(*start_tasks, return_exceptions=True)
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

    async def call_tool(self, server_name: str, tool_name: str, args: dict) -> Any:
        """
        특정 서버의 도구를 호출합니다.
        서버 이름, 도구 이름, 인자를 받아서 MCP 클라이언트를 통해 도구를 실행합니다.
        """
        logger.info(f"MCP 도구 호출: {server_name}/{tool_name} 인자: {args}")
        try:
            # 서버 확인
            if not self.is_server_available(server_name):
                available_servers = self.get_available_server_names()
                error_msg = f"MCP 서버 '{server_name}'이(가) 사용 가능하지 않습니다. 사용 가능한 서버: {', '.join(available_servers) if available_servers else '없음'}"
                logger.error(error_msg)
                raise MCPError(error_msg)
            
            # 클라이언트 가져오기
            client = self._mcp_clients.get(server_name)
            if not client:
                # 항상 여기에 도달하면 안 됨 (위에서 확인함)
                error_msg = f"MCP 클라이언트 '{server_name}'이(가) 존재하지 않습니다."
                logger.error(error_msg)
                raise MCPError(error_msg)
            
            # MCP 도구 호출
            try:
                # 모든 서버를 동일하게 처리 - 서버 유형 구분 없음
                logger.info(f"MCP 도구 호출: {server_name}/{tool_name}")
                
                # 필수 인자 유효성 검사
                if tool_name.lower().endswith("write_to_terminal") and "command" not in args:
                    raise ValueError("command 인자가 필요합니다.")
                elif tool_name.lower().endswith("send_control_character") and "letter" not in args:
                    raise ValueError("letter 인자가 필요합니다.")
                
                # 도구 호출
                response = await client.call_tool(tool_name, args)
                return response
                
            except Exception as e:
                if isinstance(e, MCPError):
                    logger.error(f"MCP 오류: {str(e)}")
                    raise
                else:
                    error_msg = f"MCP 도구 호출 중 오류 발생: {type(e).__name__}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    raise MCPError(error_msg) from e
        
        except MCPError as e:
            # MCP 관련 오류 그대로 전달
            logger.error(f"MCP 오류 발생: {str(e)}")
            raise
        except Exception as e:
            # 기타 오류는 MCPError로 변환
            error_msg = f"도구 호출 중 예상치 못한 오류 발생: {type(e).__name__}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise MCPError(error_msg) from e

    def _format_tool_name(self, tool_name: str) -> str:
        """
        도구 이름을 MCP 규약에 맞게 포맷팅합니다.
        """
        # 도구 이름에 접두사가 없을 경우 추가
        if not tool_name.startswith("mcp_"):
            # iterm-mcp의 경우 접두사 붙이기
            return tool_name
        return tool_name

    def is_server_available(self, server_name: str) -> bool:
        """주어진 이름의 MCP 서버가 등록되어 있는지 확인합니다."""
        return server_name in self._mcp_clients
    
    def is_server_connected(self, server_name: str) -> bool:
        """주어진 이름의 MCP 서버가 연결되어 있는지 확인합니다."""
        if server_name not in self._mcp_clients:
            return False
            
        client = self._mcp_clients[server_name]
        return hasattr(client, 'is_connected') and client.is_connected

    def get_server_tools(self, server_name: str) -> dict:
        """특정 MCP 서버에서 제공하는 도구 정보를 가져옵니다.
        
        Args:
            server_name: 도구 정보를 가져올 MCP 서버 이름
            
        Returns:
            도구 이름과 정보를 포함하는 딕셔너리
            예: {"tool_name": {"description": "...", "parameters": {...}}}
        """
        if server_name not in self._mcp_clients:
            logger.warning(f"MCP 서버 '{server_name}'가 등록되지 않았습니다.")
            return {}
        
        client = self._mcp_clients[server_name]
        
        try:
            # 도구 목록 캐시 확인
            if hasattr(client, 'cached_tools') and client.cached_tools:
                return client.cached_tools
            
            # 서버가 응답하지 않거나 초기화가 필요한 경우
            if not client.is_connected:
                logger.warning(f"MCP 서버 '{server_name}'에 연결되지 않았습니다.")
                return {}
            
            # MCP 클라이언트를 통해 실제 도구 정보 가져오기
            # 구현에 따라 다를 수 있으므로 예외 처리
            try:
                tools = client.get_tools()
                # 도구 정보 캐싱
                client.cached_tools = tools
                return tools
            except (NotImplementedError, AttributeError):
                logger.warning(f"MCP 서버 '{server_name}'에서 도구 정보를 가져오는 메서드를 지원하지 않습니다.")
                return {}
        except Exception as e:
            logger.error(f"도구 정보 가져오기 실패 '{server_name}': {e}")
            return {}

# Dependency function
def get_mcp_service(settings: Settings = Depends(get_settings)) -> MCPService:
     raise NotImplementedError("This function should be overridden in main.py to provide the singleton MCPService instance.") 