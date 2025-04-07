import asyncio
import json
import logging
import uuid
import time
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

        # 서버 상태 관련 속성
        self._connected = False
        self._capabilities = {}
        self._cached_tools = None
        
        # 로깅 설정
        self.logger = logging.getLogger(f"app.mcp_client.{name}")

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

                # 원시 데이터 로깅 추가
                raw_line_for_log = line_bytes.decode(errors='replace').strip() # 로깅용 디코딩 (오류시 대체 문자)
                logger.debug(f"Received raw bytes from '{self.name}': {line_bytes[:100]}... ({len(line_bytes)} bytes), Decoded (log only): {raw_line_for_log}")
                
                line = line_bytes.decode().strip()
                if not line:
                    continue # Skip empty lines

                # logger.debug(f"Received from '{self.name}': {line}") # 위에서 로깅하므로 중복 제거 가능
                try:
                    message = json.loads(line)
                    logger.debug(f"Parsed JSON message from '{self.name}': {message}") # 파싱 성공 로그 추가
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
            request_id = str(message["id"])
            if request_id in self._requests:
                future = self._requests.pop(request_id)
                # Future 완료 전 응답 내용 로깅
                logger.debug(f"Completing future for request '{request_id}' from '{self.name}'. Response: {message}")
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

    async def connect(self, timeout: float = 10.0) -> bool:
        """Establishes connection by sending initialize request and waiting for response."""
        if self._connected:
            self.logger.info(f"Already connected to '{self.name}'.")
            return True
        if not self._is_running:
            self.logger.warning(f"Cannot connect to '{self.name}', reader task is not running.")
            return False
        
        self.logger.info(f"Attempting to initialize connection with '{self.name}'...")
        try:
            # initialize 요청 파라미터 수정: params 내부에 capabilities: {} 추가
            init_params = {
                "protocolVersion": "1.0", 
                "clientInfo": {         
                    "name": "MCP Agent (Python)",
                    "version": "0.1.0", 
                },
                "capabilities": {} # 두 서버 모두 params 내부에 이 필드를 요구하는 것으로 보임
            }
            
            # self.call 메서드는 params를 그대로 전달함
            init_result = await self.call("initialize", init_params, timeout=timeout)
            
            # Process successful initialization
            if init_result:
                self._capabilities = init_result.get('capabilities', {})
                self._connected = True # Set connected flag to True
                return True
            else:
                 self.logger.error(f"Initialization call to '{self.name}' returned unexpected result: {init_result}")
                 self._connected = False
                 return False
        except (ConnectionError, TimeoutError, MCPError) as e:
            self.logger.error(f"Failed to initialize connection with '{self.name}': {e}")
            self._connected = False
            return False
        except Exception as e:
             self.logger.error(f"Unexpected error during connection initialization with '{self.name}': {e}", exc_info=True)
             self._connected = False
             return False

    async def call(self, method: str, params: Dict[str, Any] | None = None, timeout: float = 10.0) -> Any:
        """Sends a JSON-RPC request and waits for the response."""
        if not self._is_running or not self.process.stdin or self.process.stdin.is_closing():
            raise ConnectionError(f"MCP Client '{self.name}' is not running or stdin is closed.")

        # 디버깅: ID 생성 전 로깅
        logger.debug(f"Generating request ID for method '{method}'...")
        try:
            request_id = str(uuid.uuid4())
            logger.debug(f"Generated request ID: {request_id}") # 생성된 ID 로깅
        except Exception as id_gen_e:
             logger.error(f"Error generating UUID for request ID: {id_gen_e}", exc_info=True)
             raise RuntimeError(f"Failed to generate request ID: {id_gen_e}") from id_gen_e
        
        request: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            request["params"] = params

        future: asyncio.Future[Any] = asyncio.Future()
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
            result = await asyncio.wait_for(future, timeout=timeout)
            # 성공적인 결과 로깅 (Future 완료 시 로깅되지만 여기서도 확인차 로깅)
            logger.debug(f"Successfully received result for request '{request_id}' ('{method}') from '{self.name}': {result}")
            return result
        except asyncio.TimeoutError:
            # Timeout occurred, remove the pending future
            if request_id in self._requests:
                del self._requests[request_id]
            raise TimeoutError(f"Request '{request_id}' to '{self.name}' timed out after {timeout}s.")
        except Exception as e:
             # 오류 발생 시 로깅 (Future가 예외로 완료된 경우 포함)
             logger.error(f"Error waiting for response for request '{request_id}' ('{method}') from '{self.name}': {e}")
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
        """Stops the reader task and performs cleanup."""
        self.logger.info(f"Closing MCP client for '{self.name}'...")
        self._connected = False # 연결 상태 해제
        self._is_running = False # 실행 상태 해제

        if self._reader_task and not self._reader_task.done():
            self.logger.debug(f"Cancelling reader task for '{self.name}'...")
            self._reader_task.cancel()
            try:
                await self._reader_task
                self.logger.debug(f"Reader task for '{self.name}' successfully cancelled and awaited.")
            except asyncio.CancelledError:
                self.logger.debug(f"Reader task for '{self.name}' confirmed cancelled.")
            except Exception as e:
                 self.logger.error(f"Error awaiting cancelled reader task for '{self.name}': {e}", exc_info=True)
        else:
             self.logger.debug(f"Reader task for '{self.name}' was already done or not started.")

        # Clean up any remaining pending requests
        pending_reqs = list(self._requests.keys())
        if pending_reqs:
            self.logger.warning(f"Cleaning up {len(pending_reqs)} pending requests for '{self.name}' due to client closure.")
            for req_id in pending_reqs:
                future = self._requests.pop(req_id, None)
                if future and not future.done():
                    future.set_exception(RuntimeError(f"MCP client '{self.name}' was closed."))

        self.logger.info(f"MCP client '{self.name}' closed.")

    def is_connected(self) -> bool:
        """클라이언트가 MCP 서버에 연결되어 있는지 여부를 반환합니다."""
        # 프로세스가 살아있고, 리더 태스크가 활성 상태이며, _connected 플래그가 True인지 확인
        process_alive = self.process is not None and self.process.returncode is None
        reader_active = self._reader_task is not None and not self._reader_task.done()
        return process_alive and reader_active and self._connected

    @property
    def cached_tools(self) -> Optional[dict]:
        """서버 도구 목록의 캐시를 반환합니다."""
        return self._cached_tools
    
    @cached_tools.setter
    def cached_tools(self, value: dict):
        self._cached_tools = value

    async def get_tools(self, timeout: float = 10.0) -> dict:
        """Fetches the list of available tools from the server using the mcp/discover method."""
        if self._cached_tools is not None:
            return self._cached_tools
        
        self.logger.info(f"Fetching tools from '{self.name}'...")
        try:
            # tools/list 사용
            result = await self.call("tools/list", timeout=timeout)
            
            # 결과 구조 확인 (tools/list 는 보통 {"tools": [...]}) 리스트 반환 확인
            if isinstance(result, dict) and "tools" in result and isinstance(result["tools"], list):
                # 도구 이름을 키로, 도구 정보를 값으로 하는 딕셔너리로 변환 (표준화)
                formatted_tools = {tool_info["name"]: tool_info for tool_info in result["tools"] if isinstance(tool_info, dict) and "name" in tool_info}
                self._cached_tools = formatted_tools # Cache the tools
                return self._cached_tools
            else:
                self.logger.warning(f"Unexpected format in tools/list response from '{self.name}': {result}")
                self._cached_tools = {} # Cache empty dict on unexpected format
                return {}
        except (ConnectionError, TimeoutError, MCPError) as e:
            self.logger.error(f"Failed to discover tools from '{self.name}': {e}")
            self._cached_tools = {} # Cache empty dict on error
            return {}
        except Exception as e:
             self.logger.error(f"Unexpected error discovering tools from '{self.name}': {e}", exc_info=True)
             self._cached_tools = {} # Cache empty dict on error
             return {}

    def start_reader_task_sync(self):
        """동기 컨텍스트에서 리더 태스크를 시작합니다 (예: 서버 시작 시)."""
        
        # 이벤트 루프 얻기
        try:
             loop = asyncio.get_running_loop()
        except RuntimeError: # 루프가 없으면 새로 만듦 (이 경우는 드물어야 함)
             self.logger.warning("No running event loop found, creating a new one for start_reader_task_sync.")
             loop = asyncio.new_event_loop()
             asyncio.set_event_loop(loop)
             
        if self._reader_task is None or self._reader_task.done():
            self._is_running = True
            # Ensure _read_loop is scheduled correctly in the loop
            self._reader_task = loop.create_task(self._read_loop())
            self.logger.info(f"Scheduled reader task for MCP server '{self.name}' in the running loop.")
        else:
             self.logger.debug(f"Reader task for '{self.name}' is already running or scheduled.")

    async def ping(self) -> bool:
        """Sends a non-standard 'ping' request to check basic connectivity."""
        try:
            # Using a very short timeout for ping
            result = await self.call("ping", {}, timeout=2.0)
            # Basic check if we got any response back (assuming pong is just successful return)
            return True 
        except TimeoutError:
            self.logger.warning(f"Ping request to '{self.name}' timed out.")
            return False
        except Exception as e:
            self.logger.error(f"Error during ping to '{self.name}': {e}")
            return False

class MCPError(Exception):
    """Custom exception for MCP errors."""
    def __init__(self, error_obj: Dict[str, Any], request_id: Optional[str] = None):
        self.code = error_obj.get("code", -32000) # Default to Server error
        self.message = error_obj.get("message", "Unknown MCP error")
        self.data = error_obj.get("data")
        self.request_id = request_id
        super().__init__(f"MCP Error (ID: {request_id or 'N/A'}) - Code: {self.code}, Message: {self.message}") 