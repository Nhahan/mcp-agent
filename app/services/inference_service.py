import logging
import re
import json
import asyncio
import os
import random
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime

from app.services.mcp_service import MCPService
from app.services.model_service import ModelService, ModelError
from app.utils.log_utils import async_save_meta_log

logger = logging.getLogger(__name__)

class InferenceError(Exception):
    """Specific errors occurring during inference process"""
    pass

class ToolInputValidationError(InferenceError):
    """Error when tool input validation fails"""
    pass

class ParsingError(InferenceError):
    """Error when LLM response parsing fails"""
    pass

class InferenceService:
    def __init__(
        self,
        mcp_manager: MCPService,
        model_path: str,
        model_params: Dict[str, Any] = None,
    ):
        self.mcp_manager = mcp_manager  # MCP manager instance
        
        # ModelService initialization
        self.model_service = ModelService(
            model_path=model_path,
            model_params=model_params
        )
        
        # Model load status flag for checking
        self.model_loaded = False
        
        # Conversation management attributes
        self.conversation_histories = {}  # Conversation history by session
        self.max_history_length = 50  # Maximum conversation history length
            
        # Logging directory setting
        self.log_dir = None
        self._tools_schema_cache = {}  # Tool schema cache
        
        logger.info("InferenceService initialized with MCPService")

        # Regular expression patterns for parsing
        self.json_regex = re.compile(r'```json\s*([\s\S]*?)\s*```')  # Pattern to extract JSON blocks
        self.final_answer_pattern = re.compile(r'final answer:?\s*(.*)', re.IGNORECASE)  # Pattern to extract Final Answer
        
        # Tool schema caching
        self.tool_schemas = {}

    def set_log_directory(self, log_dir: Path):
        """Set the log directory."""
        self.log_dir = log_dir
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)

    async def init_model(self):
        """Initialize the model."""
        try:
            result = await self.model_service.load_model()
            self.model_loaded = result
            return result
        except ModelError as e:
            logger.critical(f"Model initialization failed: {e}")
            raise InferenceError(str(e)) from e

    async def initialize_model(self):
        """
        init_model method wrapper.
        External API called from main.py and other places.
        """
        return await self.init_model()

    async def generate_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate text using chat-style messages."""
        if not self.model_loaded:
            logger.error("LLM model is not loaded. Cannot generate text.")
            return "Error: LLM model is not available."

        try:
            return await self.model_service.generate_chat(messages, **kwargs)
        except Exception as e:
            logger.error(f"Error during chat generation: {e}", exc_info=True)
            return "Error: Could not generate chat response from LLM."

    async def shutdown_model(self):
        """Release LLM resources."""
        logger.info("Shutting down LLMInterface...")
        try:
            await self.model_service.shutdown()
            self.model_loaded = False
            logger.info("LLM model resources released successfully.")
        except Exception as e:
            logger.error(f"Error during model shutdown: {e}", exc_info=True)
            
        await asyncio.sleep(0)

    async def _cache_tool_schemas(self):
        """Pre-fetch and cache schema for all available tools."""
        logger.info("Caching tool schemas...")
        all_tools = self.mcp_manager.get_all_tools()
        self._tool_schemas = {}
        for server_name, tools in all_tools.items():
            for tool_info in tools:
                tool_key = f"{server_name}.{tool_info['name']}"
                self._tool_schemas[tool_key] = tool_info.get('inputSchema', {})
        logger.info(f"Cached schemas for {len(self._tool_schemas)} tools.")

    async def _format_tools_for_prompt(self) -> Dict[str, Any]:
        """Simplify and return available tool list."""
        # Get tools directly from MCP service
        full_tools = self.mcp_manager.get_all_tools() # Full tool information
        if not full_tools:
            logger.warning("No tools available from MCP service.")
            return {} # Return empty dictionary
        
        # Create dictionary with simplified tool information
        simplified_tools = {}
        for server_name, tools in full_tools.items():
            simplified_tools[server_name] = {}
            for tool_name, tool_info in tools.items():
                # Include only tool name and description
                simplified_tools[server_name][tool_name] = {
                    "name": tool_name,
                    "description": tool_info.get("description", "No description available.")
                    # Exclude detailed information such as input schema
                }
        
        # Return simplified tool information
        return simplified_tools
    
    async def _get_tool_details(self, server_name: str, tool_name: str) -> Dict[str, Any]:
        """Return detailed information about a specific tool."""
        full_tools = self.mcp_manager.get_all_tools()
        if not full_tools or server_name not in full_tools or tool_name not in full_tools.get(server_name, {}):
            return {"error": f"Tool {server_name}.{tool_name} not found"}
        
        # Return detailed information about the tool
        return full_tools[server_name][tool_name]

    def _parse_llm_action_json(self, llm_output: str, session_id: str = None, iteration: int = None) -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON-formatted actions from LLM output.
        Handle various incorrect formats as much as possible.
        
        Args:
            llm_output: LLM output text
            session_id: Session ID for logging
            iteration: Current iteration number (to handle first iteration differently)
            
        Returns:
            Parsed action dictionary or None if parsing fails
        """
        try:
            # 디버깅을 위한 원본 출력 로깅
            if session_id:
                logger.debug(f"[{session_id}] Raw LLM output for parsing: {llm_output[:200]}...")
            
            # JSON 직접 추출 시도 - 가장 간단한 방법부터 시도
            try:
                # 전체 텍스트가 직접 JSON인지 확인
                action_json = json.loads(llm_output.strip())
                if isinstance(action_json, dict):
                    if session_id:
                        logger.info(f"[{session_id}] Successfully parsed direct JSON format")
                    return action_json
            except json.JSONDecodeError:
                # 직접 파싱 실패, 계속 진행
                pass
                
            # JSON 블록 추출 (정규식 개선)
            # 1. ```json ... ``` 형식 우선 시도
            json_blocks = re.findall(r'```(?:json)?\s*\n?(.*?)\n?```', llm_output, re.DOTALL)
            
            # 2. 마커 없는 JSON 객체 추출 시도
            if not json_blocks:
                # 중괄호로 감싸진 텍스트 블록 찾기
                json_blocks = re.findall(r'(\{.*?\})', llm_output, re.DOTALL)
                if session_id and json_blocks:
                    logger.info(f"[{session_id}] Found JSON blocks without markers: {len(json_blocks)}")
            
            if not json_blocks:
                if session_id:
                    logger.warning(f"[{session_id}] Could not find any JSON blocks. Trying to extract from full text.")
                # 전체 텍스트에서 중괄호 패턴을 한 번 더 찾아봄
                json_pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})'
                json_blocks = re.findall(json_pattern, llm_output, re.DOTALL)
                
            if not json_blocks:
                if session_id:
                    logger.warning(f"[{session_id}] Could not find any JSON blocks")
                return None
            
            # Error handling when multiple JSON blocks are found - only one JSON block is allowed
            if len(json_blocks) > 1:
                if session_id:
                    logger.warning(f"[{session_id}] Multiple JSON blocks found ({len(json_blocks)})")
                
                # 첫 번째 블록을 시도
                first_json_str = json_blocks[0].strip()
                try:
                    action_json = json.loads(first_json_str)
                    if session_id:
                        logger.info(f"[{session_id}] Using first JSON block from multiple blocks")
                    return action_json
                except json.JSONDecodeError:
                    pass
                    
                # 모든 블록을 시도해보고 파싱되는 첫 번째 블록 사용
                for i, block in enumerate(json_blocks):
                    try:
                        json_str = block.strip()
                        action_json = json.loads(json_str)
                        if isinstance(action_json, dict):
                            if session_id:
                                logger.info(f"[{session_id}] Successfully parsed JSON block {i+1} of {len(json_blocks)}")
                            return action_json
                    except:
                        continue
                
                # 모든 블록이 파싱 실패하면 오류 반환
                return {
                    "action_type": "error",
                    "error": f"Multiple JSON blocks found ({len(json_blocks)}). Please provide exactly ONE JSON block.",
                    "original_response": llm_output[:300] + "..." if len(llm_output) > 300 else llm_output
                }
            
            # Process single JSON block
            json_str = json_blocks[0].strip()
            
            # 향상된 JSON 정리 과정
            # 1. 일반적인 이스케이프 문제 처리
            cleaned_json_str = json_str
            
            if r'\$' in cleaned_json_str:
                cleaned_json_str = cleaned_json_str.replace(r'\$', '$')
            
            # 2. 속성 이름이 따옴표로 둘러싸여 있지 않은 경우 수정
            try:
                # 속성 이름 뒤에 콜론이 오는 패턴을 찾아 따옴표로 감싸기
                # 예: thought: -> "thought":
                property_regex = re.compile(r'(\n\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)')
                cleaned_json_str = property_regex.sub(r'\1"\2"\3', cleaned_json_str)
                
                # JSON 문자열 처음에 있는 속성도 처리
                start_property_regex = re.compile(r'^{\s*([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)')
                cleaned_json_str = start_property_regex.sub(r'{ "\1"\2', cleaned_json_str)
                
                if cleaned_json_str != json_str:
                    logger.info(f"[{session_id}] Fixed unquoted property names in JSON")
            except Exception as e:
                logger.warning(f"[{session_id}] Error while fixing property names: {str(e)}")
                # 실패해도 계속 진행
                
            # 3. answer 필드에서 줄바꿈 및 특수 문자 처리
            try:
                # 정규 표현식으로 answer 필드 찾기
                answer_regex = re.compile(r'"answer"\s*:\s*"(.*?)(?<!\\)"(?=,|\s*})', re.DOTALL)
                answer_match = answer_regex.search(cleaned_json_str)
                
                if answer_match:
                    # answer 필드 내용 가져오기
                    answer_content = answer_match.group(1)
                    
                    # 개행 문자를 공백으로 변환
                    sanitized_answer = answer_content.replace('\n', ' ').replace('\r', ' ')
                    
                    # 연속된 공백을 하나로 합치기
                    sanitized_answer = re.sub(r'\s+', ' ', sanitized_answer)
                    
                    # JSON에서 문제가 될 수 있는 문자 이스케이프 처리
                    sanitized_answer = sanitized_answer.replace('"', '\\"')
                    
                    # 정리된 answer로 교체
                    cleaned_json_str = cleaned_json_str[:answer_match.start(1)] + sanitized_answer + cleaned_json_str[answer_match.end(1):]
                    
                    if sanitized_answer != answer_content:
                        logger.info(f"[{session_id}] Sanitized answer field content for better JSON parsing")
            except Exception as e:
                logger.warning(f"[{session_id}] Error during answer field sanitization: {str(e)}")
                # 실패해도 계속 진행
            
            try:
                # 다양한 방법으로 파싱 시도
                json_parsing_methods = [
                    lambda: json.loads(json_str),  # 원본 텍스트
                    lambda: json.loads(cleaned_json_str),  # 기본 정리된 텍스트
                    lambda: json.loads(re.sub(r'([\{\,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned_json_str)),  # 강력한 속성 이름 수정
                    lambda: json.loads(re.sub(r'[\n\r\t]', ' ', cleaned_json_str)),  # 모든 줄바꿈 제거
                    lambda: json.loads(re.sub(r'\\([^"])', r'\1', cleaned_json_str)),  # 불필요한 이스케이프 제거
                ]
                
                action_json = None
                last_error = None
                
                # 모든 방법 시도
                for i, parse_method in enumerate(json_parsing_methods):
                    try:
                        action_json = parse_method()
                        if session_id:
                            logger.info(f"[{session_id}] Successfully parsed JSON using method {i+1}")
                        break
                    except json.JSONDecodeError as e:
                        last_error = e
                        continue
                        
                if action_json is None:
                    if session_id:
                        logger.warning(f"[{session_id}] All JSON parsing methods failed: {last_error}")
                    return None
                    
                # answer 필드가 지나치게 길면 잘라내기
                if "answer" in action_json and isinstance(action_json["answer"], str) and len(action_json["answer"]) > 1000:
                    action_json["answer"] = action_json["answer"][:997] + "..."
                    logger.info(f"[{session_id}] Truncated overly long answer field to 1000 characters")
                
                # Basic validation
                if not isinstance(action_json, dict):
                    if session_id:
                        logger.warning(f"[{session_id}] JSON is not an object")
                    return None
                
                # Check if LLM follows the correct format and modify if necessary
                action_type = action_json.get("action_type")
                tool_name = action_json.get("tool_name")
                
                # 기본 필드 존재 여부 확인 및 자동 보정 기능 제거
                if action_type is None:
                    # 자동 보정하지 않고 오류 반환
                    if session_id:
                        logger.warning(f"[{session_id}] Missing required action_type field")
                    return {
                        "action_type": "error",
                        "error": "Missing required 'action_type' field. Must be 'tool_call' or 'final_answer'.",
                        "original_response": json_str[:300] + "..." if len(json_str) > 300 else json_str
                    }
                
                # 결과 반환
                return action_json
                
            except Exception as e:
                if session_id:
                    logger.warning(f"[{session_id}] Error during JSON parsing: {str(e)}")
                return None
                
        except Exception as e:
            if session_id:
                logger.error(f"[{session_id}] Unexpected error during action parsing: {str(e)}", exc_info=True)
            return None

    def _format_observation(self, result: Union[str, Dict, Exception]) -> str:
        """Format the result of tool execution into a string."""
        if isinstance(result, Exception):
            error_message = str(result)
            # Extract cleaner error message if possible
            if ":" in error_message:
                error_message = error_message.split(":", 1)[1].strip()
            
            return (
                "The tool execution resulted in an exception.\n"
                f"Details: {error_message}\n\n"
                "Analyze this message to determine what went wrong and how to proceed."
            )

        if isinstance(result, dict):
            # Prettify JSON for better readability
            try:
                # Format the result as JSON string
                json_str = json.dumps(result, indent=2, ensure_ascii=False)
                
                # Return the result without making assumptions about success/failure
                return f"Tool execution completed. Result:\n```json\n{json_str}\n```\n\nAnalyze this result carefully to determine your next step."
            except Exception as e:
                return f"Tool execution completed but result couldn't be formatted as JSON. Raw result: {str(result)}"
        
        if isinstance(result, str):
            # 파일 내용이나 코드 같은 텍스트는 코드 블록 안에 표시
            # 파일 내용 패턴 감지 (일반적인 파일 내용은 여러 줄 텍스트이거나 특정 패턴을 가짐)
            is_file_content = '\n' in result or result.strip().startswith('<!DOCTYPE') or result.strip().startswith('<?xml')
            
            # 코드로 보이는 패턴 감지
            is_code = any(pattern in result for pattern in ['function', 'class', 'def ', 'import ', '<html', '<?php', '// ', '/* ', '#include'])
            
            # 파일 내용이나 코드로 판단되면 코드 블록으로 표시
            if is_file_content or is_code:
                # 코드 언어 추측
                lang = ""
                if result.strip().startswith('<!DOCTYPE') or '<html' in result:
                    lang = "html"
                elif result.strip().startswith('<?xml'):
                    lang = "xml"
                elif any(pattern in result for pattern in ['function', 'const ', 'var ', '// ']):
                    lang = "javascript"
                elif any(pattern in result for pattern in ['def ', 'import ', '# ']):
                    lang = "python"
                elif any(pattern in result for pattern in ['<?php']):
                    lang = "php"
                elif any(pattern in result for pattern in ['#include', 'int main']):
                    lang = "cpp"
                
                # Format output based on length
                if len(result) > 2000:
                    truncated_result = result[:2000] + "...[TRUNCATED due to length]"
                    return f"Tool execution completed. Result (truncated):\n```{lang}\n{truncated_result}\n```\n\nAnalyze this result carefully for your next step."
                return f"Tool execution completed. Result:\n```{lang}\n{result}\n```\n\nAnalyze this result carefully for your next step."
            else:
                # 일반 텍스트는 기존 방식대로 처리
                if len(result) > 2000:
                    truncated_result = result[:2000] + "...[TRUNCATED due to length]"
                    return f"Tool execution completed. Result (truncated):\n{truncated_result}\n\nAnalyze this result carefully for your next step."
                return f"Tool execution completed. Result:\n{result}\n\nAnalyze this result carefully for your next step."

    # Conversation history management methods
    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Return conversation history for the session. Create if not exists."""
        if session_id not in self.conversation_histories:
            # Initialize new conversation history (include system message)
            self.conversation_histories[session_id] = [
                {"role": "system", "content": (
                    "You are a highly intelligent AI assistant with the ability to think step-by-step. Before giving your final answer, carefully analyze the question, break it down into logical steps, and consider multiple perspectives. Then, present your final, concise answer. by answering questions and using tools."
                    "\n\nFORMATS - Always use exactly ONE of these:"
                    "\n\n1. Tool Call (when you need a tool):"
                    "\n```json"
                    '\n{\n  "action_type": "tool_call",\n  "thought": "Why I need this tool",\n  "tool_name": "tool-name-with-server-prefix",\n  "arguments": { "param1": "value1" }\n}'
                    "\n```"
                    "\n\n2. Final Answer (only when task is complete):"
                    "\n```json"
                    '\n{\n  "action_type": "final_answer",\n  "thought": "My reasoning based on all information",\n  "answer": "Complete answer to user\'s question"\n}'
                    "\n```"
                    "\n\nCRITICAL RULES:"
                    "\n• ONE JSON BLOCK PER RESPONSE - never multiple blocks"
                    "\n• ONE STEP AT A TIME - wait for each tool result before proceeding"
                    "\n• NEVER assume previous steps were completed without confirmation"
                    "\n• NEVER skip steps or provide final answer until all necessary tools are used"
                    "\n• AVOID SPECIAL CHARACTERS in JSON values - no shell commands like $(date), no backslashes"
                    "\n• AVOID USING QUOTES WITHIN VALUES - use simple text for arguments"
                    "\n• USE BASIC TOOLS for file operations instead of executing shell commands"
                )}
            ]
        return self.conversation_histories[session_id]

    def add_to_conversation(self, session_id: str, role: str, content: str) -> None:
        """Add new message to conversation history."""
        history = self.get_conversation_history(session_id)
        history.append({"role": role, "content": content})
        
        # Manage conversation history size (system message is always kept)
        if len(history) > self.max_history_length + 1:  # +1 for system message
            # Remove oldest 2 messages (user and assistant message pair)
            history.pop(1)  # Oldest user message
            if len(history) > self.max_history_length:
                history.pop(1)  # Oldest assistant message
                
        # Update saved history
        self.conversation_histories[session_id] = history

    def update_system_message(self, role_intro=None, language=None, errors=None):
        """Update system message with optional role intro and errors."""
        if role_intro is None:
            role_intro = "You are an AI assistant that helps with any task. You answer questions and use tools to complete tasks."
            if language != "ko":
                role_intro = "You are an AI assistant that helps with any task. You answer questions and use tools to complete tasks."
                
        # ... existing code ...

    async def process_react_pattern(
        self,
        initial_prompt: str,
        session_id: str,
        max_iterations: int = 10,
        log_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a chat using ReAct pattern.
        
        Args:
            initial_prompt: The input text from the user.
            session_id: A unique identifier for the chat session.
            max_iterations: Maximum number of ReAct pattern iterations.
            log_dir: Directory to store logs. If not provided, logs will be stored in the default directory.
            
        Returns:
            A dictionary containing the final response and metadata.
        """
        if not self.model_loaded:
            await self.initialize_model()
            
        # Setup logging directory - 단일 세션 ID로 통합
        if log_dir:
            logs_path = Path(log_dir) / "api_logs" / session_id
        else:
            logs_path = Path(self.log_dir) / "api_logs" / session_id
            
        logs_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "event_type": "react_process",
            "initial_prompt": initial_prompt,
            "iterations": [],
            "final_response": "",
            "iteration_count": 0,
            "max_iterations": max_iterations,
            "model": {
                "path": self.model_service.model_path,
                "params": self.model_service.model_params
            }
        }
        
        # Get available tools and format them for the prompt
        mcp_tools = self.mcp_manager.get_all_tools()
        
        # Generate a formatted tools section for the prompt
        tools_section = "Available Tools:\n"
        if mcp_tools:
            for server_name, server_tools in mcp_tools.items():
                tools_section += f"\nServer: {server_name}\n"
                for tool_name, tool_info in server_tools.items():
                    # Provide only tool names without detailed descriptions initially
                    tools_section += f"- {tool_name}\n"
                tools_section += "\nTo use a tool, specify its name with the server prefix (e.g., 'server-name.tool-name').\nDetailed description for each tool will be provided when you decide to use it."
        else:
            tools_section += "No tools are currently available."
            
        # Log the simplified tools section
        logger.info(f"[ReAct] Simplified tools section for prompt: \n{tools_section}")
        
        # Initialize ReAct format with tools section
        system_message = f"""You are an AI assistant that helps users by answering questions and using tools.

FORMATS - Always use exactly ONE of these:

1. Tool Call (when you need a tool):
```json
{{
  "action_type": "tool_call",
  "thought": "Your detailed step-by-step reasoning including ALL your analysis and plans",
  "tool_name": "server-name.tool-name", 
  "arguments": {{ "param1": "value1" }}
}}
```

2. Final Answer (only when task is complete):
```json
{{
  "action_type": "final_answer",
  "thought": "Your detailed reasoning including what you tried and what you learned",
  "answer": "Complete answer to user's question"
}}
```

⚠️ EXTREMELY CRITICAL RULES:
• YOU MUST OUTPUT EXACTLY ONE JSON BLOCK PER RESPONSE - NOT TWO, NOT THREE, JUST ONE
• NEVER INCLUDE BOTH A TOOL CALL AND FINAL ANSWER IN THE SAME RESPONSE
• CHOOSE EITHER TOOL CALL OR FINAL ANSWER, NEVER BOTH
• ALL REASONING GOES INSIDE THE "thought" FIELD, NOT OUTSIDE THE JSON
• NEVER WRITE TEXT LIKE "**Final Answer**" OUTSIDE THE JSON BLOCK
• NEVER write any explanatory text, reasoning, or planning outside the JSON block
• PUT ALL YOUR THINKING IN THE "thought" FIELD - IT CAN BE AS LONG AS NEEDED
• DO NOT REPEAT YOURSELF IN AND OUTSIDE THE JSON - ALL TEXT GOES INSIDE THE JSON

CRITICAL JSON FORMATTING RULES:
• STRICT JSON FORMAT IS REQUIRED - Your entire response MUST be valid parseable JSON
• ALL PROPERTY NAMES MUST BE IN DOUBLE QUOTES - e.g., "property": not property:
• NEVER use single quotes for property names - only double quotes allowed for keys
• ALWAYS use double quotes around ALL keys in the JSON, including "thought" and "answer"
• Keep your "answer" field SHORT and CONCISE - Maximum 500 characters recommended
• AVOID numbered lists (1., 2., 3.) in JSON values - use commas or bullet points instead
• DO NOT use NEWLINES (\\n) in the "answer" field - provide a continuous paragraph
• AVOID special characters in JSON - no quotes within quotes, no backslashes
• ESCAPE any necessary quotes in values with a backslash: \\"
• For long responses, put brief conclusion in "answer" field and details in "thought" field
• If you need to provide a structured response, use simple markdown without complex formatting

{tools_section}

Always use server prefix for tool names (e.g., "server-name.tool-name")
Carefully analyze tool results in context to determine your next steps."""
        
        # Initialize or update the conversation history for ReAct
        if session_id in self.conversation_histories:
            # Update the system message with tools information
            if self.conversation_histories[session_id] and self.conversation_histories[session_id][0]["role"] == "system":
                self.conversation_histories[session_id][0]["content"] = system_message
            else:
                self.conversation_histories[session_id].insert(0, {"role": "system", "content": system_message})
        else:
            # Initialize new conversation history
            self.conversation_histories[session_id] = [
                {"role": "system", "content": system_message}
            ]
        
        # Get the updated conversation history
        conversation_history = self.get_conversation_history(session_id)
        
        # Add user's initial prompt if not already in history
        if len(conversation_history) <= 1:  # Only has system message
            self.add_to_conversation(session_id, "user", initial_prompt)
            conversation_history = self.get_conversation_history(session_id)
        
        # Record all iterations for logging
        iterations_data = []
        
        final_answer = None
        error_count = 0
        consecutive_same_tool_error = 0
        last_tool_name = None
        last_error = None
        
        # Main ReAct loop
        for iteration in range(max_iterations):
            metadata["iteration_count"] = iteration + 1
            iteration_data = {
                "iteration": iteration + 1, 
                "timestamp": datetime.now().isoformat(),
                "prompt": conversation_history[-1]["content"] if conversation_history else initial_prompt,
                "response": None,
                "action": None,
                "observation": None,
                "error": None,
            }
            
            # 연속 같은 도구 사용 실패 감지
            repeated_tool_failures = {}
            if 'repeated_tool_failures' in metadata:
                repeated_tool_failures = metadata['repeated_tool_failures']
            else:
                metadata['repeated_tool_failures'] = repeated_tool_failures
            
            # Generate chat response
            try:
                llm_response = await self.model_service.generate_chat(conversation_history)
                logger.info(f"[ReAct] Iteration {iteration+1} LLM Response: {llm_response}")
                iteration_data["response"] = llm_response
            except Exception as e:
                error_msg = f"Error generating LLM response: {str(e)}"
                logger.error(error_msg)
                iteration_data["error"] = error_msg
                metadata["iterations"].append(iteration_data)
                
                # Save current state to meta.json
                await async_save_meta_log(Path(logs_path), session_id, metadata)
                
                return {
                    "response": "I'm sorry, AI couldn't generate a response. Please try again later.",
                    "metadata": metadata
                }
            
            # Add LLM response to conversation history
            conversation_history.append({"role": "assistant", "content": llm_response})
            
            # Optionally provide tool details if the LLM is trying to use a tool (proactive approach)
            # await self._provide_tool_details_when_needed(conversation_history, llm_response)
            
            # Parse the LLM's response to identify the action
            action = None
            try:
                action = self._parse_llm_action_json(llm_response, session_id, iteration)
                if not action:
                    # If we failed to parse action, create error message about format
                    raise ParsingError("JSON format is incorrect. Please provide a valid JSON.")
                    
                iteration_data["action"] = action
                logger.info(f"[ReAct] Parsed action: {action}")
                
                # Check if this is the first iteration with multiple JSON blocks warning
                if "_multiple_blocks_warning" in action and iteration == 0:
                    # Remove the warning flag before proceeding
                    del action["_multiple_blocks_warning"]
                    
                    # Add warning message to the conversation
                    warning_message = (
                        "WARNING: I noticed you provided multiple JSON blocks in your response. "
                        "It's good to plan ahead, but you can only execute ONE action at a time. "
                        "For this time, I'll use just the first JSON block you provided. "
                        "Going forward, please include EXACTLY ONE JSON block per response.\n\n"
                        "Remember to put all your thoughts and reasoning inside the 'thought' field of a single JSON block."
                    )
                    conversation_history.append({"role": "user", "content": warning_message})
                    logger.info(f"[ReAct] Added multiple blocks warning for first iteration")
            except Exception as e:
                error_msg = f"Failed to parse action: {str(e)}"
                logger.error(error_msg)
                
                # Add error message to the history to guide model
                error_feedback = (
                    f"ERROR: {error_msg}\n\n"
                    f"I noticed you provided multiple JSON blocks or invalid JSON format. Please remember:\n\n"
                    f"1. Your ENTIRE response should contain EXACTLY ONE JSON block\n"
                    f"2. Put ALL your reasoning in the 'thought' field - do not write separate thoughts outside the JSON\n"
                    f"3. Choose ONE tool to use - do not suggest multiple alternatives\n\n"
                    f"Example of correct format:\n```json\n"
                    f'{{"action_type": "tool_call", "thought": "I need to do X because of Y. First I\'ll try this approach. If that fails, I might need to try another approach later.", "tool_name": "server-name.tool-name", "arguments": {{"param1": "value1"}} }}\n'
                    f"```\n\n"
                    f"OR\n\n```json\n"
                    f'{{"action_type": "final_answer", "thought": "Based on all the steps I\'ve taken and the results I\'ve seen, I\'ve determined that...", "answer": "The answer to your question is..."}}\n'
                    f"```"
                )
                
                conversation_history.append({"role": "user", "content": error_feedback})
                iteration_data["error"] = error_msg
                metadata["iterations"].append(iteration_data)
                error_count += 1
                
                # Save current state to meta.json
                await async_save_meta_log(Path(logs_path), session_id, metadata)
                
                # Skip to next iteration
                continue
            
            # Process the action based on its type
            if action.get("action_type") == "tool_call":
                tool_name = action.get("tool_name", "")
                arguments = action.get("arguments", {})
                
                # Parse server_name and actual tool_name from the full tool_name
                server_name = None
                actual_tool_name = None
                
                if tool_name and "." in tool_name:
                    # If tool_name contains a dot, split it to get server_name and actual_tool_name
                    parts = tool_name.split(".", 1)
                    server_name = parts[0]
                    actual_tool_name = parts[1]
                    
                    # 도구 실행 전에 도구의 정확한 파라미터 정보를 가져와 제공
                    tool_details = self._get_detailed_tool_info(server_name, actual_tool_name)
                    tool_schema = await self._get_tool_schema(server_name, actual_tool_name)
                    
                    # 해당 도구의 연속 실패 횟수 확인
                    tool_key = f"{server_name}.{actual_tool_name}"
                    if tool_key in repeated_tool_failures:
                        repeated_tool_failures[tool_key] += 1
                    else:
                        repeated_tool_failures[tool_key] = 1
                    
                    # 연속 실패가 3회 이상인 경우 AI에게 다른 접근 방식 제안
                    if repeated_tool_failures.get(tool_key, 0) >= 3:
                        logger.warning(f"[ReAct] Tool '{tool_key}' has failed repeatedly ({repeated_tool_failures[tool_key]} times)")
                        reset_message = (
                            f"I notice you've tried using '{tool_key}' multiple times with the same parameters, but it's not working.\n\n"
                            f"Required parameters for this tool: {self._format_required_params(tool_schema)}\n\n"
                            f"Current parameters you're using: {json.dumps(arguments, indent=2)}\n\n"
                            f"Consider one of these approaches:\n"
                            f"1. Try a completely different tool\n"
                            f"2. Provide a final answer based on what you know so far\n"
                            f"3. Use the correct parameter names as shown above\n\n"
                            f"If you continue using this tool, please make significant changes to your approach."
                        )
                        conversation_history.append({"role": "user", "content": reset_message})
                        iteration_data["error"] = f"Repeated tool failure: {tool_key}"
                        metadata["iterations"].append(iteration_data)
                        
                        # 실패 카운터 리셋
                        repeated_tool_failures[tool_key] = 0
                        
                        # Save current state to meta.json
                        await async_save_meta_log(Path(logs_path), session_id, metadata)
                        continue
                    
                    if tool_schema:
                        # 파라미터 불일치 감지 및 자동 수정
                        fixed_arguments = arguments.copy()  # 원본 인수 복사만 하고 수정 없이 사용
                        
                        # 파라미터가 수정되었다면 AI에게 알림
                        if fixed_arguments != arguments:
                            param_diff = []
                            for k in set(list(arguments.keys()) + list(fixed_arguments.keys())):
                                if k not in arguments:
                                    param_diff.append(f"Added '{k}'")
                                elif k not in fixed_arguments:
                                    param_diff.append(f"Removed '{k}'")
                                elif arguments[k] != fixed_arguments[k]:
                                    param_diff.append(f"Changed '{k}'")
                            
                            param_changes = ", ".join(param_diff)
                            logger.info(f"[ReAct] Fixed tool arguments: {param_changes}")
                            
                            # 변경 내용을 AI에게 알림
                            info_message = (
                                f"NOTE: I've adjusted the parameters for '{server_name}.{actual_tool_name}' based on its schema.\n"
                                f"Changes: {param_changes}\n\n"
                                f"Required parameters for this tool: {self._format_required_params(tool_schema)}\n\n"
                                f"Proceeding with corrected parameters."
                            )
                            conversation_history.append({"role": "user", "content": info_message})
                            arguments = fixed_arguments
                    
                    # Validate and fix tool arguments
                    is_valid, validation_message, fixed_arguments = await self._validate_tool_arguments(server_name, actual_tool_name, arguments)
                    
                    # Check for same tool repeated with error
                    if f"{server_name}.{actual_tool_name}" == last_tool_name and last_error is not None:
                        consecutive_same_tool_error += 1
                    else:
                        consecutive_same_tool_error = 0
                        last_tool_name = f"{server_name}.{actual_tool_name}"
                else:
                    error_feedback = (
                        f"ERROR: Invalid tool name format: {tool_name}. Must include server name.\n\n"
                        f"Please use the correct format for tool calls with server name included."
                    )
                    conversation_history.append({"role": "user", "content": error_feedback})
                    iteration_data["error"] = "Invalid tool name format"
                    metadata["iterations"].append(iteration_data)
                    
                    # Save current state to meta.json
                    await async_save_meta_log(Path(logs_path), session_id, metadata)
                    
                    continue
                
                # If the same tool fails more than twice, provide a stronger hint
                if consecutive_same_tool_error >= 2:
                    error_feedback = (
                        f"ERROR: You've tried the same tool '{server_name}.{actual_tool_name}' multiple times with errors.\n\n"
                        f"Last error: {last_error}\n\n"
                        f"Please try a completely different approach or tool, or if you have enough information, provide a final answer."
                    )
                    conversation_history.append({"role": "user", "content": error_feedback})
                    iteration_data["error"] = error_feedback
                    metadata["iterations"].append(iteration_data)
                    
                    # Save current state to meta.json
                    await async_save_meta_log(Path(logs_path), session_id, metadata)
                    
                    # Reset counter and skip to next iteration
                    consecutive_same_tool_error = 0
                    last_error = None
                    continue
                
                # Execute the tool
                observation = None
                try:
                    # Get detailed tool information first to help the model understand how to use it
                    tool_details = self._get_detailed_tool_info(server_name, actual_tool_name)
                    tool_details_message = (
                        f"Before executing tool '{server_name}.{actual_tool_name}', here's important information about it:\n\n"
                        f"{tool_details}\n\n"
                        f"Required parameters: {self._format_required_params(tool_schema)}"
                    )
                    conversation_history.append({"role": "user", "content": tool_details_message})
                    
                    # Execute tool via MCP service
                    tool_result = await self.mcp_manager.execute_tool(server_name, actual_tool_name, arguments)
                    observation = self._format_observation(tool_result)
                    
                    # 도구 성공 시 연속 실패 카운터 리셋
                    if tool_key in repeated_tool_failures:
                        repeated_tool_failures[tool_key] = 0
                    
                    logger.info(f"[ReAct] Tool execution result: {observation[:200]}...")
                except Exception as e:
                    logger.error(f"[ReAct] Tool execution error: {str(e)}")
                    observation = self._format_observation(e)
                    error_count += 1
                
                # Add observation to conversation history
                if observation:
                    observation_message = (
                        f"Result from tool call {server_name}.{actual_tool_name}:\n"
                        f"{observation}\n\n"
                        "Carefully analyze this complete result to determine if the tool executed successfully and what information you can extract from it."
                    )
                    conversation_history.append({"role": "user", "content": observation_message})
                    iteration_data["observation"] = observation
            
            elif action.get("action_type") == "final_answer":
                final_answer = action.get("answer", "")
                thought = action.get("thought", "")
                
                # Add debug logging
                logger.info(f"[ReAct] Final answer detected: {final_answer[:100]}...")
                
                # Return final answer immediately
                metadata["final_response"] = final_answer
                metadata["iterations"].append(iteration_data)
                break
                
            elif action.get("action_type") == "error":
                # Handle parsing errors like multiple JSON blocks
                error_msg = action.get("error", "Unknown error during JSON parsing")
                original_response = action.get("original_response", "")
                details = action.get("details", {})
                
                # More specific and helpful error messages
                if "JSON parsing error" in error_msg:
                    # Specialized message for JSON parsing errors
                    problematic_chars = details.get("problematic_chars", [])
                    error_location = details.get("error_location", "")
                    hint = details.get("hint", "")
                    
                    error_feedback = (
                        f"ERROR: {error_msg}\n\n"
                        f"There was a problem with JSON parsing. Please check the following issues:\n"
                        + (f"- Problematic characters: {', '.join(problematic_chars)}\n" if problematic_chars else "")
                        + (f"- {error_location}\n\n" if error_location else "\n")
                        + f"How to fix it:\n"
                        + f"1. Start with simple tools for file operations\n"
                        + f"2. Avoid special characters (backslash, dollar sign, quotes) in your command\n"
                        + f"3. Use simple arguments only\n\n"
                        + f"Example:\n```json\n"
                        + f'{{"action_type": "tool_call", "tool_name": "server_name.tool_name", "arguments": {{"path": "/Users/user/document.txt", "content": "Content to write"}} }}\n'
                        + f"```\n\n"
                        + f"OR\n\n```json\n"
                        + f'{{"action_type": "final_answer", "answer": "After completing all necessary tool usage, the result is..."}}\n'
                        + f"```"
                    )
                else:
                    # General error message
                    error_feedback = (
                        f"ERROR: {error_msg}\n\n"
                        f"CRITICAL: Submit EXACTLY ONE JSON block using proper format for tool calls. Do not use curly braces {{}}.\n\n"
                        f"To make a tool call, your response should contain EXACTLY ONE block in this format:\n"
                        + f"```json\n"
                        + f'{{"action_type": "tool_call", "tool_name": "server_name.tool_name", "arguments": {{"arg1": "value1", "arg2": "value2"}} }}\n'
                        + f"```\n\n"
                        + f"OR\n\n```json\n"
                        + f'{{"action_type": "final_answer", "answer": "After completing all necessary tool usage, the result is..."}}\n'
                        + f"```"
                    )
                
                conversation_history.append({"role": "user", "content": error_feedback})
                iteration_data["error"] = error_msg
                metadata["iterations"].append(iteration_data)
                logger.warning(f"[ReAct] Action parsing error: {error_msg}")
                continue
                
            else:
                # Unknown action type
                error_feedback = (
                    f"ERROR: Unknown action type '{action.get('action_type')}'.\n\n"
                    f"Please use only 'tool_call' or 'final_answer' as the action_type value."
                )
                conversation_history.append({"role": "user", "content": error_feedback})
                iteration_data["error"] = "Unknown action type"
                metadata["iterations"].append(iteration_data)
                continue
            
            # Add iteration data to the list
            metadata["iterations"].append(iteration_data)
            
            # Save intermediate state after each iteration
            await async_save_meta_log(Path(logs_path), session_id, metadata)
            
            # Trim conversation history if it gets too long
            if len(conversation_history) > self.max_history_length:
                # Keep the system message and the most recent messages
                conversation_history = [conversation_history[0]] + conversation_history[-(self.max_history_length-1):]
        
        # If we reached max iterations without a final answer, use the last response
        if final_answer is None:
            final_answer = "I'm sorry, I couldn't find an appropriate response. Please make your question more specific or try asking in a different way."
            logger.warning("[ReAct] Reached max iterations without final answer")
            metadata["final_response"] = final_answer
        
        # Save final state
        await async_save_meta_log(Path(logs_path), session_id, metadata)
        
        # Return the final result
        return {
            "response": final_answer,
            "metadata": metadata
        }

    def _format_full_response(self, iterations_history: List[Dict[str, Any]]) -> str:
        """Format the iterations history into a readable full response."""
        full_response = []
        
        for iteration in iterations_history:
            response = iteration.get("response", "")
            observation = iteration.get("observation", "")
            
            full_response.append(response)
            if observation:
                full_response.append(f"Observation: {observation}")
        
        return "\n\n".join(full_response)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using LLM. (Text prompt style)"""
        if not self.model_loaded:
            logger.error("LLM model is not loaded. Cannot generate text.")
            return "Error: LLM model is not available."

        logger.debug(f"Generating text for prompt (last 500 chars): ...{prompt[-500:]}")
        
        try:
            # ModelService의 generate_text 메서드 사용
            return await self.model_service.generate_text(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Error during text generation: {e}", exc_info=True)
            return "Error: Could not generate text from LLM."

    # More detailed tool information for when a tool is selected
    def _get_detailed_tool_info(self, server_name: str, tool_name: str) -> str:
        """Get detailed information about a specific tool including schema and examples."""
        all_tools = self.mcp_manager.get_all_tools()
        
        if not all_tools or server_name not in all_tools or tool_name not in all_tools.get(server_name, {}):
            return f"Tool information not found: {server_name}.{tool_name}"
        
        tool_info = all_tools[server_name][tool_name]
        description = tool_info.get("description", "No description available")
        
        # Get input schema if available
        input_schema = tool_info.get("inputSchema", {})
        schema_str = json.dumps(input_schema, indent=2, ensure_ascii=False) if input_schema else "No schema information available"
        
        # Format detailed information including examples if available
        result = f"Tool: {server_name}.{tool_name}\nDescription: {description}\n\nInput Schema:\n```json\n{schema_str}\n```\n\n"
        
        # Add examples if available
        examples = tool_info.get("examples", [])
        if examples:
            result += "Usage Examples:\n"
            for i, example in enumerate(examples, 1):
                example_str = json.dumps(example, indent=2, ensure_ascii=False)
                result += f"Example {i}:\n```json\n{example_str}\n```\n\n"
        
        return result

    # Add a new method to provide tool details when needed
    async def _provide_tool_details_when_needed(self, conversation_history: List[Dict[str, str]], llm_response: str) -> None:
        """Analyze LLM response to see if it's trying to use a tool, and if so, provide details."""
        # Extract JSON from response
        json_match = self.json_regex.search(llm_response)
        if not json_match:
            return
        
        try:
            # Parse JSON to check if it's a tool call
            json_str = json_match.group(1).strip()
            action_json = json.loads(json_str)
            
            # Check if it's a tool call
            if action_json.get("action_type") == "tool_call":
                tool_name = action_json.get("tool_name", "")
                
                # Parse server_name and actual_tool_name
                if tool_name and "." in tool_name:
                    parts = tool_name.split(".", 1)
                    server_name = parts[0]
                    actual_tool_name = parts[1]
                    
                    # Get detailed tool information
                    tool_details = self._get_detailed_tool_info(server_name, actual_tool_name)
                    
                    # Add tool details to conversation history
                    tool_details_message = f"Here is detailed information about this tool. Please review the parameters and format:\n\n{tool_details}"
                    conversation_history.append({"role": "user", "content": tool_details_message})
                    
                    logger.info(f"[ReAct] Provided detailed information for tool: {tool_name}")
        except Exception as e:
            logger.warning(f"[ReAct] Error providing tool details: {str(e)}")
            # Continue without providing details 

    async def _validate_tool_arguments(self, server_name: str, tool_name: str, arguments: Dict) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Validate tool arguments against schema."""
        try:
            # 1. 도구 스키마 가져오기
            schema = await self._get_tool_schema(server_name, tool_name)
            if not schema:
                return False, None, f"Schema not found for tool '{server_name}.{tool_name}'"
                
            # 2. 스키마 검증 전에 인수 수정 시도
            fixed_args = arguments.copy()  # 원본 인수 복사만 하고 수정 없이 사용
                
            # 3. 필수 파라미터 확인
            missing_params = []
            required_params = schema.get("required", [])
            properties = schema.get("properties", {})
            
            for param in required_params:
                if param not in fixed_args or fixed_args[param] is None:
                    missing_params.append(param)
                    
            if missing_params:
                # 오류 메시지 생성
                error_msg = f"Missing required parameter(s) for '{server_name}.{tool_name}': {', '.join(missing_params)}\n\n"
                error_msg += "Required parameters:\n"
                for param in required_params:
                    param_info = properties.get(param, {})
                    param_type = param_info.get("type", "string")
                    param_desc = param_info.get("description", "No description available")
                    error_msg += f"- {param}: {param_type} - {param_desc}\n"
                
                return False, error_msg, None
                
            # 4. 타입 검증 (간단한 검증)
            # TODO: 더 상세한 JSON 스키마 검증 구현
            
            return True, None, fixed_args
            
        except Exception as e:
            logger.error(f"Error validating tool arguments: {str(e)}")
            return False, f"Error validating arguments: {str(e)}", None

    def _process_tool_validation_error(
        self, 
        server_name: str, 
        tool_name: str, 
        error_msg: str, 
        conversation: List[Dict],
        metadata: Dict
    ) -> None:
        """Process tool validation error and add helpful feedback."""
        logger.warning(f"Tool validation error for {server_name}.{tool_name}: {error_msg}")
        
        # 도구 상세 설명 가져오기
        tool_details = self._get_detailed_tool_info(server_name, tool_name)
        
        # JSON 파싱 에러 처리
        if "Expecting property name" in error_msg or "Invalid JSON" in error_msg:
            feedback = (
                f"Invalid JSON format was used for tool call. "
                f"All property names and string values within the JSON block must be wrapped in double quotes (\")."
            )
            
        # 요구되는 파라미터 누락 처리
        elif "Missing required parameter" in error_msg:
            feedback = error_msg  # 이미 상세한 메시지가 생성됨
            
        # 기타 검증 오류
        else:
            feedback = error_msg
        
        # 도구 설명을 피드백에 추가
        feedback += f"\n\n{tool_details}\n\n"
        feedback += "위 도구 정보를 참고하여 올바른 매개변수로 다시 시도해보세요."
        
        # 피드백을 관찰 결과로 대화에 추가
        self._add_observation_to_conversation(feedback, conversation)
        
        # 메타데이터 업데이트
        if "errors" not in metadata:
            metadata["errors"] = []
        metadata["errors"].append({
            "error_type": "tool_validation",
            "tool": f"{server_name}.{tool_name}",
            "message": error_msg
        })

    async def _get_tool_schema(self, server_name: str, tool_name: str) -> Optional[Dict]:
        """Get the schema for a specific tool."""
        try:
            # 모든 도구 가져오기
            all_tools = self.mcp_manager.get_all_tools()
            logger.debug(f"Getting tool schema for '{server_name}.{tool_name}'. Available servers: {list(all_tools.keys())}")
            
            # 서버 및 도구 존재 여부 확인
            if server_name not in all_tools:
                logger.warning(f"Server '{server_name}' not found in available tools")
                return None
                
            if tool_name not in all_tools[server_name]:
                logger.warning(f"Tool '{tool_name}' not found in server '{server_name}'")
                return None
                
            # 도구 정보 가져오기
            tool_info = all_tools[server_name][tool_name]
            logger.debug(f"Tool info keys for '{server_name}.{tool_name}': {list(tool_info.keys())}")
            
            # 입력 스키마 가져오기 (inputSchema 필드)
            schema = tool_info.get("inputSchema", {})
            if not schema:
                logger.warning(f"No inputSchema found for '{server_name}.{tool_name}'. Available keys: {list(tool_info.keys())}")
            else:
                logger.debug(f"Found schema for '{server_name}.{tool_name}' with properties: {list(schema.get('properties', {}).keys())}")
            return schema
            
        except Exception as e:
            logger.error(f"Error getting tool schema: {str(e)}", exc_info=True)
            return None
            
    def _format_required_params(self, schema: Dict) -> str:
        """Format required parameters from schema."""
        if not schema or not isinstance(schema, dict):
            return "Schema information not available"
        
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        
        param_info = []
        for param in required:
            param_type = properties.get(param, {}).get('type', 'unknown')
            param_desc = properties.get(param, {}).get('description', '')
            param_info.append(f"'{param}' ({param_type}){': ' + param_desc if param_desc else ''}")
        
        return ", ".join(param_info) if param_info else "No required parameters"
    
    def _add_observation_to_conversation(self, observation: str, conversation: List[Dict]) -> None:
        """Add an observation as an assistant message to the conversation."""
        observation_msg = {
            "role": "assistant",
            "content": f"Observation: {observation}"
        }
        conversation.append(observation_msg) 