import logging
import re
import json
import asyncio
import time
import os
from typing import List, Dict, Any, Tuple, Optional, Pattern, Union
from pathlib import Path
from datetime import datetime
from app.utils.log_utils import save_meta_log # Import path corrected

# LLM inference libraries
try:
    from llama_cpp import Llama # Using llama-cpp-python
except ImportError:
    Llama = None
    logging.error("llama-cpp-python library not found. Please install it for LLM features.")

# Actual dependencies import
from app.services.mcp_service import MCPService # MCPService import
# from app.services.sequential_thinking import SequentialThinking # If needed

logger = logging.getLogger(__name__)

# Extract JSON blocks from LLM output (assuming ```json ... ``` format)
JSON_BLOCK_PATTERN: Pattern[str] = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
# Add Final Answer pattern
FINAL_ANSWER_PATTERN: Pattern[str] = re.compile(r"Final Answer:(.*)", re.DOTALL | re.IGNORECASE)

MAX_REACT_ITERATIONS: int = 10 # Consider increasing iteration count
MAX_OBSERVATION_LENGTH: int = 1500 # Maximum length for observation results

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
        mcp_service: 'MCPService',  # Inter-dependency type hint 
        model_path: Optional[Union[Path, str]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        log_dir: Optional[Path] = None,
    ):
        self.mcp_service = mcp_service
        # If a string is passed, convert it to a Path object
        self.model_path = Path(model_path) if model_path else None
        self.model_params = model_params or {}
        self.model: Optional[Llama] = None
        self.model_loaded = False
        self.log_dir = log_dir if log_dir else Path("logs")
        # Session-specific conversation history store (use session ID as key)
        self.conversation_histories: Dict[str, List[Dict[str, str]]] = {}
        # Conversation history size limit (remove old messages if too long)
        self.max_history_length = 20

        if Llama is None:
             logger.error("llama-cpp-python is not installed. LLM features will be disabled.")

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

    async def initialize_model(self):
        """Load the LLM model from the specified path."""
        if self.model_loaded or Llama is None:
            logger.info("Model already loaded or llama-cpp-python is not available.")
            return

        # Check if model_path is None or does not exist
        if not self.model_path:
            logger.error("Model path is not set")
            self.model_loaded = False
            return
            
        # If model_path is a string, convert it to a Path object
        if isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)
            
        if not self.model_path.exists():
            logger.error(f"Model path does not exist: {self.model_path}")
            self.model_loaded = False
            return

        logger.info(f"Loading LLM model from: {self.model_path}")
        try:
            n_ctx = 32768  # Qwen model's maximum context size (32K)
            n_gpu_layers = self.model_params.get("n_gpu_layers", -1) # GPU layers (-1 means as many as possible)
            n_batch = self.model_params.get("n_batch", 512) # Batch size
            # Check the effective level of the application's logging to determine verbose
            verbose = logging.getLogger().getEffectiveLevel() <= logging.DEBUG

            # Load llama-cpp-python model (I/O operations, so process asynchronously)
            self.model = await asyncio.to_thread(
                 Llama,
                 model_path=str(self.model_path),
                 n_ctx=n_ctx,
                 n_gpu_layers=n_gpu_layers,
                 n_batch=n_batch,
                 verbose=verbose,
                 chat_format="qwen",
            )

            self.model_loaded = True
            logger.info(f"LLM model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}", exc_info=True)
            self.model = None
            self.model_loaded = False

    async def generate_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate text using chat-style messages."""
        if not self.model_loaded or self.model is None:
            logger.error("LLM model is not loaded. Cannot generate text.")
            return "Error: LLM model is not available."

        try:
            # Set generation parameters
            temperature = kwargs.get("temperature", 0.2)
            max_tokens = kwargs.get("max_tokens", 4096)  # Default to 8K (1/4 of 32K)
            top_p = kwargs.get("top_p", 0.9)
            top_k = kwargs.get("top_k", 40)
            repeat_penalty = kwargs.get("repeat_penalty", 1.1)
            
            # Log messages (sensitive information may be present, so log only a part)
            logger.debug(f"Generating chat completion for {len(messages)} messages")
            if messages:
                last_message = messages[-1].get('content', '')
                logger.debug(f"Last message (first 100 chars): {last_message[:100]}...")
            
            # Use chat-style API to generate text
            start_time = time.time()
            
            try:
                completion_result = await asyncio.to_thread(
                    self.model.create_chat_completion,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stream=False
                )
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                # Extract response
                if "choices" in completion_result and completion_result["choices"]:
                    response_text = completion_result["choices"][0]["message"]["content"]
                    logger.debug(f"Generated text in {generation_time:.2f}s: {response_text[:512]}...")
                    return response_text.strip()
                else:
                    logger.error(f"Unexpected response format from llama-cpp: {completion_result}")
                    return "Error: Unexpected response format from LLM."
                    
            except ValueError as e:
                if "exceed context window" in str(e):
                    logger.error(f"Context window exceeded: {e}")
                    # Handle case where context window size of 32K is also exceeded
                    return "Final Answer: Sorry, I cannot generate a response due to the length of the conversation. Please refer to previous results."
                raise
                
        except Exception as e:
            logger.error(f"Error during chat generation: {e}", exc_info=True)
            return "Error: Could not generate chat response from LLM."

    async def shutdown_model(self):
         """Release LLM resources."""
         logger.info("Shutting down LLMInterface...") # Restore class name
         if self.model is not None:
             # llama-cpp-python may not have an explicit unload function
              # Let Python GC handle it, or use del if needed
              try:
                  del self.model
              except Exception as e:
                  logger.error(f"Error trying to delete model object: {e}")
              self.model = None
              self.model_loaded = False
              # Force memory cleanup (optional)
              # import gc
              # gc.collect()
              # if torch.cuda.is_available():
              #      torch.cuda.empty_cache()
              logger.info("LLM model resources released.")
         await asyncio.sleep(0)

    async def _cache_tool_schemas(self):
        """Pre-fetch and cache schema for all available tools."""
        logger.info("Caching tool schemas...")
        all_tools = self.mcp_service.get_all_tools()
        self._tool_schemas = {}
        for server_name, tools in all_tools.items():
            for tool_info in tools:
                tool_key = f"{server_name}.{tool_info['name']}"
                self._tool_schemas[tool_key] = tool_info.get('inputSchema', {})
        logger.info(f"Cached schemas for {len(self._tool_schemas)} tools.")

    async def _format_tools_for_prompt(self) -> Dict[str, Any]:
        """Simplify and return available tool list."""
        # Get tools directly from MCPService
        full_tools = self.mcp_service.get_all_tools() # Full tool information
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
        full_tools = self.mcp_service.get_all_tools()
        if not full_tools or server_name not in full_tools or tool_name not in full_tools.get(server_name, {}):
            return {"error": f"Tool {server_name}.{tool_name} not found"}
        
        # Return detailed information about the tool
        return full_tools[server_name][tool_name]

    def _parse_llm_action_json(self, llm_output: str, session_id: str = None) -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON-formatted actions from LLM output.
        Handle various incorrect formats as much as possible.
        
        Args:
            llm_output: LLM output text
            session_id: Session ID for logging
            
        Returns:
            Parsed action dictionary or None if parsing fails
        """
        try:
            # Extract JSON block (regular expression)
            json_match = self.json_regex.search(llm_output)
            if not json_match:
                if session_id:
                    logger.warning(f"[{session_id}] Could not find JSON block")
                return None
                
            # Try parsing JSON
            json_str = json_match.group(1).strip()
            try:
                action_json = json.loads(json_str)
            except json.JSONDecodeError as e:
                if session_id:
                    logger.warning(f"[{session_id}] Invalid JSON format: {e}")
                return None
                
            # Basic validation
            if not isinstance(action_json, dict):
                if session_id:
                    logger.warning(f"[{session_id}] JSON is not an object: {action_json}")
                return None
                
            # Check if LLM follows the correct format and modify if necessary
            action_type = action_json.get("action_type")
            tool_name = action_json.get("tool_name")
            
            # If action_type field is missing
            if action_type is None:
                # If tool_name exists, set action_type to "tool_call"
                if tool_name:
                    action_json["action_type"] = "tool_call"
                    if session_id:
                        logger.warning(f"[{session_id}] Missing action_type field, setting to 'tool_call'")
                else:
                    # Check if direct tool name and arguments are present in action_json
                    potential_tools = [k for k in action_json.keys() if k not in ["action_type", "tool_name", "arguments"]]
                    if len(potential_tools) == 1 and isinstance(action_json[potential_tools[0]], dict):
                        tool_name = potential_tools[0]
                        arguments = action_json[tool_name]
                        action_json = {
                            "action_type": "tool_call",
                            "tool_name": tool_name,
                            "arguments": arguments
                        }
                        if session_id:
                            logger.warning(f"[{session_id}] Reconstructed JSON format for: {tool_name}")
                    else:
                        if session_id:
                            logger.warning(f"[{session_id}] Both action_type and tool_name are missing")
                        return None
            
            # Handle case where action_type is tool name
            elif action_type != "tool_call" and action_type != "final_answer":
                if tool_name is None:  # tool_name is missing and action_type looks like tool name
                    # Set tool_name as action_type and change action_type to "tool_call"
                    tool_name = action_type
                    action_json["tool_name"] = tool_name
                    action_json["action_type"] = "tool_call"
                    
                    # If arguments field is missing, treat all additional keys as arguments
                    if "arguments" not in action_json:
                        arguments = {k: v for k, v in action_json.items() 
                                     if k not in ["action_type", "tool_name"]}
                        action_json["arguments"] = arguments
                        
                        # Remove original keys
                        for k in list(arguments.keys()):
                            if k in action_json:
                                del action_json[k]
                                
                    if session_id:
                        logger.warning(f"[{session_id}] Fixed tool call format: Set '{action_type}' as tool_name")
            
            # Handle case where action_type is "tool_call" but tool_name is missing
            elif action_type == "tool_call" and tool_name is None:
                if session_id:
                    logger.warning(f"[{session_id}] 'tool_call' action missing tool_name")
                return None
                
            # Handle case where arguments are not object form
            if action_type == "tool_call" and "arguments" not in action_json:
                # Treat all additional keys as arguments
                arguments = {k: v for k, v in action_json.items() 
                             if k not in ["action_type", "tool_name"]}
                
                if arguments:
                    action_json["arguments"] = arguments
                    
                    # Remove original keys
                    for k in list(arguments.keys()):
                        if k in action_json:
                            del action_json[k]
                            
                    if session_id:
                        logger.warning(f"[{session_id}] Created missing arguments object")
                else:
                    action_json["arguments"] = {}
            
            return action_json
            
        except Exception as e:
            if session_id:
                logger.error(f"[{session_id}] Error during JSON parsing: {str(e)}")
            return None

    def _format_observation(self, result: Union[str, Dict, Exception]) -> str:
        """Format the result of tool execution into a string."""
        if isinstance(result, Exception):
            error_message = str(result)
            # Extract cleaner error message if possible
            if ":" in error_message:
                error_message = error_message.split(":", 1)[1].strip()
            
            return (
                "ERROR: Tool execution failed. Please analyze this error and try a different approach.\n"
                f"Error details: {error_message}\n\n"
                "To proceed, you should:\n"
                "1. CAREFULLY review the error message above\n"
                "2. Consider an alternative tool or different parameters\n"
                "3. If the tool doesn't exist or parameters are incorrect, check your available tools list\n"
                "4. Provide a new Thought that acknowledges this error and explains your new approach"
            )

        if isinstance(result, dict):
            # Prettify JSON for better readability
            try:
                return f"Tool successfully executed. Result:\n```json\n{json.dumps(result, indent=2, ensure_ascii=False)}\n```\n\nAnalyze this result carefully for your next step."
            except Exception as e:
                return f"Tool successfully executed but result couldn't be formatted as JSON. Raw result: {str(result)}"
        
        if isinstance(result, str):
            # Format output based on length
            if len(result) > 2000:
                truncated_result = result[:2000] + "...[TRUNCATED due to length]"
                return f"Tool successfully executed. Result (truncated):\n{truncated_result}\n\nAnalyze this result carefully for your next step."
            return f"Tool successfully executed. Result:\n{result}\n\nAnalyze this result carefully for your next step."
        
        return f"Tool successfully executed. Result: {str(result)}\n\nAnalyze this result for your next step."

    # Conversation history management methods
    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Return conversation history for the session. Create if not exists."""
        if session_id not in self.conversation_histories:
            # Initialize new conversation history (include system message)
            self.conversation_histories[session_id] = [
                {"role": "system", "content": (
                    "You are an AI assistant that follows the ReAct pattern to solve tasks step by step. "
                    "Your role is to act as an intelligent agent that carefully maintains context across multiple interactions. "
                    "Follow these steps exactly:\n\n"
                    "1. **Thought:** Analyze the query, available tools, and PREVIOUS OBSERVATIONS. Plan your approach concisely.\n"
                    "2. **Action:** Choose ONE of the following formats:\n"
                    "   - Tool Call: If a tool would help, output a properly formatted JSON block:\n"
                    "     ```json\n"
                    '     {"action_type": "tool_call", "tool_name": "server-name.tool-name", "arguments": {"param1": "value1", "param2": "value2"}}\n'
                    "     ```\n"
                    "   - Final Answer: If you know the answer or no tool is needed:\n"
                    "     ```json\n"
                    '     {"action_type": "final_answer", "answer": "Your final answer to the user question"}\n'
                    "     ```\n"
                    "3. When shown an Observation, LEARN from it and adapt your next Thought accordingly.\n\n"
                    "Available tools:\n{tools_section}\n\n"
                    "IMPORTANT RULES:\n"
                    "- MAINTAIN CONTEXT: Each response should build on previous observations and thoughts\n"
                    "- LEARN FROM FAILURES: If a previous approach failed, try something different\n"
                    "- NEVER provide more than one action per response\n"
                    "- ALWAYS use exact format: Thought: [reasoning], then either JSON block or Final Answer: [answer]\n"
                    "- NEVER skip providing a Thought before your Action\n"
                    "- ALWAYS output complete JSON blocks between ```json and ``` tags\n"
                    "- DO NOT repeat the same text multiple times in your response\n\n"
                    "TOOL FORMAT REQUIREMENTS:\n"
                    '- ALWAYS use "action_type": "tool_call" for tool calls\n'
                    '- ALWAYS include "tool_name" with the full server and tool name (example: "server-name.tool-name")\n'
                    '- ALWAYS put parameters inside the "arguments" object, not at the root level\n\n'
                    "CORRECT EXAMPLES:\n"
                    "```json\n"
                    '{"action_type": "tool_call", "tool_name": "server-name.tool-name", "arguments": {"path": "/example/path"}}\n'
                    "```\n\n"
                    "```json\n"
                    '{"action_type": "tool_call", "tool_name": "another-server.execute-command", "arguments": {"command": "ls -la", "timeout_ms": 5000}}\n'
                    "```\n\n"
                    "```json\n"
                    '{"action_type": "final_answer", "answer": "Based on my investigation using the tools, I found that..."}\n'
                    "```\n\n"
                    "INCORRECT EXAMPLES (DO NOT USE THESE FORMATS):\n"
                    "```json\n"
                    '{"action_type": "sample-tool", "path": "/example/path"}\n'
                    "```\n\n"
                    "```json\n"
                    '{"tool_name": "server-name.tool-name", "arguments": {"path": "/example/path"}}\n'
                    "```"
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

    def update_system_message(self, session_id: str, tools_section: str) -> None:
        """Update the system message with tool information for the session."""
        history = self.get_conversation_history(session_id)
        
        # Update system message (include tool information)
        system_content = (
            "You are an AI assistant that follows the ReAct pattern to solve tasks step by step. "
            "Your role is to act as an intelligent agent that carefully maintains context across multiple interactions. "
            "Follow these steps exactly:\n\n"
            "1. **Thought:** Analyze the query, available tools, and PREVIOUS OBSERVATIONS. Plan your approach concisely.\n"
            "2. **Action:** Choose ONE of the following formats:\n"
            "   - Tool Call: If a tool would help, output a properly formatted JSON block:\n"
            "     ```json\n"
            '     {"action_type": "tool_call", "tool_name": "server-name.tool-name", "arguments": {"param1": "value1", "param2": "value2"}}\n'
            "     ```\n"
            "   - Final Answer: If you know the answer or no tool is needed:\n"
            "     ```json\n"
            '     {"action_type": "final_answer", "answer": "Your final answer to the user question"}\n'
            "     ```\n"
            "3. When shown an Observation, LEARN from it and adapt your next Thought accordingly.\n\n"
            f"Available tools:\n{tools_section}\n\n"
            "IMPORTANT RULES:\n"
            "- MAINTAIN CONTEXT: Each response should build on previous observations and thoughts\n"
            "- LEARN FROM FAILURES: If a previous approach failed, try something different\n"
            "- NEVER provide more than one action per response\n"
            "- ALWAYS use exact format: Thought: [reasoning], then either JSON block or Final Answer: [answer]\n"
            "- NEVER skip providing a Thought before your Action\n"
            "- ALWAYS output complete JSON blocks between ```json and ``` tags\n"
            "- DO NOT repeat the same text multiple times in your response\n\n"
            "TOOL FORMAT REQUIREMENTS:\n"
            '- ALWAYS use "action_type": "tool_call" for tool calls\n'
            '- ALWAYS include "tool_name" with the full server and tool name (example: "server-name.tool-name")\n'
            '- ALWAYS put parameters inside the "arguments" object, not at the root level\n\n'
            "CORRECT EXAMPLES:\n"
            "```json\n"
            '{"action_type": "tool_call", "tool_name": "server-name.tool-name", "arguments": {"path": "/example/path"}}\n'
            "```\n\n"
            "```json\n"
            '{"action_type": "tool_call", "tool_name": "another-server.execute-command", "arguments": {"command": "ls -la", "timeout_ms": 5000}}\n'
            "```\n\n"
            "```json\n"
            '{"action_type": "final_answer", "answer": "Based on my investigation using the tools, I found that..."}\n'
            "```\n\n"
            "INCORRECT EXAMPLES (DO NOT USE THESE FORMATS):\n"
            "```json\n"
            '{"action_type": "sample-tool", "path": "/example/path"}\n'
            "```\n\n"
            "```json\n"
            '{"tool_name": "server-name.tool-name", "arguments": {"path": "/example/path"}}\n'
            "```"
        )
        
        # Check if the first message is system message and update
        if history and history[0]["role"] == "system":
            history[0]["content"] = system_content
        else:
            # If system message does not exist, add new
            history.insert(0, {"role": "system", "content": system_content})

    async def process_react_pattern(
        self,
        initial_prompt: str,
        session_id: str,
        max_iterations: int = 3,
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
        if not self.model_loaded or self.model is None:
            await self.initialize_model()
            
        # Setup logging directory - 단일 세션 ID로 통합
        if log_dir:
            logs_path = os.path.join(log_dir, "api_logs", session_id)
        else:
            logs_path = os.path.join(self.log_dir, "api_logs", session_id)
            
        os.makedirs(logs_path, exist_ok=True)
        meta_log_path = os.path.join(logs_path, "meta.json")
        
        # 기존 메타데이터 파일이 있으면 읽기
        meta_data = []
        if os.path.exists(meta_log_path):
            try:
                with open(meta_log_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                    if not isinstance(meta_data, list):
                        meta_data = [meta_data]  # 단일 객체일 경우 리스트로 변환
            except Exception as e:
                logger.error(f"Error reading existing meta.json: {e}. Creating new file.")
                meta_data = []
        
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
        }
        
        # Get available tools and format them for the prompt
        mcp_tools = self.mcp_service.get_all_tools()
        
        # Log available tools for debugging
        logger.info(f"[ReAct] Available MCP tools: {json.dumps(mcp_tools, indent=2, ensure_ascii=False)}")
        
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
        system_message = f"""You are an AI assistant that helps users by answering questions and using tools when necessary.

When responding, ALWAYS use one of these TWO formats:

1. When you need to use a tool:
```json
{{
  "action_type": "tool_call",
  "thought": "Here I analyze the query and explain why I need to use a tool",
  "tool_name": "server-name.tool-name",
  "arguments": {{
    "param1": "value1",
    "param2": "value2"
  }}
}}
```

2. When you can answer directly:
```json
{{
  "action_type": "final_answer",
  "thought": "Here I analyze the query and explain why I can answer directly",
  "answer": "My answer to the user's question"
}}
```

Important rules:
1. Always include a detailed "thought" field explaining your reasoning
2. Remember and use information from previous observations
3. Never use tools or parameters that don't exist
4. If a tool fails, analyze the error and try a different approach
5. Respond in the same language the user initially used

{tools_section}

When using tools, always use the server-name.tool-name format and include all parameters in the arguments object."""
        
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
            
            # Generate chat response
            try:
                llm_response = await self.generate_chat(conversation_history)
                logger.info(f"[ReAct] Iteration {iteration+1} LLM Response: {llm_response}")
                iteration_data["response"] = llm_response
            except Exception as e:
                error_msg = f"Error generating LLM response: {str(e)}"
                logger.error(error_msg)
                iteration_data["error"] = error_msg
                metadata["iterations"].append(iteration_data)
                
                # meta.json에 현재 상태 저장
                meta_data.append(metadata)
                with open(meta_log_path, "w", encoding="utf-8") as f:
                    json.dump(meta_data, f, indent=2, ensure_ascii=False)
                
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
                action = self._parse_llm_action_json(llm_response, session_id)
                if not action:
                    # If we failed to parse action, create error message about format
                    raise ParsingError("JSON format is incorrect. Please provide a valid JSON.")
                    
                iteration_data["action"] = action
                logger.info(f"[ReAct] Parsed action: {action}")
            except Exception as e:
                error_msg = f"Failed to parse action: {str(e)}"
                logger.error(error_msg)
                
                # Add error message to the history to guide model
                error_feedback = (
                    f"ERROR: {str(e)}\n\n"
                    f"I couldn't parse your response correctly. Please respond ONLY in the format specified:\n\n"
                    f"For tool calls:\n```json\n{{\"action_type\": \"tool_call\", \"tool_name\": \"server-name.tool-name\", \"arguments\": {{...}}}}\n```\n\n"
                    f"For final answers:\n```json\n{{\"action_type\": \"final_answer\", \"answer\": \"...\"}}\n```\n\n"
                    f"Previous response: {llm_response}\n\n"
                    f"Please try again with the correct format."
                )
                
                conversation_history.append({"role": "user", "content": error_feedback})
                iteration_data["error"] = error_msg
                metadata["iterations"].append(iteration_data)
                error_count += 1
                
                # 중간 상태 저장
                meta_data.append(metadata)
                with open(meta_log_path, "w", encoding="utf-8") as f:
                    json.dump(meta_data, f, indent=2, ensure_ascii=False)
                
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
                    
                    # Check for same tool repeated with error
                    if f"{server_name}.{actual_tool_name}" == last_tool_name and last_error is not None:
                        consecutive_same_tool_error += 1
                    else:
                        consecutive_same_tool_error = 0
                        last_tool_name = f"{server_name}.{actual_tool_name}"
                else:
                    error_feedback = (
                        f"ERROR: Invalid tool name format: {tool_name}. Must be in format 'server-name.tool-name'\n\n"
                        f"Please use the correct format for tool calls with server name included."
                    )
                    conversation_history.append({"role": "user", "content": error_feedback})
                    iteration_data["error"] = "Invalid tool name format"
                    metadata["iterations"].append(iteration_data)
                    
                    # 중간 상태 저장
                    meta_data.append(metadata)
                    with open(meta_log_path, "w", encoding="utf-8") as f:
                        json.dump(meta_data, f, indent=2, ensure_ascii=False)
                    
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
                    
                    # 중간 상태 저장
                    meta_data.append(metadata)
                    with open(meta_log_path, "w", encoding="utf-8") as f:
                        json.dump(meta_data, f, indent=2, ensure_ascii=False)
                    
                    # Reset counter and skip to next iteration
                    consecutive_same_tool_error = 0
                    last_error = None
                    continue
                
                # Execute the tool
                observation = None
                try:
                    logger.info(f"[ReAct] Executing tool '{server_name}.{actual_tool_name}' with arguments: {arguments}")
                    
                    # Get detailed tool information first to help the model understand how to use it
                    tool_details = self._get_detailed_tool_info(server_name, actual_tool_name)
                    tool_details_message = f"Detailed information about this tool before execution:\n\n{tool_details}"
                    conversation_history.append({"role": "user", "content": tool_details_message})
                    
                    # Execute tool via MCP service
                    tool_result = await self.mcp_service.execute_tool(server_name, actual_tool_name, arguments)
                    observation = self._format_observation(tool_result)
                    last_error = None  # Reset error state on success
                    
                    logger.info(f"[ReAct] Tool execution result: {observation[:200]}...")
                except Exception as e:
                    logger.error(f"[ReAct] Tool execution error: {str(e)}")
                    observation = self._format_observation(e)
                    last_error = str(e)
                    error_count += 1
                
                # Add observation to conversation history
                if observation:
                    observation_message = (
                        f"Observation from tool call {server_name}.{actual_tool_name}:\n"
                        f"{observation}\n\n"
                        "Based on this observation, continue your reasoning and decide on the next action."
                    )
                    conversation_history.append({"role": "user", "content": observation_message})
                    iteration_data["observation"] = observation
            
            elif action.get("action_type") == "final_answer":
                final_answer = action.get("answer", "")
                thought = action.get("thought", "")
                
                # Check if the answer or thought mentions using a tool but didn't actually call it
                tool_references = []
                for server_name, tools in mcp_tools.items():
                    for tool_name in tools.keys():
                        # Look for tool mentions in thought or answer
                        if (isinstance(thought, str) and tool_name.lower() in thought.lower()) or \
                           (isinstance(final_answer, str) and tool_name.lower() in final_answer.lower()):
                            tool_references.append(f"{server_name}.{tool_name}")
                
                # If the AI referenced tools without using them properly, provide feedback
                if tool_references and not isinstance(final_answer, dict):
                    error_feedback = (
                        f"It seems you mentioned using the following tools: {', '.join(tool_references)}, "
                        f"but you provided a final answer without using them. "
                        f"If you need to use a tool, please use the 'tool_call' action format instead of 'final_answer'."
                    )
                    conversation_history.append({"role": "user", "content": error_feedback})
                    iteration_data["error"] = "Tool mentioned but not used properly"
                    metadata["iterations"].append(iteration_data)
                    continue
                    
                # Check if the final answer contains a data structure that should have come from a tool call
                if isinstance(final_answer, dict) and len(final_answer) > 0:
                    # This looks like structured data that should have come from a tool
                    # Check if any tool was actually called in this session
                    tool_was_used = False
                    for iter_data in metadata["iterations"]:
                        if iter_data.get("observation") is not None:
                            tool_was_used = True
                            break
                    
                    if not tool_was_used:
                        logger.warning("[ReAct] Detected potential hallucinated data in final answer without tool usage")
                        error_feedback = (
                            "I notice you're providing structured data in your answer, but you haven't used any tools to retrieve this information. "
                            "If you need to get real data about files, directories, or other system information, please use the appropriate tool first, "
                            "then provide your answer based on the tool's response."
                        )
                        conversation_history.append({"role": "user", "content": error_feedback})
                        iteration_data["error"] = "Hallucinated structured data detected"
                        metadata["iterations"].append(iteration_data)
                        continue
                    
                logger.info(f"[ReAct] Final answer: {final_answer}")
                metadata["final_response"] = final_answer
                metadata["iterations"].append(iteration_data)
                break
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
            
            # 각 반복 후 중간 상태 저장
            meta_data.append(metadata)
            with open(meta_log_path, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2, ensure_ascii=False)
            
            # Trim conversation history if it gets too long
            if len(conversation_history) > self.max_history_length:
                # Keep the system message and the most recent messages
                conversation_history = [conversation_history[0]] + conversation_history[-(self.max_history_length-1):]
        
        # If we reached max iterations without a final answer, use the last response
        if final_answer is None:
            final_answer = "I'm sorry, I couldn't find an appropriate response. Please make your question more specific or try asking in a different way."
            logger.warning("[ReAct] Reached max iterations without final answer")
            metadata["final_response"] = final_answer
        
        # 최종 상태 저장
        meta_data.append(metadata)
        with open(meta_log_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)
        
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
        if not self.model_loaded or self.model is None:
            logger.error("LLM model is not loaded. Cannot generate text.")
            return "Error: LLM model is not available."

        logger.debug(f"Generating text for prompt (last 500 chars): ...{prompt[-500:]}")
        start_time = time.time()

        # Use chat-style API to generate text (convert single user message)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Use chat-style API
            return await self.generate_chat(messages, **kwargs)
        except Exception as e:
            logger.error(f"Error during text generation: {e}", exc_info=True)
            return "Error: Could not generate text from LLM."

    # More detailed tool information for when a tool is selected
    def _get_detailed_tool_info(self, server_name: str, tool_name: str) -> str:
        """Get detailed information about a specific tool including schema and examples."""
        all_tools = self.mcp_service.get_all_tools()
        
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