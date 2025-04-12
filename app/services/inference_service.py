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

# Helper for JSON serialization in _format_observation
def default_serializer(obj):
    """Fallback serializer for JSON dumping."""
    return str(obj)

class InferenceService:
    def __init__(
        self,
        mcp_manager: MCPService,
        model_service: ModelService,
    ):
        self.mcp_manager = mcp_manager
        self.model_service = model_service
        
        # Model load status flag for checking
        self.model_loaded = False
        
        # Conversation management attributes
        self.conversation_histories = {}  # Conversation history by session
        self.max_history_length = 50  # Maximum conversation history length
            
        # Logging directory setting
        self.log_dir = None
        self._tools_schema_cache = {}  # Tool schema cache
        
        logger.info(f"InferenceService initialized with MCPService and ModelService (model path: {self.model_service.model_path})")

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

    def _parse_llm_action_json(self, llm_output: str, session_id: str = None, iteration: int = None) -> Optional[Dict[str, Any]]:
        """Parse the LLM's response to extract the action JSON."""
        if session_id:
            logger.debug(f"[{session_id}] Raw LLM output for parsing: {llm_output[:500]}{'...' if len(llm_output) > 500 else ''}")

        try:
            # Extract potential JSON block
            json_blocks = re.findall(r'```(?:json)?\s*\n?(.*?)\n?```', llm_output, re.DOTALL)

            # Check for multiple JSON blocks
            if len(json_blocks) > 1:
                if session_id:
                    logger.warning(f"[{session_id}] Multiple JSON blocks found ({len(json_blocks)}). Returning error immediately.")
                # Return an error action immediately
                return {
                    "action_type": "error",
                    "error": f"Parsing Error: Multiple JSON blocks found ({len(json_blocks)}). You MUST provide exactly ONE JSON block in your entire response, enclosed in ```json ... ```.",
                    "original_response": llm_output[:300] + "..." if len(llm_output) > 300 else llm_output
                }

            if len(json_blocks) == 1:
                json_str = json_blocks[0].strip()
                if not json_str:
                    if session_id:
                         logger.warning(f"[{session_id}] Found an empty JSON block.")
                    return None # Treat empty block as no valid JSON

                # --- JSON Cleaning Stages ---
                # 1. Basic cleaning (comments, trailing commas, whitespace)
                cleaned_json_str_basic = re.sub(r'^//.*?\n', '', json_str, flags=re.MULTILINE)
                cleaned_json_str_basic = re.sub(r'\s*/\*.*?\*/', '', cleaned_json_str_basic, flags=re.DOTALL)
                cleaned_json_str_basic = re.sub(r',\s*([\}\]])', r'\1', cleaned_json_str_basic)
                cleaned_json_str_basic = cleaned_json_str_basic.strip()

                # 2. Clean control characters within string values
                #    This regex finds strings ("...") and replaces \n, \r, \t within them
                def clean_control_chars_in_strings(match):
                    string_content = match.group(1)
                    # Replace \n, \r, \t with spaces within the matched string content
                    cleaned_content = re.sub(r'[\n\r\t]', ' ', string_content)
                    # Replace multiple spaces with a single space
                    cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
                    return f'"{cleaned_content}"' # Return the cleaned string with quotes

                try:
                     # Apply the cleaning only within quoted strings
                     # This regex is complex: it tries to correctly handle escaped quotes (\") inside strings.
                     # Pattern: " ( (?: [^"\\] | \\. )* ) "
                     cleaned_json_str_controls = re.sub(r'"((?:[^"\\]|\\.)*)"', clean_control_chars_in_strings, cleaned_json_str_basic)
                     if cleaned_json_str_controls != cleaned_json_str_basic:
                          logger.info(f"[{session_id}] Attempted to clean control characters within JSON string values.")
                     else:
                          cleaned_json_str_controls = cleaned_json_str_basic

                except Exception as regex_err:
                     logger.warning(f"[{session_id}] Regex error during control character cleaning: {regex_err}. Using basic cleaned string.")
                     cleaned_json_str_controls = cleaned_json_str_basic


                # --- JSON Parsing Attempts ---
                json_parsing_methods = [
                    # Try the most aggressively cleaned string first (controls cleaned)
                    lambda: json.loads(cleaned_json_str_controls),
                    # Fallback to basic cleaned string
                    lambda: json.loads(cleaned_json_str_basic),
                    # Fallback to original extracted string
                    lambda: json.loads(json_str),
                    # Fallback: replace all newlines globally (might break structure, last resort)
                    lambda: json.loads(re.sub(r'[\n\r\t]', ' ', cleaned_json_str_basic)),
                    # Fallback: remove unnecessary escapes globally (might break structure, last resort)
                    # lambda: json.loads(re.sub(r'\\([^\"])', r'\\1', cleaned_json_str_basic)), # Potentially risky, keep commented
                ]

                action_json = None
                last_error = None

                for i, parse_method in enumerate(json_parsing_methods):
                    try:
                        action_json = parse_method()
                        if session_id:
                            logger.info(f"[{session_id}] Successfully parsed JSON using method {i+1}")
                        break
                    except json.JSONDecodeError as e:
                        last_error = e
                        # Log which method failed for debugging
                        logger.debug(f"[{session_id}] JSON parsing method {i+1} failed: {e}")
                        continue
                    except Exception as e: # Catch other potential errors from lambda
                         last_error = e
                         logger.error(f"[{session_id}] Unexpected error during JSON parsing method {i+1}: {e}", exc_info=True)
                         continue


                if action_json is None:
                    if session_id:
                        logger.warning(f"[{session_id}] All JSON parsing methods failed: {last_error}. Original text: {json_str[:200]}...")
                    # 파싱 실패 시 에러 액션 반환
                    return {
                        "action_type": "error",
                        "error": f"JSON Parsing Error: Failed to parse the JSON block after multiple attempts. Last error: {last_error}. Ensure JSON is valid, especially check for unescaped control characters (like newlines) within string values.",
                        "original_response": json_str[:300] + "..." if len(json_str) > 300 else json_str
                    }

                # answer 필드가 지나치게 길면 잘라내기
                if "answer" in action_json and isinstance(action_json["answer"], str) and len(action_json["answer"]) > 1000:
                    action_json["answer"] = action_json["answer"][:997] + "..."
                    if session_id:
                        logger.info(f"[{session_id}] Truncated overly long answer field to 1000 characters")

                # Basic validation
                if not isinstance(action_json, dict):
                    if session_id:
                        logger.warning(f"[{session_id}] Parsed JSON is not an object/dictionary.")
                    # 유효하지 않은 JSON 형태 에러 반환
                    return {
                        "action_type": "error",
                        "error": "Invalid JSON Format: The parsed content is not a valid JSON object (dictionary).",
                        "original_response": json_str[:300] + "..." if len(json_str) > 300 else json_str
                    }

                # Check if LLM follows the correct format
                action_type = action_json.get("action_type")
                if action_type not in ["tool_call", "final_answer"]:
                    if session_id:
                        logger.warning(f"[{session_id}] Missing or invalid 'action_type'. Must be 'tool_call' or 'final_answer'. Found: {action_type}")
                    return {
                        "action_type": "error",
                        "error": f"Invalid Action Type: Missing or invalid 'action_type'. It must be either 'tool_call' or 'final_answer'. Found: '{action_type}'.",
                        "original_response": json_str[:300] + "..." if len(json_str) > 300 else json_str
                    }

                # 결과 반환
                return action_json

            # Handle case with zero blocks
            else: # len(json_blocks) == 0
                if session_id:
                    logger.warning(f"[{session_id}] No JSON block found in the response.")
                # 블록이 없을 경우, LLM이 JSON 형식을 따르지 않은 것으로 간주하고 에러 반환
                return {
                    "action_type": "error",
                    "error": "Formatting Error: No JSON block found in your response. You MUST provide your response (tool call or final answer) inside a single ```json ... ``` block.",
                    "original_response": llm_output[:300] + "..." if len(llm_output) > 300 else llm_output
                }

        except Exception as e:
            # Catch any unexpected errors during the parsing process
            if session_id:
                logger.error(f"[{session_id}] Unexpected error during action parsing: {str(e)}", exc_info=True)
            return {
                "action_type": "error",
                "error": f"Internal Server Error: An unexpected error occurred during response parsing: {str(e)}",
                "original_response": llm_output[:300] + "..." if len(llm_output) > 300 else llm_output
            }

    def _format_observation(self, result: Union[str, Dict, Exception]) -> str:
        """Format the result of tool execution into a simple string representation."""
        try:
            if isinstance(result, Exception):
                # Extract just the core error message
                error_message = str(result)
                # Attempt to get a cleaner message after the first colon if present
                if ":" in error_message:
                     parts = error_message.split(":", 1)
                     if len(parts) > 1 and parts[1].strip():
                           error_message = parts[1].strip()
                # Prefix with ERROR: for clarity
                return f"ERROR: {error_message}"

            elif isinstance(result, dict):
                # Directly return the JSON string representation
                try:
                    # Use default_serializer for objects that aren't directly serializable
                    return json.dumps(result, indent=2, ensure_ascii=False, default=default_serializer)
                except Exception as e:
                    logger.warning(f"Failed to serialize dict result to JSON: {e}")
                    # Fallback to simple string representation
                    return str(result)

            elif isinstance(result, str):
                # Return the string directly, truncate if excessively long
                if len(result) > 3000: # Increased limit slightly
                    return result[:3000] + "...[TRUNCATED]"
                return result
            else:
                # For any other type, return its string representation
                 return str(result)

        except Exception as e:
            # Catch any unexpected error during formatting itself
            logger.error(f"Error during _format_observation: {str(e)}", exc_info=True)
            return f"ERROR: Failed to format the observation result ({type(result).__name__})."

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

    async def process_react_pattern(
        self,
        initial_prompt: str,
        session_id: str,
        session_log_dir: Path,
        max_iterations: int = 10,
    ) -> Dict[str, Any]:
        """
        Process a chat using ReAct pattern.
        
        Args:
            initial_prompt: The input text from the user.
            session_id: A unique identifier for the chat session.
            session_log_dir: Directory to store logs for the session.
            max_iterations: Maximum number of ReAct pattern iterations.
            
        Returns:
            A dictionary containing the final response and metadata.
        """
        if not self.model_loaded:
            logger.warning("Model not loaded. Initializing...")
            initialized = await self.init_model()
            if not initialized:
                logger.error("Failed to initialize model during ReAct processing.")
                return {"response": "Error: Model initialization failed.", "error": "Model init failed"}

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
                    tools_section += f"- {server_name}.{tool_name}\n"
            tools_section += "\nTo use a tool, specify its full name including the server prefix (e.g., 'server-name.tool-name')."
            tools_section += "\nDetailed descriptions for tools will be provided if needed."
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
  "thought": "(Your detailed step-by-step reasoning IN ENGLISH including analysis and plans)",
  "tool_name": "your_server_name.your_tool_name",
  "arguments": {{ "param1": "value1" }}
}}
```

2. Final Answer (only when task is complete):
```json
{{
  "action_type": "final_answer",
  "thought": "(Your detailed reasoning IN ENGLISH including what you tried and learned)",
  "answer": "(Complete answer to user's question IN ENGLISH)"
}}
```

⚠️ EXTREMELY CRITICAL RULES:
• **ALL YOUR OUTPUT MUST BE IN ENGLISH.** This includes the 'thought', 'tool_name', 'arguments', and 'answer' fields.
• YOU MUST OUTPUT EXACTLY ONE JSON BLOCK PER RESPONSE - NOT TWO, NOT THREE, JUST ONE.
• NEVER INCLUDE BOTH A TOOL CALL AND FINAL ANSWER IN THE SAME RESPONSE.
• CHOOSE EITHER TOOL CALL OR FINAL ANSWER, NEVER BOTH.
• ALL REASONING GOES INSIDE THE "thought" FIELD (IN ENGLISH), NOT OUTSIDE THE JSON.
• NEVER WRITE TEXT LIKE "**Final Answer**" OUTSIDE THE JSON BLOCK.
• NEVER write any explanatory text, reasoning, or planning outside the JSON block.
• PUT ALL YOUR THINKING IN THE "thought" FIELD (IN ENGLISH) - IT CAN BE AS LONG AS NEEDED.
• DO NOT REPEAT YOURSELF IN AND OUTSIDE THE JSON - ALL TEXT GOES INSIDE THE JSON.
• **CAREFULLY MATCH ARGUMENT NAMES**: Meticulously check the tool's 'Input Schema Properties' and 'Required Parameters' in the feedback. Ensure the argument names you use in the `arguments` object **EXACTLY** match the names specified in the schema's `properties`. Do not invent new argument names.

CRITICAL JSON FORMATTING RULES:
• STRICT JSON FORMAT IS REQUIRED - Your entire response MUST be valid parseable JSON.
• ALL PROPERTY NAMES MUST BE IN DOUBLE QUOTES - e.g., "property": not property:.
• NEVER use single quotes for property names - only double quotes allowed for keys.
• ALWAYS use double quotes around ALL keys in the JSON, including "thought" and "answer".
• Keep your "answer" field SHORT and CONCISE - Maximum 500 characters recommended.
• AVOID numbered lists (1., 2., 3.) in JSON values - use commas or bullet points instead.
• DO NOT use NEWLINES (\n) in the "answer" field - provide a continuous paragraph.
• AVOID special characters in JSON - no quotes within quotes, no backslashes.
• ESCAPE any necessary quotes in values with a backslash: \\".
• For long responses, put brief conclusion in "answer" field and details in "thought" field.
• If you need to provide a structured response, use simple markdown without complex formatting.
• **NEVER embed raw JSON strings, complex structures, raw file content, or unescaped quotes/control characters directly inside your "thought" or "answer" string values.** Summarize or rephrase the information as plain text if needed. Make sure the final JSON is always valid.

{tools_section}

Always use the full server prefix for tool names (example format: 'your_server_name.your_tool_name').
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
        elif conversation_history[-1]["role"] != "user":
             # If the last message wasn't from the user, add the current prompt
             # This handles cases where a tool was just used
             self.add_to_conversation(session_id, "user", initial_prompt)
             conversation_history = self.get_conversation_history(session_id)
            # else: # If last message was user, assume it's part of the current turn? Or replace?
            # logger.debug(f"Last message was already user for {session_id}. Not re-adding initial prompt.")
            # Let's assume the API call always represents a new user turn for now.
            # If this behavior needs change (e.g. allow multiple user messages), this logic needs update.
            # For now, we add the prompt only if the history is short or last message != user.

        # Record all iterations for logging
        iterations_data = []
        
        final_answer = None
        error_count = 0
        consecutive_same_tool_error = 0
        last_tool_name = None
        last_error = None
        
        # Maintain failure counts across iterations for specific tool + argument combos
        repeated_failures = {} # Key: (tool_key, frozenset(arguments.items())), Value: count

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
                "tool_name_requested": None,
                "arguments_provided": None,
                "schema_found": None,
                "schema_content": None,
                "validation_result": None,
                "validation_skipped_reason": None,
            }
            
            # Generate chat response
            try:
                logger.info(f"[ReAct Info] Iteration {iteration+1} Conversation History: {conversation_history}")
                llm_response = await self.model_service.generate_chat(conversation_history)
                logger.info(f"[ReAct] Iteration {iteration+1} LLM Response: {llm_response}")
                iteration_data["response"] = llm_response
            except Exception as e:
                error_msg = f"Error generating LLM response: {str(e)}"
                logger.error(error_msg)
                iteration_data["error"] = error_msg
                metadata["iterations"].append(iteration_data)
                
                # Log this iteration event
                log_event_data = {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "react_iteration",
                    "session_id": session_id,
                    "iteration_data": iteration_data
                }
                await async_save_meta_log(session_log_dir, session_id, log_event_data)
                
                return {
                    "response": "I'm sorry, AI couldn't generate a response. Please try again later.",
                    "metadata": metadata
                }
            
            # Add LLM response to conversation history
            conversation_history.append({"role": "assistant", "content": llm_response})
            
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
                    f"CRITICAL ERROR: Your response was not valid JSON. Common mistakes include:\n"
                    f"- Missing commas between elements.\n"
                    f"- Unescaped quotes (\") or control characters (like newlines) inside string values (e.g., in 'thought' or 'answer'). **Do not put raw file content or other complex text directly into string fields without proper escaping or summarization.**\n"
                    f"- Providing multiple JSON blocks instead of ONE.\n\n"
                    f"Please carefully correct the JSON format and ensure it contains EXACTLY ONE valid JSON block.\n\n"
                    f"Example format:\n```json\n"
                    f'{{"action_type": "tool_call", "thought": "I need to do X because of Y...", "tool_name": "server-name.tool-name", "arguments": {{"param1": "value1"}} }}\n'
                    f"```\n\n"
                    f"OR\n\n```json\n"
                    f'{{"action_type": "final_answer", "thought": "Based on all the steps...", "answer": "The answer is..."}}\n'
                    f"```"
                )
                
                conversation_history.append({"role": "user", "content": error_feedback})
                iteration_data["error"] = error_msg
                metadata["iterations"].append(iteration_data)
                error_count += 1
                
                # Log this iteration event
                log_event_data = {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "react_iteration",
                    "session_id": session_id,
                    "iteration_data": iteration_data
                }
                await async_save_meta_log(session_log_dir, session_id, log_event_data)
                
                # Skip to next iteration
                continue
            
            # Process the action based on its type
            if action.get("action_type") == "tool_call":
                tool_name = action.get("tool_name", "")
                arguments = action.get("arguments", {})
                arguments_key = frozenset(arguments.items()) # Immutable key for dict

                iteration_data["tool_name_requested"] = tool_name
                iteration_data["arguments_provided"] = arguments

                server_name = None
                actual_tool_name = None
                tool_key = None

                # --- Handle final_answer provided as tool_name ---
                if tool_name == "final_answer":
                    error_feedback = "ERROR: Invalid action. Use 'action_type': 'final_answer' for final answers, not 'tool_name': 'final_answer'."
                    conversation_history.append({"role": "user", "content": error_feedback})
                    iteration_data["error"] = "Invalid action: final_answer used as tool_name"
                    metadata["iterations"].append(iteration_data)
                    # Log iteration event
                    log_event_data = {
                        "timestamp": datetime.now().isoformat(),
                        "event_type": "react_iteration",
                        "session_id": session_id,
                        "iteration_data": iteration_data
                    }
                    await async_save_meta_log(session_log_dir, session_id, log_event_data)
                    continue

                if tool_name and "." in tool_name:
                    parts = tool_name.split(".", 1)
                    server_name = parts[0]
                    actual_tool_name = parts[1]
                    tool_key = f"{server_name}.{actual_tool_name}"

                    # --- Get Tool Schema and Log ---
                    tool_schema = await self._get_tool_schema(server_name, actual_tool_name)
                    schema_found = isinstance(tool_schema, dict) and bool(tool_schema)
                    iteration_data["schema_found"] = schema_found
                    iteration_data["schema_content"] = tool_schema if schema_found else None
                    iteration_data["validation_result"] = None
                    iteration_data["validation_skipped_reason"] = None

                    # --- Schema-Based Validation or Skip ---
                    is_valid = True
                    validation_message = "Schema not found, validation skipped."
                    validation_failed = False # Flag for failure counting

                    if schema_found:
                        is_valid, validation_message, _ = self.mcp_manager.validate_tool_arguments(server_name, actual_tool_name, arguments)
                        iteration_data["validation_result"] = {"is_valid": is_valid, "message": validation_message}

                        if not is_valid:
                            validation_failed = True # Mark validation as failed
                            logger.warning(f"[ReAct] Tool argument validation failed for '{tool_key}': {validation_message}")
                            required_params_str = self._format_required_params(tool_schema)
                            schema_properties = tool_schema.get('properties', {})
                            provided_args_str = json.dumps(arguments, indent=2)

                            # --- NEW: More explicit feedback based on parsed errors ---
                            error_detail = validation_message # Default

                            # Use more specific parsing of jsonschema errors
                            required_match = re.search(r"'(.+?)' is a required property", validation_message)
                            additional_match = re.search(r"Additional properties are not allowed \('(.+?)' was unexpected\)", validation_message)
                            type_match = re.search(r"parameter '(.+?)': .+ is not of type '(\w+)'", validation_message) # Path might be nested
                            # --- NEW: More explicit feedback based on parsed errors ---
                            specific_guidance = ""
                            if required_match:
                                missing_param = required_match.group(1)
                                # Check if the AI tried to provide a similar but incorrect name (common mistake)
                                possible_mistake = None
                                for provided_arg in arguments.keys():
                                    # Simple check for common variations (e.g., file_path vs path)
                                    if missing_param in provided_arg or provided_arg in missing_param:
                                         possible_mistake = provided_arg
                                         break
                                if possible_mistake:
                                     error_detail = f"You provided the parameter '{possible_mistake}', but the schema requires the parameter named **'{missing_param}'**. Please use the exact name '{missing_param}'."
                                     specific_guidance = f"Rename the parameter '{possible_mistake}' to '{missing_param}'."
                                else:
                                     error_detail = f"You missed the required parameter **'{missing_param}'**. Please include it in the `arguments`."
                                     specific_guidance = f"Add the required parameter '{missing_param}' to the `arguments` object."

                            elif additional_match:
                                unexpected_param = additional_match.group(1)
                                valid_params = list(schema_properties.keys())
                                error_detail = f"You provided an unexpected parameter **'{unexpected_param}'**. The schema only allows these parameters: {valid_params}. Please remove '{unexpected_param}'."
                                specific_guidance = f"Remove the unexpected parameter '{unexpected_param}' from the `arguments`."
                            elif type_match:
                                param_name = type_match.group(1) # Might be nested path like properties -> address -> street
                                expected_type = type_match.group(2)
                                # Try to get the actual provided type if possible
                                provided_value = arguments.get(param_name) # Simple lookup, might fail for nested
                                provided_type = type(provided_value).__name__ if provided_value is not None else "unknown"
                                error_detail = f"The parameter **'{param_name}'** requires a **{expected_type}** value, but you provided a value of type **{provided_type}**. Please provide a value of the correct type."
                                specific_guidance = f"Ensure the value for '{param_name}' is a {expected_type}."
                            # Add more specific error parsing if needed...
                            # --- Feedback Generation End ---

                            error_feedback = (
                                f"❌ **ERROR: Tool Argument Validation Failed for '{tool_key}'**\n\n"
                                f"**Reason:** {error_detail}\n\n"
                                f"**Arguments you provided:**\n```json\n{provided_args_str}\n```\n\n"
                                f"**Tool Schema Reminder:**\n"
                                f"- Tool Description: {tool_schema.get('description', 'N/A')}\n"
                                f"- Input Schema Properties: {json.dumps(schema_properties, indent=2)}\n"
                                f"- Required Parameters: {required_params_str}\n\n"
                                f"**ACTION REQUIRED:** {specific_guidance} Correct the `arguments` object in your next JSON response based on the schema and try the tool call again, or choose a different action."
                            )

                            conversation_history.append({"role": "user", "content": error_feedback})
                            iteration_data["error"] = f"Argument validation failed: {validation_message}"
                            metadata["iterations"].append(iteration_data)
                            # (Log iteration event)
                            log_event_data = {
                                "timestamp": datetime.now().isoformat(),
                                "event_type": "react_iteration",
                                "session_id": session_id,
                                "iteration_data": iteration_data
                            }
                            await async_save_meta_log(session_log_dir, session_id, log_event_data)
                            # continue # Skip execution handled below by validation_failed flag
                        else:
                            logger.info(f"[ReAct] Tool arguments validated successfully for '{tool_key}'.")

                    else: # Schema not found
                        iteration_data["validation_skipped_reason"] = "Schema not found"
                        # (Schema not found warning message logic - keep as is)
                        logger.warning(f"[ReAct] No schema found for '{tool_key}'. Skipping argument validation.")
                        tool_details_str = self._get_detailed_tool_info(server_name, actual_tool_name)
                        schema_warning_feedback = (
                            f"WARNING: I could not find the input schema (parameter list) for the tool '{tool_key}'.\n"
                            f"Description: {tool_details_str}\n\n"
                            f"I will attempt to execute it with the arguments you provided: {json.dumps(arguments)}\n"
                            f"However, if it fails, you may need to guess the correct arguments based on the description or try calling it with no arguments if that seems appropriate."
                        )
                        conversation_history.append({"role": "user", "content": schema_warning_feedback})


                    # --- Execute Tool (only if validation passed) ---
                    observation = None
                    tool_error = None
                    execution_failed = False # Flag for failure counting

                    if not validation_failed: # Only execute if validation passed (or was skipped)
                        try:
                            # (Existing execution logic - inform LLM, call execute_tool, format observation)
                            if schema_found:
                                required_params_str = self._format_required_params(tool_schema)
                                schema_properties = tool_schema.get('properties', {})
                                tool_info_message = (
                                    f"INFO: You requested to call the tool '{tool_key}'. Arguments validated successfully.\n"
                                    f"Schema Reminder:\n"
                                    f"Input Schema Properties: {json.dumps(schema_properties, indent=2)}\n"
                                    f"Required Parameters: {required_params_str}\n\n"
                                    f"I will now execute the tool with the arguments provided: {json.dumps(arguments)}"
                                )
                            else: # Schema was not found
                                tool_info_message = (
                                    f"INFO: Attempting to execute tool '{tool_key}' (schema was not found). "
                                    f"Arguments provided: {json.dumps(arguments)}"
                                )
                            if not iteration_data.get("validation_skipped_reason"):
                                conversation_history.append({"role": "user", "content": tool_info_message})
                            
                            logger.info(f"[ReAct {session_id}] Executing tool '{tool_key}' with arguments: {arguments}")
                            tool_result = await self.mcp_manager.execute_tool(server_name, actual_tool_name, arguments)
                            observation = self._format_observation(tool_result)
                            iteration_data["observation"] = observation
                            logger.info(f"[ReAct {session_id}] Tool execution successful. Result: {observation[:200]}...")
                            # Reset failure count for this specific tool+args combo on success
                            failure_key = (tool_key, arguments_key)
                            if failure_key in repeated_failures:
                                 del repeated_failures[failure_key]

                        except Exception as e:
                            execution_failed = True # Mark execution as failed
                            logger.error(f"[ReAct {session_id}] Tool execution error for '{tool_key}': {str(e)}", exc_info=True)
                            observation = self._format_observation(e)
                            tool_error = str(observation)
                            iteration_data["observation"] = observation
                            iteration_data["error"] = tool_error
                    else:
                        # If validation failed, skip execution and set error message for feedback
                        tool_error = iteration_data["error"] # Use validation error for feedback context
                        observation = None # No observation if validation failed

                    # --- Handle Repeated Failures ---
                    # Increment failure count only if validation or execution failed
                    if validation_failed or execution_failed:
                         failure_key = (tool_key, arguments_key)
                         repeated_failures[failure_key] = repeated_failures.get(failure_key, 0) + 1
                         current_fail_count = repeated_failures[failure_key]

                         # Provide repeated failure warning if count reaches threshold
                         if current_fail_count >= 3:
                              logger.warning(f"[ReAct {session_id}] Tool '{tool_key}' with args {arguments} failed {current_fail_count} times consecutively. Adding final guidance.")
                              required_params_str = self._format_required_params(iteration_data.get("schema_content"))
                              last_error_msg = iteration_data["error"] # Use the logged error
                              reset_message = (
                                   f"NOTICE: Trying to use tool '{tool_key}' with the arguments:\n```json\n{json.dumps(arguments, indent=2)}\n```\n"
                                   f"has failed {current_fail_count} times in a row.\n"
                                   f"Last Error: {last_error_msg}\n"
                                   f"Tool Schema Reminder: Required parameters are {required_params_str}.\n\n"
                                   f"**STOP trying this specific tool call.** Either:\n"
                                   f"1. Try the tool '{tool_key}' again but with **significantly different arguments** based on the schema.\n"
                                   f"2. Try a **completely different tool**.\n"
                                   f"3. Provide a **final_answer** based on the information you already have."
                              )
                              conversation_history.append({"role": "user", "content": reset_message})
                              iteration_data["error"] += f" (Repeated Failure: {current_fail_count} times)"
                              # Do not reset counter here, let a future *successful* call reset it.

                    # --- Construct feedback message for LLM ---                    
                    observation_message_for_llm = ""
                    if tool_error: # This covers both validation and execution errors
                         tool_schema_content = iteration_data.get("schema_content")
                         schema_properties = tool_schema_content.get('properties', {}) if tool_schema_content else {}
                         required_params_str = self._format_required_params(tool_schema_content) if tool_schema_content else "Unknown (schema not found)"

                         observation_message_for_llm = (
                             f"ERROR: Executing tool '{tool_key}' failed.\n"
                             f"Error details: {tool_error}\n\n" # Use formatted error
                             + (f"Recall the tool's schema:\n"
                                f"Input Schema Properties: {json.dumps(schema_properties, indent=2)}\n"
                                f"Required Parameters: {required_params_str}\n\n" if schema_found else # Only show if schema was found
                                "Tool schema was not found, so requirements are unknown.\n\n")
                             + f"ACTION REQUIRED: Review the error and schema (if available). Correct the `arguments` and try again, or choose a different action."
                         )

                    elif observation is not None: # Check observation is not None
                        observation_message_for_llm = (
                            f"Result from tool call {tool_key}:\n" # Use tool_key
                            f"{observation}\n\n"
                            "Carefully analyze this complete result to determine your next step."
                        )
                    else:
                        # This case should ideally not happen if validation passed and execution was attempted
                        # But handle it just in case
                        observation_message_for_llm = f"Tool '{tool_key}' seems to have executed (validation passed), but no specific observation was generated. This might be unexpected."
                        logger.warning(f"[ReAct {session_id}] No observation generated for {tool_key} after successful validation.")


                    if observation_message_for_llm:
                        conversation_history.append({"role": "user", "content": observation_message_for_llm})
                    
                    # If validation or execution failed, continue to next iteration
                    if validation_failed or execution_failed:
                         metadata["iterations"].append(iteration_data) # Log the failed iteration
                         # Define iter_log_event_data here before using it
                         iter_log_event_data = {
                             "timestamp": datetime.now().isoformat(),
                             "event_type": "react_iteration",
                             "session_id": session_id,
                             "iteration_data": iteration_data
                         }
                         await async_save_meta_log(session_log_dir, session_id, iter_log_event_data)
                         continue # Go to next iteration

            elif action.get("action_type") == "final_answer":
                final_answer = action.get("answer", "")
                thought = action.get("thought", "")
                
                # Add debug logging
                logger.info(f"[ReAct] Final answer detected: {final_answer[:100]}...")
                
                # Return final answer immediately
                metadata["final_response"] = final_answer
                iteration_data["action"] = action # Ensure final action is logged
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
                    f"ERROR: Unknown action type '{action.get('action_type')}'. Use 'tool_call' or 'final_answer'."
                )
                conversation_history.append({"role": "user", "content": error_feedback})
                iteration_data["error"] = "Unknown action type"
                metadata["iterations"].append(iteration_data)
                continue
            
            # Add iteration data to the list (only if not already added due to error/failure)
            # The check `if validation_failed or execution_failed:` above handles appending on failure
            if not iteration_data.get("error"):
                 metadata["iterations"].append(iteration_data)
            
            # Log iteration data using the central async function
            # (The check above prevents double logging on failure)
            if not iteration_data.get("error"):
                await async_save_meta_log(session_log_dir, session_id, {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "react_iteration",
                    "session_id": session_id,
                    "iteration_data": iteration_data
                })
            
            # Check conversation history length and trim if necessary
            self.trim_conversation_history(session_id)
        
        # If we reached max iterations without a final answer, use the last response
        if final_answer is None:
            final_answer = "I'm sorry, I couldn't find an appropriate response. Please make your question more specific or try asking in a different way."
            logger.warning("[ReAct] Reached max iterations without final answer")
            metadata["final_response"] = final_answer
        
        # Save final state
        await async_save_meta_log(session_log_dir, session_id, metadata)
        
        # Return the final result
        return {
            "response": final_answer,
            "metadata": metadata
        }

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

    def trim_conversation_history(self, session_id: str):
        """Trim conversation history to the maximum allowed length."""
        history = self.get_conversation_history(session_id)
        if len(history) > self.max_history_length:
            # Keep the system message and the most recent messages
            self.conversation_histories[session_id] = [history[0]] + history[-self.max_history_length+1:]
        else:
            # No need to trim
            pass 