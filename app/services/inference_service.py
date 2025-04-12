import logging
import re
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime

from app.services.mcp_service import MCPService
from app.services.model_service import ModelService, ModelError
from app.utils.log_utils import async_save_meta_log, default_serializer
from app.core.config import settings # Import settings

logger = logging.getLogger(__name__)

# Define allowed control characters in JSON strings
# JSON spec allows \", \\, \/, \b, \f, \n, \r, \t and \uXXXX escapes
# We target ASCII control characters 0x00-0x1F and 0x7F (DEL)
# Basic printable ASCII range is 0x20-0x7E
# Stricter removal: Remove ALL control chars 0x00-0x1F except TAB (0x09), plus DEL (0x7F).
CONTROL_CHARS_TO_REMOVE_PATTERN = re.compile(r"[\x00-\x1F\x7F]")

# Helper function to clean control characters from a string
def _clean_control_chars(text: str) -> str:
    if not isinstance(text, str):
        return text
    # Replace common problematic control chars with space or remove
    # Specifically replace newline and tab with space
    cleaned_text = text.replace('\\n', ' ').replace('\\t', ' ')
    # Remove other control chars (0x00-0x1F excluding TAB(0x09), plus DEL 0x7F)
    # Note: TAB was already replaced above
    cleaned_text = CONTROL_CHARS_TO_REMOVE_PATTERN.sub('', cleaned_text)
    return cleaned_text

class InferenceError(Exception):
    """Specific errors occurring during inference process"""
    pass

class ToolInputValidationError(InferenceError):
    """Error when tool input validation fails"""
    pass

class ParsingError(InferenceError):
    """Error when LLM response parsing fails"""
    pass

# Helper to detect duplicate keys during JSON parsing
def _detect_duplicate_keys(ordered_pairs):
    d = {}
    keys_seen = set()
    for k, v in ordered_pairs:
        if k in keys_seen:
            raise ValueError(f"Duplicate key found: '{k}'") # Raise error on duplicate
        keys_seen.add(k)
        d[k] = v
    return d

class InferenceService:
    def __init__(
        self,
        mcp_manager: MCPService,
        model_service: ModelService,
    ):
        self.mcp_manager = mcp_manager
        self.model_service = model_service
        
        # Use settings for configuration
        self.max_history_length = 50 # Keep a reasonable default or make configurable via settings if needed
        self.log_dir = Path(settings.log_dir) if settings.log_dir else None

        # Conversation management attributes
        self.conversation_histories = {}  # Conversation history by session
        self._tools_schema_cache = {}  # Tool schema cache
        
        logger.info(f"InferenceService initialized with MCPService and ModelService (model path: {self.model_service.model_path}, log dir: {self.log_dir})")

        # Regular expression patterns for parsing
        self.json_regex = re.compile(r'```json\s*([\s\S]*?)\s*```')  # Pattern to extract JSON blocks
        self.final_answer_pattern = re.compile(r'final answer:?\s*(.*)', re.IGNORECASE)  # Pattern to extract Final Answer
        
        # Tool schema caching
        self.tool_schemas = {}

    # Removed set_log_directory as it's now handled by settings
    # def set_log_directory(self, log_dir: Path):
    #     """Set the log directory."""
    #     self.log_dir = log_dir
    #     if self.log_dir and not self.log_dir.exists():
    #         self.log_dir.mkdir(parents=True, exist_ok=True)

    def _basic_clean_json_string(self, json_str: str) -> str:
        """Performs very basic cleaning like removing trailing commas before closing braces/brackets."""
        # Remove trailing commas before closing brace/bracket
        cleaned = re.sub(r",\\s*([}\\]])", r"\\1", json_str)
        return cleaned

    def _clean_json_control_chars(self, json_str: str) -> str:
        """Removes common problematic control characters within JSON strings ONLY."""
        cleaned = json_str
        # Attempt to remove ONLY unescaped newlines/tabs *within* string values
        # This is tricky and might need refinement based on observed LLM outputs
        try:
            # More robustly find string values and replace internal newlines/tabs
            def replace_control_chars(match):
                key = match.group(1)
                string_val = match.group(2)
                # Replace unescaped control chars inside the string value
                cleaned_val = re.sub(r'(?<!\\\\)\\n', ' ', string_val) # Replace newline with space
                cleaned_val = re.sub(r'(?<!\\\\)\\t', ' ', cleaned_val) # Replace tab with space
                # Remove potential lingering backslashes from failed escape sequences if any
                cleaned_val = cleaned_val.replace('\\\\', '\\\\\\\\') # Ensure backslashes are properly escaped for JSON
                return f'"{key}": "{cleaned_val}"'

            # Target key-value pairs where value is a string
            cleaned = re.sub(r'\\"([^\\"]+)\\":\\s*\\"((?:\\\\.|[^\\"\\\\])*)\\"', replace_control_chars, cleaned)
        except Exception as e:
            logger.warning(f"Error during advanced control char cleaning, falling back: {e}")
            # Fallback basic cleaning if regex fails
            cleaned = self._basic_clean_json_string(json_str)
            cleaned = cleaned.replace("\\n", " ").replace("\\t", " ") # Basic global replace as fallback

        # Remove trailing commas before closing brace/bracket (apply again after cleaning)
        cleaned = re.sub(r",\\s*([}\\]])", r"\\1", cleaned)
        return cleaned

    def _parse_llm_action_json(self, llm_output: str, log_prefix: str = "") -> Dict:
        """Parses the LLM output to extract and validate the JSON action block for ReAct.

        Args:
            llm_output: The raw string output from the LLM.
            log_prefix: Prefix for log messages (e.g., session ID).

        Returns:
            A dictionary representing the validated action (tool call or final answer),
            or an error dictionary if parsing/validation fails.
            Structure for success:
                - Tool Call: {'is_final': False, 'thought': str, 'tool_name': str, 'arguments': dict}
                - Final Answer: {'is_final': True, 'thought': str, 'answer': str}
            Structure for error:
                {'is_final': None, 'error': str, 'original_response': str}
        """
        # 1. Extract content within ```json ... ``` block
        # Regex to find ```json blocks, handling potential leading/trailing whitespace
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", llm_output, re.DOTALL | re.IGNORECASE)
        json_str = ""
        original_response_for_error = llm_output # Default to full output for error context

        if not match:
            # If no block found, try to parse the whole output if it looks like JSON
            stripped_output = llm_output.strip()
            if stripped_output.startswith("{") and stripped_output.endswith("}"):
                logger.warning(f"{log_prefix} LLM response looks like JSON but missing ```json block. Attempting parse anyway.")
                json_str = stripped_output
                original_response_for_error = json_str # Use the stripped version for error context
            else:
                logger.warning(f"{log_prefix} No ```json block found in LLM output: {llm_output[:200]}...")
                return {
                    "is_final": None, # Indicate error state
                    "error": "Formatting Error: No ```json block found in your response. You MUST provide your response inside a single ```json ... ``` block.",
                    "original_response": llm_output
                }
        else:
            json_str = match.group(1).strip()
            original_response_for_error = json_str # Use the extracted block for error context

        # --- START MODIFICATION: Clean json_str BEFORE parsing --- 
        if json_str:
             original_json_str = json_str # Keep original for error reporting if needed
             # 1. Strip leading/trailing whitespace
             json_str = json_str.strip()
             # 2. Clean problematic characters within the string
             cleaned_json_str = _clean_control_chars(json_str)
             if cleaned_json_str != original_json_str:
                  logger.debug(f"{log_prefix} Applied control character cleaning to JSON string.")
                  # logger.debug(f"{log_prefix} Original: {original_json_str[:500]}... Cleaned: {cleaned_json_str[:500]}...") # Optional: log diff
                  json_str = cleaned_json_str # Use the cleaned string for parsing
        # --- END MODIFICATION --- 

        # 2. Attempt to parse the cleaned JSON string
        parsed_dict = None
        parsing_methods = [
            lambda hook=_detect_duplicate_keys: json.loads(json_str, object_pairs_hook=hook) # Parse the potentially cleaned json_str
        ]
        last_error_msg = "Unknown JSON parsing error"

        for i, parse_method in enumerate(parsing_methods):
            try:
                parsed_dict = parse_method()
                logger.info(f"{log_prefix} Successfully parsed JSON using method {i+1} (checked for duplicate keys)")
                break
            except ValueError as ve:
                if "Duplicate key found" in str(ve):
                    logger.warning(f"{log_prefix} Duplicate key detected during parsing: {ve}")
                    last_error_msg = f"JSON Formatting Error: {ve}. Each key must appear only ONCE."
                    return {
                        "is_final": None,
                        "error": last_error_msg,
                        "original_response": original_response_for_error
                    }
            except json.JSONDecodeError as e:
                last_error_msg = f"JSON Parsing Error: {e}. Invalid JSON structure or syntax. Original near error: '{json_str[max(0, e.pos - 20):e.pos + 20]}'"
                logger.warning(f"{log_prefix} JSON parsing failed with method {i+1}: {e}")
                # Keep trying other methods if defined
            except Exception as e:
                last_error_msg = f"Unexpected parsing error: {e}"
                logger.error(f"{log_prefix} Unexpected error during JSON parsing: {e}", exc_info=True)
                # Stop trying on unexpected errors
                break

        if parsed_dict is None:
            logger.error(f"{log_prefix} All JSON parsing attempts failed. Last error: {last_error_msg}")
            return {
                "is_final": None,
                "error": f"Formatting Error: Could not parse the JSON in your response. Reason: {last_error_msg}",
                "original_response": original_response_for_error
            }

        # 3. Validate the structure and content of the parsed JSON
        # --- is_final --- (New format)
        is_final = parsed_dict.get("is_final") # Allow boolean or None
        if not isinstance(is_final, bool) and is_final is not None:
             logger.warning(f"{log_prefix} Invalid 'is_final' value: {is_final}. Must be true, false, or absent.")
             return {
                 "is_final": None,
                 "error": "Formatting Error: The 'is_final' field must be either true or false (boolean).",
                 "original_response": original_response_for_error
             }
        # If is_final is None, treat as False (tool call) for simplicity, but log it
        if is_final is None:
             logger.debug(f"{log_prefix} 'is_final' field missing, assuming tool call (is_final=false).")
             is_final = False # Default to tool call if missing

        # --- thought --- (Required for both)
        thought = parsed_dict.get("thought")
        if not isinstance(thought, str) or not thought.strip():
            logger.warning(f"{log_prefix} Missing or invalid 'thought' field: {thought}")
            return {
                "is_final": None,
                "error": "Formatting Error: The 'thought' field (string) is required and cannot be empty.",
                "original_response": original_response_for_error
            }
        
        validated_action = {"is_final": is_final, "thought": thought.strip()}

        # --- Validate based on is_final --- 
        if is_final:
            # Final Answer specific validation
            answer = parsed_dict.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                logger.warning(f"{log_prefix} Missing or invalid 'answer' field for final response: {answer}")
                return {
                    "is_final": None,
                    "error": "Formatting Error: For the final answer (`is_final: true`), the 'answer' field (string) is required and cannot be empty.",
                    "original_response": original_response_for_error
                }
            # Check for forbidden keys in final answer
            if "tool_name" in parsed_dict or "arguments" in parsed_dict:
                 logger.warning(f"{log_prefix} Forbidden keys ('tool_name'/'arguments') found in final answer.")
                 return {
                     "is_final": None,
                     "error": "Formatting Error: Do not include 'tool_name' or 'arguments' when `is_final` is true.",
                     "original_response": original_response_for_error
                 }
            validated_action["answer"] = answer.strip()
        else:
            # Tool Call specific validation
            tool_name = parsed_dict.get("tool_name")
            arguments = parsed_dict.get("arguments", {})

            if not isinstance(tool_name, str) or not tool_name.strip():
                logger.warning(f"{log_prefix} Missing or invalid 'tool_name' field for tool call: {tool_name}")
                return {
                    "is_final": None,
                    "error": "Formatting Error: When calling a tool (`is_final: false`), the 'tool_name' field (string) is required.",
                    "original_response": original_response_for_error
                }
            
            # Check for forbidden keys in tool call
            if "answer" in parsed_dict:
                logger.warning(f"{log_prefix} Forbidden key ('answer') found in tool call.")
                return {
                    "is_final": None,
                    "error": "Formatting Error: Do not include the 'answer' field when calling a tool (`is_final: false`).",
                    "original_response": original_response_for_error
                }
                
            # Arguments: Optional, but if present, must be a dict
            if arguments is not None and not isinstance(arguments, dict):
                logger.warning(f"{log_prefix} Invalid 'arguments' field: {arguments}. Must be a dictionary.")
                return {
                    "is_final": None,
                    "error": "Formatting Error: The 'arguments' field, if provided for a tool call, must be a dictionary (JSON object).",
                    "original_response": original_response_for_error
                }

            validated_action["tool_name"] = tool_name.strip()
            validated_action["arguments"] = arguments if arguments is not None else {} # Default to empty dict if None

        # Log the validated action details separately
        thought_summary = f"{validated_action['thought'][:50]}..." if validated_action['thought'] else "No thought"
        logger.info(f"{log_prefix}Successfully parsed and validated action: is_final={validated_action['is_final']}, thought='{thought_summary}'")
        return validated_action

    def _format_observation(self, result: Union[str, Dict, Exception]) -> str:
        """Formats the result of a tool execution into a string observation for the LLM."""
        # Ensure the result is JSON serializable before formatting
        try:
            if isinstance(result, Exception):
                # Format exceptions clearly
                return f"Error: Tool execution failed with error: {type(result).__name__}: {str(result)}"
            elif isinstance(result, dict):
                # Serialize dicts to a JSON string
                try:
                    # Try standard JSON serialization
                    return json.dumps(result, ensure_ascii=False, indent=2)
                except TypeError as te:
                    logger.warning(f"Could not serialize result dict directly: {te}. Using fallback serializer.")
                    # Fallback for complex objects within the dict
                    return json.dumps(result, ensure_ascii=False, indent=2, default=default_serializer)
            elif isinstance(result, str):
                 # Return strings directly, potentially truncating if very long
                 max_len = 2000 # Configurable? Maybe add to settings
                 if len(result) > max_len:
                      return result[:max_len] + "... [truncated]"
                 return result
            else:
                # Convert other types to string
                return str(result)
        except Exception as e:
            logger.error(f"Error formatting observation: {e}", exc_info=True)
            return f"Error: Could not format the tool result: {e}"

    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieves the conversation history for a given session ID."""
        return self.conversation_histories.get(session_id, [])

    def add_to_conversation(self, session_id: str, role: str, content: str) -> None:
        """Adds a message to the conversation history for a given session ID."""
        if session_id not in self.conversation_histories:
            self.conversation_histories[session_id] = []
            
        # Ensure content is a string
        if not isinstance(content, str):
            content = str(content)

        self.conversation_histories[session_id].append({"role": role, "content": content})
        
        # Optional: Trim history if it exceeds max length
        self.trim_conversation_history(session_id)

    async def process_react_pattern(
        self,
        initial_prompt: str,
        session_id: str,
        session_log_dir: Path,
    ) -> Dict[str, Any]:
        """Implements the ReAct loop using the LLM and tools.

        Args:
            initial_prompt: The initial user query.
            session_id: The unique ID for the current session.
            session_log_dir: The directory to save session logs.

        Returns:
            A dictionary containing the final response or an error.
        """
        
        start_time = datetime.now()
        log_prefix = f"Session {session_id} | "
        
        # Clear previous history for the session if it exists
        if session_id in self.conversation_histories:
            logger.debug(f"{log_prefix}Clearing existing conversation history for session.")
            del self.conversation_histories[session_id]

        # Initialize conversation with the initial prompt
        self.add_to_conversation(session_id, "user", initial_prompt)

        # Build the system prompt once
        system_prompt = self._build_system_prompt()
        
        iteration_results = [] # Store results of each iteration for meta log
        
        # Use max_iterations from settings
        max_iterations = settings.react_max_iterations 

        for i in range(max_iterations):
            iteration_start_time = datetime.now()
            logger.info(f"{log_prefix}--- Iteration {i + 1}/{max_iterations} --- ")
            iteration_data = {
                "iteration": i + 1,
                "timestamp": iteration_start_time.isoformat(),
                "prompt": None, # Will be the full conversation history
                "response": None,
                "action": None,
                "observation": None,
                "error": None,
                "tool_name_requested": None,
                "arguments_provided": None,
                "schema_found": False,
                "schema_content": None,
                "validation_result": None,
                "validation_skipped_reason": None,
                "parsed_action": None
            }
            
            # Get current conversation history
            conversation = self.get_conversation_history(session_id)
            
            # Combine system prompt and conversation for LLM
            full_prompt_messages = [
                {"role": "system", "content": system_prompt},
                *conversation
            ]
            iteration_data["prompt"] = json.dumps(full_prompt_messages, indent=2) # Log the full prompt

            # 1. Generate LLM response (Reasoning + Action determination)
            try:
                llm_response = await self._generate_llm_response(full_prompt_messages, log_prefix)
                iteration_data["llm_response"] = llm_response
            except ModelError as e:
                error_msg = f"Model generation failed: {e}"
                logger.error(f"{log_prefix}{error_msg}", exc_info=True)
                iteration_data["error"] = error_msg
                iteration_results.append(iteration_data)
                return {"error": error_msg, "details": str(e)}
            except Exception as e:
                error_msg = f"Unexpected error during LLM response generation: {e}"
                logger.error(f"{log_prefix}{error_msg}", exc_info=True)
                iteration_data["error"] = error_msg
                iteration_results.append(iteration_data)
                return {"error": error_msg}

            # 2. Parse the LLM response to get the action (or error)
            parsed_action = self._parse_llm_action_json(llm_response, log_prefix)
            iteration_data["parsed_action"] = parsed_action # Log the parsing result

            # Handle parsing errors
            if parsed_action.get("error"):
                 error_message = parsed_action["error"]
                 logger.warning(f"{log_prefix}LLM response parsing failed: {error_message}")
                 # Provide feedback to the LLM and retry
                 feedback = f"System Error: Your previous response could not be parsed. Reason: {error_message}. Please correct your JSON format and try again. Remember to output ONLY a single JSON object inside ```json ... ```."                 
                 self.add_to_conversation(session_id, "assistant", llm_response) # Add the failed response
                 self.add_to_conversation(session_id, "user", feedback) # Add feedback as if user said it
                 iteration_data["error"] = error_message
                 iteration_data["observation"] = feedback # Log the feedback given
                 iteration_results.append(iteration_data)
                 continue # Retry loop

            # Successfully parsed action
            iteration_data["action"] = parsed_action # Log the valid action structure
            # logger.info(f"{log_prefix}Parsed Action: {'is_final': {parsed_action['is_final']}, 'thought': '{parsed_action['thought'][:50]}...'}")
            # Updated log format to avoid f-string issue
            thought_summary = f"{parsed_action['thought'][:50]}..." if parsed_action.get('thought') else "No thought"
            logger.info(f"{log_prefix}Parsed Action: is_final={parsed_action['is_final']}, thought='{thought_summary}'")

            # Add thought to conversation history
            self.add_to_conversation(session_id, "assistant", llm_response) # Add the full successful response block

            # 3. Execute the action
            if parsed_action["is_final"]:
                # Final answer reached
                final_answer = parsed_action.get("answer", "No answer provided.")
                logger.info(f"{log_prefix}Final answer received after {i + 1} iterations: {final_answer[:100]}...")
                
                # Save final iteration data
                iteration_results.append(iteration_data)
                
                # Save meta log before returning
                end_time = datetime.now()
                final_meta = {
                    "event_type": "api_response", # 이벤트 타입 명시
                    "iterations": iteration_results,
                    "final_response": final_answer, # 최종 응답
                    "end_time": end_time.isoformat(),
                    "iteration_count": i + 1,
                    "max_iterations": max_iterations,
                }
                await async_save_meta_log(session_log_dir, final_meta, session_id, merge=True)
                
                return {"response": final_answer}
            else:
                # Tool call requested
                tool_name_full = parsed_action.get("tool_name")
                arguments = parsed_action.get("arguments", {})
                iteration_data["tool_name_requested"] = tool_name_full
                iteration_data["arguments_provided"] = arguments
                
                logger.info(f"{log_prefix}Executing tool: {tool_name_full} with args: {arguments}")

                # Split tool_name into server and tool
                if '.' not in tool_name_full:
                     error_msg = f"Invalid tool name format '{tool_name_full}'. Must be 'server_name.tool_name' format."
                     logger.warning(f"{log_prefix}{error_msg}")
                     observation = f"System Error: {error_msg} Available tools are: {self.mcp_manager.list_tools_for_prompt()}"
                     iteration_data["error"] = error_msg
                else:
                    server_name, tool_name = tool_name_full.split('.', 1)
                    
                    # Validate tool arguments before execution
                    validation_error_msg = None
                    schema = await self._get_tool_schema(server_name, tool_name)
                    if schema:
                         iteration_data["schema_found"] = True
                         iteration_data["schema_content"] = schema
                         validation_result = self._validate_tool_arguments(arguments, schema)
                         iteration_data["validation_result"] = validation_result
                         if not validation_result["valid"]:
                              validation_error_msg = f"Tool Input Error: {validation_result['error']}. Check the schema and your arguments."
                              logger.warning(f"{log_prefix}Tool '{tool_name_full}' arguments validation failed: {validation_result['error']}")
                    else:
                         logger.warning(f"{log_prefix}Could not retrieve schema for tool '{tool_name_full}'. Skipping argument validation.")
                         iteration_data["validation_skipped_reason"] = "Schema not found"
                         validation_error_msg = None # Allow execution without schema

                    if validation_error_msg:
                        # --- Simplified Error Feedback ---
                        # Get only the tool description, not the full schema details
                        tool_info = self.mcp_manager.get_tool_info(server_name, tool_name)
                        tool_description = tool_info.get('description', 'No description available.') if tool_info else 'Tool description not found.'
                        
                        # Extract missing required arguments from the validation error message if possible
                        missing_args_str = ""
                        if "Missing required argument" in validation_result['error']:
                            missing = [arg.split(": '")[-1].rstrip("'") for arg in validation_result['error'].split(';') if arg.strip().startswith("Missing required argument")]
                            if missing:
                                missing_args_str = f" You seem to be missing the following required arguments: {', '.join(missing)}."
                                
                        # Construct a more focused observation message
                        # Old message with full schema:
                        # schema_details = self._get_detailed_tool_info(server_name, tool_name)
                        # observation = f"System Error: {validation_error_msg}\n\nTool Schema for '{tool_name_full}':\n{schema_details}\nPlease correct the 'arguments' in your next JSON response based on the schema and try the tool call again."
                        
                        # New simplified message:
                        observation = (
                            f"System Error: Your previous attempt to use the tool '{tool_name_full}' failed due to invalid arguments. "
                            f"Error details: {validation_result['error']}.{missing_args_str}\n"
                            f"\n"
                            f"Tool Description: {tool_description}\n"
                            f"\n"
                            f"Please review the error, check the tool description, and correct the 'arguments' in your next JSON response. Ensure all required arguments are provided."
                        )
                        logger.info(f"{log_prefix}Providing simplified feedback to LLM for failed validation.")
                        iteration_data["error"] = validation_error_msg # Log the validation error itself
                    else:
                        # Argument validation passed or skipped, proceed with execution
                        try:
                             tool_result = await self.mcp_manager.execute_tool(server_name, tool_name, arguments)
                             observation = self._format_observation(tool_result)
                             logger.info(f"{log_prefix}Tool '{tool_name_full}' executed successfully. Observation (truncated): {observation[:200]}...")
                        except Exception as e:
                             error_msg = f"Tool execution failed for '{tool_name_full}': {type(e).__name__}: {e}"
                             logger.error(f"{log_prefix}{error_msg}", exc_info=True)
                             # Provide schema info in feedback even for execution errors
                             schema_details = self._get_detailed_tool_info(server_name, tool_name)
                             observation = f"System Error: {error_msg}. Tool execution failed. Check your arguments against the schema.\n\nTool Schema for '{tool_name_full}':\n{schema_details}"
                             iteration_data["error"] = error_msg

                # Add the observation to conversation history
                self.add_to_conversation(session_id, "user", observation) # Present observation as user input
                iteration_data["observation"] = observation

            # Append iteration data to results
            iteration_results.append(iteration_data)

        # Max iterations reached
        logger.warning(f"{log_prefix}Maximum iterations ({max_iterations}) reached. Returning current state or error.")
        final_error_msg = f"Processing Error: Maximum iterations ({max_iterations}) reached without a final answer."

        # Save meta log for max iteration reached
        end_time = datetime.now()
        final_meta = {
            "event_type": "api_response", # 이벤트 타입 명시
            "iterations": iteration_results,
            "final_response": f"Error: {final_error_msg}",
            "end_time": end_time.isoformat(),
            "iteration_count": max_iterations,
            "max_iterations": max_iterations,
            "errors": [final_error_msg] # API 응답 레벨 에러 추가
        }
        await async_save_meta_log(session_log_dir, final_meta, session_id, merge=True)

        return {"error": final_error_msg}

    def _validate_tool_arguments(self, arguments: Dict, schema: Dict) -> Dict:
        """Validates the provided arguments against the tool's input schema using simplified checks.
        
        Args:
            arguments: The dictionary of arguments provided by the LLM.
            schema: The input schema dictionary for the tool (simplified version expected).

        Returns:
            A dictionary {"valid": bool, "error": Optional[str]}
        """
        if not schema or not isinstance(schema, dict):
            return {"valid": True, "error": None} # Cannot validate without schema

        required_params = schema.get('required', [])
        properties = schema.get('properties', {})
        validation_errors = []

        # 1. Check for missing required arguments
        for param_name in required_params:
            if param_name not in arguments:
                validation_errors.append(f"Missing required argument: '{param_name}'")

        # 2. Check argument types (basic type checking)
        for arg_name, arg_value in arguments.items():
            if arg_name in properties:
                expected_type_str = properties[arg_name].get('type')
                if expected_type_str:
                    expected_type = None
                    if expected_type_str == 'string':
                        expected_type = str
                    elif expected_type_str == 'integer':
                        expected_type = int
                    elif expected_type_str == 'number':
                        expected_type = (int, float) # Allow both int and float for number
                    elif expected_type_str == 'boolean':
                        expected_type = bool
                    elif expected_type_str == 'array':
                        expected_type = list
                    elif expected_type_str == 'object':
                        expected_type = dict
                    
                    if expected_type and not isinstance(arg_value, expected_type):
                        validation_errors.append(f"Invalid type for argument '{arg_name}'. Expected {expected_type_str}, got {type(arg_value).__name__}")
            else:
                 # Optional: Warn about arguments provided that are not in the schema?
                 # logger.debug(f"Argument '{arg_name}' provided but not defined in schema properties.")
                 pass

        if validation_errors:
            return {"valid": False, "error": "; ".join(validation_errors)}
        else:
            return {"valid": True, "error": None}
            
    async def _generate_llm_response(self, messages: List[Dict], log_prefix: str = "") -> str:
        """Generates a response from the LLM using the model service."""
        if not self.model_service:
            raise ModelError("ModelService is not initialized.")
        
        logger.debug(f"{log_prefix}Generating LLM response with {len(messages)} messages...")
        
        # Use generation parameters from settings
        generation_params = {
            "max_tokens": settings.model_max_tokens,
            "temperature": settings.model_temperature,
            "top_p": settings.model_top_p,
            "top_k": settings.model_top_k,
            "min_p": settings.model_min_p,
            # Add other parameters like stop sequences if needed, maybe from settings too?
            # "stop": ["Observation:"] # Example stop sequence
        }

        # Filter out params with default/None values if the model service expects clean args
        # generation_params = {k: v for k, v in generation_params.items() if v is not None} 
        # ^-- Check if ModelService handles None/defaults or needs explicit filtering

        logger.debug(f"{log_prefix}Generation parameters: {generation_params}")
        
        try:
            response = await self.model_service.generate_chat(
                messages=messages,
                **generation_params
            )
            # Assuming generate_chat returns the text content directly or within a structure
            if isinstance(response, dict) and 'choices' in response and response['choices']: 
                 content = response['choices'][0].get('message', {}).get('content', '')
            elif isinstance(response, str):
                 content = response
            else:
                 logger.warning(f"{log_prefix}Unexpected response format from generate_chat: {type(response)}")
                 content = str(response) # Fallback
                 
            logger.debug(f"{log_prefix}LLM Raw Response Received (type: {type(content)}): {content[:200]}...")
            return content
        except ModelError as e:
            logger.error(f"{log_prefix}ModelError during generation: {e}")
            raise # Re-raise specific ModelError
        except Exception as e:
            logger.error(f"{log_prefix}Unexpected error during LLM generation: {e}", exc_info=True)
            raise ModelError(f"LLM generation failed unexpectedly: {e}") # Wrap in ModelError

    def _get_tool_list_prompt(self) -> str:
        """
        시스템 프롬프트용 도구 목록 설명을 생성합니다.
        """
        try:
            all_tools = self.mcp_manager.get_all_tool_details()
            if not all_tools:
                return "No tools available."
            
            tools_text = []
            for server_name, tools in all_tools.items():
                tools_text.append(f"\n## Server: {server_name}")
                if not tools:
                    tools_text.append("- No tools available on this server")
                    continue
                    
                for tool_name, tool_info in tools.items():
                    description = tool_info.get('description', 'No description available')
                    tools_text.append(f"- **{server_name}.{tool_name}**: {description}")
            
            tools_text.append("\n\nUse the format 'server_name.tool_name' when specifying the 'tool_name' in your JSON response.")
            
            tools_description = "\n".join(tools_text)
            return f"**Available Tools:**\n{tools_description}"
            
        except Exception as e:
            logger.error(f"Error generating tool list: {e}")
            return "Error retrieving tools. Please check the MCP configuration."

    def _get_detailed_tool_info(self, server_name: str, tool_name: str) -> str:
        """Generates a detailed description string for a specific tool, including schema."""
        tool_info = self.mcp_manager.get_tool_info(server_name, tool_name)
        if not tool_info:
            return f"Tool '{server_name}.{tool_name}' not found."

        description = tool_info.get('description', 'No description provided.')
        input_schema = tool_info.get('input_schema', {})
        
        # Format the schema nicely
        schema_str = "Input Schema:\n"
        properties = input_schema.get('properties', {})
        required = input_schema.get('required', [])
        
        if not properties:
            schema_str += "  No input parameters defined.\n"
        else:
            schema_str += "  Parameters:\n"
            for name, prop in properties.items():
                prop_type = prop.get('type', 'any')
                prop_desc = prop.get('description', '')
                is_required = " (required)" if name in required else ""
                schema_str += f"    - '{name}' ({prop_type}){is_required}: {prop_desc}\n"
        
        if not required:
            schema_str += "  All parameters are optional.\n"

        return f"**Tool: {server_name}.{tool_name}**\nDescription: {description}\n{schema_str}"

    async def _get_tool_schema(self, server_name: str, tool_name: str) -> Optional[Dict]:
        """Retrieves the input schema for a specific tool, using cache."""
        cache_key = f"{server_name}.{tool_name}"
        if cache_key in self._tools_schema_cache:
            return self._tools_schema_cache[cache_key]
        
        schema = await self.mcp_manager.get_tool_schema(server_name, tool_name)
        if schema:
            self._tools_schema_cache[cache_key] = schema
        return schema
        
    def _format_required_params(self, schema: Dict) -> str:
        """Extracts and formats required parameters from schema."""
        if not schema or 'required' not in schema or not schema['required']:
             return "None required."
        return ", ".join([f"'{p}'" for p in schema['required']])

    def trim_conversation_history(self, session_id: str):
        """Trims the conversation history if it exceeds the maximum length."""
        if session_id in self.conversation_histories:
            history = self.conversation_histories[session_id]
            if len(history) > self.max_history_length:
                # Keep the first message (initial prompt) and the most recent ones
                # Keep first user prompt + last (max_history_length - 1) messages
                num_to_keep = self.max_history_length - 1 
                self.conversation_histories[session_id] = [history[0]] + history[-num_to_keep:]
                logger.debug(f"Session {session_id} | Trimmed conversation history to {self.max_history_length} messages.")

    def _build_system_prompt(self) -> str:
        """Builds the system prompt including tool descriptions dynamically."""
        tools_section = self._get_tool_list_prompt() # Get dynamic tool list

        # Define the system prompt with the new format based on is_final
        # System prompt emphasizing stricter error handling, tool usage, and JSON format
        system_prompt_template = f"""URGENT: Your response MUST follow the JSON format using 'is_final'.
ABSOLUTELY CRITICAL: Your response MUST contain EXACTLY ONE JSON object inside the ```json block. DO NOT output multiple JSON objects.
ONE ACTION AT A TIME: Perform only ONE step per response - either ONE tool_call OR ONE final_answer. Do not plan multiple steps in one response.

You are an AI assistant. Use tools or provide a final answer step-by-step.

**RESPONSE FORMAT:** Your *entire* response MUST be inside ONE ```json ... ``` block containing ONLY ONE JSON object.
Choose ONE of the following single actions:

**1. Single Tool Call (when ONE tool is needed next - `is_final` is false or omitted):**
```json
{{
  "is_final": false, /* Must be false or omitted for tool call */
  "thought": "Your reasoning about why this tool is needed and what parameters to use.",
  "tool_name": "server_name.tool_name", /* REQUIRED: The tool to use (e.g., 'calculator.add') */
  "arguments": {{}}  /* REQUIRED (even if empty): Dictionary of arguments for the tool based on its schema */
}}
```
- `answer` key is NOT ALLOWED here.

**2. Single Final Answer (ONLY when the entire task is fully complete - `is_final` MUST be true):**
```json
{{
  "is_final": true, /* Must be true for the final answer */
  "thought": "Your final reasoning summarizing the process and the answer.",
  "answer": "The final answer to the user's query." /* REQUIRED: The final, user-facing answer string */
}}
```
- `tool_name` and `arguments` keys are NOT ALLOWED here.

**CRITICAL RULES:**
1.  **ONE JSON OBJECT ONLY:** Your entire output MUST be a single ```json block containing EXACTLY ONE JSON object inside. NO text outside, NO multiple objects.
2.  **ONE ACTION PER TURN:** You must choose EITHER ONE tool_call OR ONE final_answer. Never combine them or list multiple actions.
3.  **`is_final` DETERMINES FORMAT:** Use `is_final: true` for the final answer, `is_final: false` (or omit `is_final`) for tool calls.
4.  **ENGLISH LANGUAGE ONLY:** ALL text values inside the JSON (`thought`, `answer`, generated advice/stories, argument string values, etc.) MUST be strictly in ENGLISH. No exceptions.
5.  **VALID JSON:** Use double quotes for all keys and string values. Escape internal quotes properly (e.g., `\\\\"\\"\\\\"` for a quote inside a string). No trailing commas. Ensure the `answer` string is a valid JSON string value (no unescaped special characters or incorrect quoting).
6.  **`arguments` IS ALWAYS A DICT:** Even if a tool takes no arguments, provide `\\\\"arguments\\\\": {{}}`.
7.  **TOOL SCHEMA:** If a tool call fails due to incorrect arguments, you will receive the tool's schema. Use it to correct the `arguments` in your next attempt.
8.  **PROGRESSIVE REASONING:** Your 'thought' should explain your reasoning for the *current* step based on the request and previous observations.
9.  **AVOID REPETITION:** Do NOT repeat the same tool call with the exact same arguments if the previous attempt failed (e.g., timeout, error). Analyze the failure and **IMMEDIATELY try a different approach or tool** in the next step.
10. **STEP-BY-STEP COMPLETION:** Break down complex user requests into smaller, manageable steps. Ensure each step makes clear progress towards the final goal.
11. **MANDATORY ERROR CORRECTION:** If you receive an observation starting with \\\\"System Error: Tool Input Error...\\\\", you MUST READ the error message and Tool Schema carefully. In your VERY NEXT JSON response, you **MUST** correct the `arguments` based on the schema and error message. DO NOT perform any other action or ignore the error. Your primary task is to fix the tool call.

{tools_section}

Analyze the user request and previous steps. Decide the SINGLE NEXT STEP according to your plan and the rules above: either ONE tool call (`is_final: false`) or ONE final answer (`is_final: true`) if the entire task is complete.

FINAL REMINDER: Output MUST be in ENGLISH ONLY and contain EXACTLY ONE **VALID** JSON object for ONE action. Double-check quotes and escaping in the `answer` field.
"""
        return system_prompt_template 