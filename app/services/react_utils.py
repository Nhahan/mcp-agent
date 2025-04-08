import logging
import re
import json
from pathlib import Path
from typing import Any, Tuple, Optional

logger = logging.getLogger(__name__)

def detect_language(text: str) -> str:
    """Detects if the primary language of the text is Korean ('ko') or English ('en')."""
    if not text or len(text.strip()) < 2:
        return "en"
    korean_char_count = len([c for c in text if ord('가') <= ord(c) <= ord('힣') or
                             ord('ㄱ') <= ord(c) <= ord('ㅎ') or
                             ord('ㅏ') <= ord(c) <= ord('ㅣ')])
    if korean_char_count > 0 and korean_char_count / len(text.strip()) > 0.1:
        return "ko"
    return "en"

def format_tool_result(result: Any) -> str:
    """Formats the result from an MCP tool into a string."""
    if result is None:
        return "No result returned from tool."

    try:
        # Structured content processing
        if isinstance(result, dict) and "content" in result:
            content = result["content"]

            # List content: combine parts
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if "text" in item:
                            parts.append(str(item["text"]))
                        # Extend here for other content types if needed
                    else:
                        parts.append(str(item))
                return " ".join(parts).strip()

            # Single content item
            if isinstance(content, dict) and "text" in content:
                return str(content["text"]).strip()

            # Direct string or other content
            return str(content).strip()

        # String result
        if isinstance(result, str):
            return result.strip()

        # Other types (int, float, bool)
        if isinstance(result, (int, float, bool)):
            return str(result)

        # General object string conversion
        return str(result).strip()

    except Exception as e:
        logger.error(f"Error formatting tool result: {e}", exc_info=True)
        return f"Error formatting tool result: {str(e)}"

def clean_response_json(text: str) -> Tuple[str, Optional[str]]:
    """
    Clean and extract JSON from text.
    
    Returns:
        Tuple[str, Optional[str]]: The extracted JSON string and an optional error message.
        If successful, the error message will be None.
    """
    if not text:
        return "", "Empty response text"
    
    # 1. First try to find JSON blocks in markdown format (```json ... ```)
    json_block_pattern = re.compile(r"```json\s*({.*?})\s*```", re.DOTALL | re.IGNORECASE)
    matches = json_block_pattern.findall(text)
    
    if matches:
        logger.debug(f"Found {len(matches)} potential JSON blocks in response")
        # Try each match until we find a valid JSON
        for i, json_candidate in enumerate(matches):
            try:
                # Verify it's valid JSON
                json.loads(json_candidate)
                logger.debug(f"Successfully extracted valid JSON from block {i+1}")
                return json_candidate, None
            except json.JSONDecodeError as e:
                logger.warning(f"JSON block {i+1} was invalid: {e}")
                continue
    
    # 2. Try to extract from any code block
    code_block_pattern = re.compile(r"```(?:json|js|javascript|python)?\s*([\s\S]*?)\s*```")
    code_blocks = code_block_pattern.findall(text)
    
    if code_blocks:
        logger.debug(f"Extracting from {len(code_blocks)} code blocks")
        for i, block in enumerate(code_blocks):
            try:
                # Remove any leading/trailing whitespace
                clean_block = block.strip()
                json.loads(clean_block)
                logger.debug(f"Valid JSON found in code block {i+1}")
                return clean_block, None
            except json.JSONDecodeError as e:
                logger.debug(f"Code block {i+1} is not valid JSON: {e}")
                continue
    
    # 3. Try to find the outer-most curly braces pattern
    logger.debug("Trying to extract JSON directly using brace matching")
    # Find all potential JSON objects (outermost balanced braces)
    pattern = re.compile(r"({(?:[^{}]|(?:\{[^{}]*\}))*})", re.DOTALL)
    matches = pattern.findall(text)
    
    if matches:
        for i, match in enumerate(matches):
            try:
                # Some basic cleanup: remove trailing commas and fix common issues
                cleaned = re.sub(r',\s*}', '}', match)
                json_obj = json.loads(cleaned)
                # Verify it has the expected structure
                if isinstance(json_obj, dict) and any(key in json_obj for key in ["action", "answer", "thought"]):
                    logger.debug(f"Valid ReAct JSON found using brace matching (candidate {i+1})")
                    return cleaned, None
            except json.JSONDecodeError as e:
                logger.debug(f"JSON candidate {i+1} from braces is not valid: {e}")
                continue
    
    # 4. Final attempt with line-by-line scanning for a complete JSON object
    logger.debug("Trying line-by-line JSON reconstruction")
    lines = text.split('\n')
    start_index = -1
    end_index = -1
    
    # Find potential start and end of a JSON object
    for i, line in enumerate(lines):
        if '{' in line and start_index == -1:
            start_index = i
        if '}' in line and start_index != -1:
            end_index = i
            
            # Try the current range
            potential_json = '\n'.join(lines[start_index:end_index+1])
            try:
                # Extract just the JSON part
                json_parts = re.search(r'({.*})', potential_json, re.DOTALL)
                if json_parts:
                    json_str = json_parts.group(1)
                    # Try to parse it
                    json.loads(json_str)
                    logger.debug(f"Valid JSON found with line scanning from line {start_index+1} to {end_index+1}")
                    return json_str, None
            except json.JSONDecodeError:
                # Continue searching - this might be a nested brace
                continue
    
    # Last resort - try to salvage from any chunks
    try:
        logger.debug("Applying last-resort JSON extraction")
        # Remove any markdown indicators, indentation, or decorations
        simplified = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # Remove code blocks
        simplified = re.sub(r'`|#|\*|>|\|', '', simplified)  # Remove markdown symbols
        
        # Look for the most promising JSON-like structure
        structures = re.findall(r'({.*?})', simplified, re.DOTALL)
        for struct in structures:
            try:
                struct = re.sub(r',\s*}', '}', struct)  # Fix trailing commas
                json_obj = json.loads(struct)
                if isinstance(json_obj, dict) and any(key in json_obj for key in ["action", "answer", "thought"]):
                    logger.debug("Salvaged JSON from text after cleaning")
                    return struct, None
            except:
                continue
    except Exception as e:
        logger.error(f"Error during last-resort JSON extraction: {e}")
    
    # If we get here, we couldn't find valid JSON
    logger.error(f"Failed to extract valid JSON from text: {text[:200]}...")
    return "", "Failed to extract valid JSON from response"

def save_step_log(log_dir: Path, step: int, step_type: str, data: dict):
    """Saves a ReAct step log as a JSON file."""
    if not log_dir.exists():
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create log directory {log_dir}: {e}")
            return # Cannot save if directory creation fails

    try:
        filename = f"{step:02d}_{step_type}.json"
        log_file_path = log_dir / filename
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save log file {log_file_path}: {e}")
