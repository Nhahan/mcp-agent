import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent.parent / "prompts"

class PromptManager:
    def __init__(self):
        self.prompt_templates: Dict[str, str] = {}
        self._load_all_prompt_templates()

    def _load_prompt_template(self, template_name: str) -> str:
        """Loads a prompt template file from the prompts directory."""
        if template_name in self.prompt_templates:
            return self.prompt_templates[template_name]

        file_path = PROMPT_DIR / f"{template_name}.txt"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.prompt_templates[template_name] = content
                logger.info(f"Prompt template '{template_name}' loaded from {file_path}")
                return content
        except FileNotFoundError:
            logger.error(f"Prompt template file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompt template {file_path}: {e}", exc_info=True)
            raise

    def _load_all_prompt_templates(self):
        """Loads all required prompt templates into the cache."""
        required_templates = [
            "react_system_core",
            "react_system_examples",
            "react_system_final_reminder"
        ]
        logger.info(f"Loading required prompt templates: {required_templates}")
        for name in required_templates:
            try:
                self._load_prompt_template(name)
            except Exception:
                # Error is logged in _load_prompt_template, continue loading others
                pass
        logger.info("Finished loading prompt templates.")


    def build_react_system_prompt(self, tool_details: Dict[str, Dict], iteration: int = 1) -> str:
        """Builds the simplified ReAct system prompt using consolidated templates."""
        
        # 기본 프롬프트 - 모든 실패 상황에서 사용될 수 있는 최소한의 템플릿
        default_prompt = """You are Axistant Agent. Output a single JSON object and NOTHING ELSE.

Available Tools:
No tools available.

**REQUIRED JSON STRUCTURE:**
Your output MUST be a single JSON object containing these keys exactly: "thought" (string), "action" (string tool call OR null), "answer" (string final answer OR null).

**CRITICAL RULES - FOLLOW THESE STRICTLY:**
1. VALID JSON ONLY: Output ONLY a single, valid JSON object.
2. ACTION NULL IS DEFAULT: Default is `action: null` for simple interactions.
3. TOOL OR ANSWER - EXACTLY ONE: Exactly one non-null between `action` and `answer`.
4. ERROR HANDLING: Learn from errors and adapt your approach.

Generate a valid JSON response now.
"""
        
        try:
            # 템플릿 로딩 시도
            logger.info("Loading prompt templates for build_react_system_prompt...")
            
            try:
                core_prompt_template = self.prompt_templates.get("react_system_core", "")
                examples = self.prompt_templates.get("react_system_examples", "")
                final_reminder = self.prompt_templates.get("react_system_final_reminder", "")
                
                # 필수 템플릿 체크
                if not core_prompt_template:
                    logger.error("Missing 'react_system_core' template. Attempting to reload...")
                    core_prompt_template = self._load_prompt_template("react_system_core")
                    
                if not examples:
                    logger.warning("Missing 'react_system_examples' template. Attempting to reload...")
                    try:
                        examples = self._load_prompt_template("react_system_examples")
                    except:
                        examples = ""  # 예제는 없어도 작동 가능
                        
                if not final_reminder:
                    logger.warning("Missing 'react_system_final_reminder' template. Attempting to reload...")
                    try:
                        final_reminder = self._load_prompt_template("react_system_final_reminder")
                    except:
                        final_reminder = ""  # 리마인더는 없어도 작동 가능
                
            except Exception as e:
                logger.error(f"Error loading templates: {e}", exc_info=True)
                # 템플릿 로딩 실패 시 기본 템플릿 사용
                return default_prompt

            # Format available tools
            tools_str = "No tools available."
            if tool_details:
                formatted_tools = []
                for server, tools_info in tool_details.items():
                    # Ensure tools_info is a non-empty dict
                    if isinstance(tools_info, dict) and tools_info:
                        for tool_name, tool_spec in tools_info.items():
                            description = tool_spec.get('description', 'No description available')
                            formatted_tools.append(f"- `{server}/{tool_name}`: {description}")
                    elif not isinstance(tools_info, dict):
                         logger.warning(f"build_react_system_prompt: Expected dict for tools_info of server '{server}', but got {type(tools_info)}. Skipping server.")
                    else: # Empty dict
                        logger.debug(f"build_react_system_prompt: Tools dictionary for server '{server}' is empty. Skipping server.") # Debug level might be more appropriate
                if formatted_tools:
                     tools_str = "\n".join(formatted_tools)

            # Insert tools into the core prompt template
            try:
                if not core_prompt_template:
                    logger.error("Core prompt template is empty after loading attempt")
                    return default_prompt
                    
                logger.debug(f"Formatting core prompt template with tools_str (length: {len(tools_str)})")
                core_prompt_formatted = core_prompt_template.format(tools=tools_str.strip())
                logger.debug(f"Core prompt template formatted successfully (length: {len(core_prompt_formatted)})")
            except KeyError as e:
                 logger.error(f"Error formatting core prompt template. Missing key: {e}. Template content: {core_prompt_template[:500]}...", exc_info=True)
                 # 포맷팅 실패 시 기본 프롬프트 사용
                 return default_prompt
            except Exception as e:
                 logger.error(f"Unexpected error formatting core prompt: {e}", exc_info=True)
                 return default_prompt

            # Combine prompt parts
            prompt_parts = []
            
            if core_prompt_formatted:
                prompt_parts.append(core_prompt_formatted)
                
            if examples:
                prompt_parts.append(examples)
                
            if final_reminder:
                prompt_parts.append(final_reminder)
            
            # 모든 부분이 비어 있다면 기본 프롬프트 사용
            if not prompt_parts:
                logger.error("All prompt parts are empty")
                return default_prompt
                
            formatted_prompt = "\n\n".join(prompt_parts)
            logger.debug(f"Built ReAct system prompt for iteration {iteration}. Tool details provided: {'Yes' if tool_details else 'No'}. Prompt length: {len(formatted_prompt)}")
            
            if not formatted_prompt:
                logger.error("Formatted prompt is empty")
                return default_prompt
                
            return formatted_prompt

        except Exception as e:
            logger.error(f"Unexpected error in build_react_system_prompt: {e}", exc_info=True)
            # 예외 발생 시 기본 프롬프트 사용
            return default_prompt

    def build_thought_prompt(self, prompt: str, tools: Dict[str, Dict], error_messages: List[str], previous_steps: int, use_sequential_thinking: bool = False) -> str:
        """
        사고 단계 프롬프트를 생성합니다.
        이 프롬프트는 오직 'thought' 필드만 포함하는 JSON 응답을 생성해야 합니다.
        
        Args:
            prompt: 사용자 프롬프트
            tools: 사용 가능한 도구 정보 딕셔너리 (서버별 도구 및 설명)
            error_messages: 이전 오류 메시지 목록
            previous_steps: 이전 단계 수
            use_sequential_thinking: Sequential Thinking 사용 여부
            
        Returns:
            str: 생성된 프롬프트
        """
        # 도구 목록 문자열 포맷 (도구 이름과 설명 포함)
        tools_str = "No tools available."
        if tools:
            formatted_tools = []
            for server, tools_info in tools.items():
                if isinstance(tools_info, dict) and tools_info:
                    for tool_name, tool_spec in tools_info.items():
                        description = tool_spec.get('description', 'No description available')
                        # 도구 이름과 설명을 함께 포함
                        formatted_tools.append(f"- `{server}/{tool_name}`: {description}")
                elif not isinstance(tools_info, dict):
                    logger.warning(f"build_thought_prompt: Expected dict for tools_info of server '{server}', but got {type(tools_info)}. Skipping server.")
                else:  # Empty dict
                    logger.debug(f"build_thought_prompt: Tools dictionary for server '{server}' is empty. Skipping server.")
            
            if formatted_tools:
                tools_str = "\n".join(formatted_tools)
        
        # 오류 메시지 문자열 포맷
        errors_str = "\n".join([f"- {error}" for error in error_messages])
        
        # Sequential Thinking 관련 지시 추가
        sequential_thinking_guidance = """
Your task requires deep analysis. Use a Sequential Thinking approach:
1. Break down the problem step-by-step
2. Consider multiple aspects before reaching conclusions
3. Structure your thinking in a logical sequence
4. Identify and address potential issues in your reasoning
5. Revise earlier thoughts if necessary as your understanding evolves
""" if use_sequential_thinking else ""
        
        # 프롬프트 조합
        errors_part = f"Previous errors to address:\n{errors_str}\n" if error_messages else ""
        
        # JSON 예시 문자열
        thought_json_example = '{"thought": "Your detailed reasoning and plan here..."}'

        prompt_template = f"""
User question: {prompt}

Available tools:
{tools_str}

{sequential_thinking_guidance}

{errors_part}
Your task is to **analyze the user question and previous context** (including errors) and formulate a **plan or reasoning process** to address it. Consider if any tools are needed.

**Output ONLY a JSON object with a single key 'thought'**: {thought_json_example}
Do NOT include 'action' or 'answer' in this response.
"""
        
        return prompt_template
    
    def build_sequential_thought_prompt(self, prompt: str, previous_thoughts: List[str], tools: Dict[str, Dict], error_messages: List[str], current_thought_number: int, total_thoughts: int) -> str:
        """
        Sequential Thinking의 후속 사고 단계를 위한 프롬프트를 생성합니다.
        이 프롬프트는 오직 'thought' 필드만 포함하는 JSON 응답을 생성해야 합니다.
        
        Args:
            prompt: 사용자 프롬프트
            previous_thoughts: 이전 사고 목록
            tools: 사용 가능한 도구 정보 딕셔너리 (서버별 도구 및 설명)
            error_messages: 이전 오류 메시지 목록
            current_thought_number: 현재 사고 번호
            total_thoughts: 총 사고 단계 수
            
        Returns:
            str: 생성된 프롬프트
        """
        # 이전 사고 문자열 포맷
        previous_thoughts_str = "\n\n".join([f"Thought {i+1}: {thought}" for i, thought in enumerate(previous_thoughts)])
        
        # 도구 목록 문자열 포맷 (도구 이름과 설명 포함)
        tools_str = "No tools available."
        if tools:
            formatted_tools = []
            for server, tools_info in tools.items():
                if isinstance(tools_info, dict) and tools_info:
                    for tool_name, tool_spec in tools_info.items():
                        description = tool_spec.get('description', 'No description available')
                        # 도구 이름과 설명을 함께 포함
                        formatted_tools.append(f"- `{server}/{tool_name}`: {description}")
                elif not isinstance(tools_info, dict):
                    logger.warning(f"build_sequential_thought_prompt: Expected dict for tools_info of server '{server}', but got {type(tools_info)}. Skipping server.")
                else:  # Empty dict
                    logger.debug(f"build_sequential_thought_prompt: Tools dictionary for server '{server}' is empty. Skipping server.")
            
            if formatted_tools:
                tools_str = "\n".join(formatted_tools)
        
        # JSON 예시 문자열
        seq_thought_json_example = '{"thought": "Your next detailed thought step here..."}'
        
        # 프롬프트 조합
        prompt_template = f"""
User question: {prompt}

Available tools:
{tools_str}

Previous thoughts:
{previous_thoughts_str}

You are now at thought #{current_thought_number} of {total_thoughts}.
Continue your sequential thinking process by building on previous thoughts.
You can revise earlier thoughts if you spot issues or have new insights.
Aim for a comprehensive and logical analysis.

**Output ONLY a JSON object with a single key 'thought'**: {seq_thought_json_example}
Do NOT include 'action' or 'answer' in this response.
"""
        
        return prompt_template
    
    def build_action_prompt(self, prompt: str, thought: str, tools: Dict[str, Dict], error_messages: List[str], previous_steps: int) -> str:
        """
        행동 또는 답변 단계 프롬프트를 생성합니다.
        주어진 'thought'를 바탕으로 'action' 또는 'answer' 중 하나를 반드시 선택해야 합니다.
        
        Args:
            prompt: 사용자 프롬프트
            thought: 현재 사고 내용 (generate_thought의 결과)
            tools: 사용 가능한 도구 정보 딕셔너리 (서버별 도구 및 설명)
            error_messages: 이전 오류 메시지 목록
            previous_steps: 이전 단계 수
            
        Returns:
            str: 생성된 프롬프트
        """
        # 도구 목록 문자열 포맷 (도구 이름과 설명 포함)
        tools_str = "No tools available."
        if tools:
            formatted_tools = []
            for server, tools_info in tools.items():
                if isinstance(tools_info, dict) and tools_info:
                    for tool_name, tool_spec in tools_info.items():
                        description = tool_spec.get('description', 'No description available')
                        # 도구 이름과 설명을 함께 포함
                        formatted_tools.append(f"- `{server}/{tool_name}`: {description}")
                elif not isinstance(tools_info, dict):
                    logger.warning(f"build_action_prompt: Expected dict for tools_info of server '{server}', but got {type(tools_info)}. Skipping server.")
                else:  # Empty dict
                    logger.debug(f"build_action_prompt: Tools dictionary for server '{server}' is empty. Skipping server.")
            
            if formatted_tools:
                tools_str = "\n".join(formatted_tools)
        
        # 오류 분석 및 학습 포인트 추출
        error_analysis = ""
        if error_messages:
            error_analysis = "PREVIOUS ERROR ANALYSIS:\n"
            
            for i, error in enumerate(error_messages):
                error_analysis += f"- Error {i+1}: {error}\n"
                
                # 일반적인 오류 패턴 분석
                if "invalid_type" in error and "Required" in error:
                    # 필수 파라미터 누락 패턴
                    error_analysis += "  * INSIGHT: A required parameter is missing in your request.\n"
                    
                    # 파라미터 이름 추출 시도
                    import re
                    param_matches = re.findall(r'"path":\s*\[\s*"([^"]+)"\s*\]', error)
                    if param_matches:
                        for param in param_matches:
                            error_analysis += f"  * SOLUTION: Include the '{param}' parameter which is REQUIRED.\n"
                    
                    # 매개변수 이름 불일치 분석
                    if "filename" in error or "filename" in thought.lower():
                        error_analysis += "  * CORRECTION: You might be using 'filename' when the tool actually requires 'path'.\n"
                    
                elif "invalid_type" in error:
                    # 타입 불일치 오류
                    expected = re.search(r'"expected":\s*"([^"]+)"', error)
                    received = re.search(r'"received":\s*"([^"]+)"', error)
                    
                    if expected and received:
                        error_analysis += f"  * INSIGHT: Type mismatch - expected {expected.group(1)} but received {received.group(1)}.\n"
                        error_analysis += f"  * SOLUTION: Ensure your parameter is of type {expected.group(1)}.\n"
                
                elif "key must be a string" in error:
                    error_analysis += "  * INSIGHT: JSON format issue - all keys in JSON must be strings.\n"
                    error_analysis += "  * SOLUTION: Check your JSON formatting and ensure all keys are properly quoted.\n"
                
                # 이전 시도에서 잘못된 정보 추출
                if "desktop-commander/write_file" in thought and "filename" in thought:
                    error_analysis += "  * CRITICAL CORRECTION: For file operations, use 'path' instead of 'filename' parameter.\n"
            
            error_analysis += "\nLEARNING POINTS:\n"
            error_analysis += "1. ALWAYS check the tool's description carefully for required parameters.\n"
            error_analysis += "2. Parameter names must EXACTLY match what the tool expects (e.g., 'path' vs 'filename').\n"
            error_analysis += "3. Learn from previous errors and adapt your approach accordingly.\n\n"
        
        # JSON 예시 문자열
        action_example = '{"action": "server_name/tool_name::{\\"param1\\": \\"value1\\", \\"param2\\": \\"value2\\"}", "answer": null, "should_end_conversation": false}'
        answer_example = '{"action": null, "answer": "Your final response to the user...", "should_end_conversation": true}'
        
        prompt_template = f"""
User question: {prompt}

Your reasoning/plan (thought):
--- BEGIN THOUGHT ---
{thought}
--- END THOUGHT ---

Available tools:
{tools_str}

{error_analysis}
Based on the thought process above AND applying lessons from any previous errors, decide your next step:

1. **First, determine if this conversation requires a tool or should end with a direct answer:**
   * Set `should_end_conversation` to `true` if:
      - This is a simple greeting or query that needs only a direct answer
      - The user's request is inappropriate and needs no further processing
      - The thought process clearly concludes no tool is needed
   * Set `should_end_conversation` to `false` if a tool must be used to complete the request

2. **Then, based on that decision:**
   * **If should_end_conversation is TRUE:**
      * Set `action` to `null`
      * Provide an appropriate `answer` in the answer field
      
   * **If should_end_conversation is FALSE:**
      * Study the tool descriptions CAREFULLY to identify the EXACT tool name and its required parameters
      * Pay close attention to parameter names - they must match EXACTLY what the tool expects
      * If previous attempts failed, ensure you've corrected the issues identified in the error analysis
      * Construct a precise JSON arguments string with ALL required parameters correctly named
      * Set `action` to: `"server_name/tool_name::{{{{properly_formatted_json_args}}}}"`
      * Set `answer` to `null`

**TOOL USAGE BEST PRACTICES:**
* CAREFULLY READ each tool's description to identify ALL required parameters
* PARAMETER NAMES MATTER - use exactly the names mentioned in the tool's description
* If you're using a file-related tool, check if it needs 'path' (not 'filename')
* Learn from previous errors and don't repeat the same mistakes
* When you encounter an error, analyze what went wrong and adjust your approach

**CRITICAL RULES:**
* Output MUST be valid JSON with `action`, `answer`, AND `should_end_conversation` fields
* If `should_end_conversation` is true, `action` MUST be null and `answer` MUST be non-null
* If `should_end_conversation` is false, `action` MUST be non-null with the correct format and `answer` MUST be null
* ALWAYS include ALL REQUIRED parameters for the tool you're using
* Example format: {action_example} OR {answer_example}

Output the JSON object for the immediate next step now.
"""
        
        return prompt_template
