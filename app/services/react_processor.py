import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Tuple, List, Dict, Optional
import uuid

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.services.mcp_service import MCPService
from app.mcp_client.client import MCPError
from app.services.llm_interface import LLMInterface
from app.services.prompt_manager import PromptManager
# 유틸리티 함수 임포트
from app.services.react_utils import (
    detect_language,
    format_tool_result,
    clean_response_json,
)
from app.services.sequential_thinking import get_sequential_thinking

logger = logging.getLogger(__name__)

class ReactProcessor:
    def __init__(self,
                 settings: Settings = Depends(get_settings),
                 mcp_service: MCPService = Depends(),
                 llm_interface: LLMInterface = Depends(),
                 prompt_manager: PromptManager = Depends()
                ):
        self.settings = settings
        self.mcp_service = mcp_service
        self.llm_interface = llm_interface
        self.prompt_manager = prompt_manager
        
        # 최대 반복 횟수 - 설정에서 가져오기
        self.max_iterations = getattr(settings, 'react_max_iterations', 5)
        
        # 로그 디렉토리 설정
        self.logs_dir = Path('/app/logs') if settings.is_docker else Path('./logs')
        self.react_logs_dir = self.logs_dir / "react_logs"

        # 시스템 프롬프트 템플릿 초기화
        self.initial_prompt_template = ""
        try:
            # MCP 도구 정보 가져오기
            logger.info("Attempting to get MCP tool details...")
            try:
                tool_details = self.mcp_service.get_all_tool_details()
                logger.info(f"Retrieved tool details: {len(tool_details)} servers available")
            except Exception as tool_e:
                logger.error(f"Failed to get tool details: {tool_e}", exc_info=True)
                logger.info("Proceeding with empty tool details")
                tool_details = {}  # 빈 도구 정보로 계속 진행
                
            # 시스템 프롬프트 생성
            logger.info("Building system prompt...")
            if not tool_details:
                logger.warning("No tool details available. System prompt will have empty tools list.")
                
            try:
                self.initial_prompt_template = self.prompt_manager.build_react_system_prompt(tool_details)
                logger.info(f"Initial prompt template loaded successfully. Length: {len(self.initial_prompt_template)}")
                
                # 템플릿이 비어있는지 검증
                if not self.initial_prompt_template:
                    logger.error("build_react_system_prompt returned an empty template")
                    self.initial_prompt_template = "You are Axistant Agent. Output a single JSON object and NOTHING ELSE.\n\nAvailable Tools:\nNo tools available.\n\n**REQUIRED JSON STRUCTURE:**\nYour output MUST be a single JSON object containing these keys exactly: \"thought\" (string), \"action\" (string tool call OR null), \"answer\" (string final answer OR null)."
                    logger.info("Using fallback template")
            except Exception as prompt_e:
                logger.error(f"Failed to build system prompt: {prompt_e}", exc_info=True)
                # 기본 프롬프트 템플릿 설정
                self.initial_prompt_template = "You are Axistant Agent. Output a single JSON object and NOTHING ELSE.\n\nAvailable Tools:\nNo tools available.\n\n**REQUIRED JSON STRUCTURE:**\nYour output MUST be a single JSON object containing these keys exactly: \"thought\" (string), \"action\" (string tool call OR null), \"answer\" (string final answer OR null)."
                logger.info("Using fallback template")
                
        except Exception as e:
            logger.error(f"Failed to load initial prompt template: {e}", exc_info=True)
            # 기본 프롬프트 템플릿 설정
            self.initial_prompt_template = "You are Axistant Agent. Output a single JSON object and NOTHING ELSE.\n\nAvailable Tools:\nNo tools available.\n\n**REQUIRED JSON STRUCTURE:**\nYour output MUST be a single JSON object containing these keys exactly: \"thought\" (string), \"action\" (string tool call OR null), \"answer\" (string final answer OR null)."
            logger.info("Using fallback template due to exception")
        
        logger.info("ReactProcessor initialized")

    async def process_react_pattern(self, prompt: str, session_id: str = None) -> Tuple[str, List[Dict], str, Optional[str]]:
        """
        이전 코드와의 호환성을 위한 메서드로, process 메서드로 요청을 위임합니다.
        
        Args:
            prompt: 사용자 프롬프트
            session_id: 선택적 세션 ID
            
        Returns:
            Tuple[str, List[Dict], str, Optional[str]]: 최종 답변, 사고/행동 기록, 대화 로그, 오류 메시지
        """
        logger.info(f"process_react_pattern 호출 (세션 ID: {session_id})")
        
        # 언어 감지
        language = detect_language(prompt)
        
        # 세션 디렉토리 생성
        session_dir = self.react_logs_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # process 메서드 호출
        result = await self.process(prompt, session_id=session_id)
        
        # 결과 변환
        final_answer = result.get("answer", "")
        thoughts_actions = result.get("trace", [])
        
        # 대화 로그 구성
        conversation_parts = [f"User: {prompt}"]
        
        for step in result.get("step_logs", []):
            if step.get("thought"):
                conversation_parts.append(f"Thought: {step.get('thought')}")
            if step.get("action"):
                conversation_parts.append(f"Action: {step.get('action')}")
            if step.get("action_result"):
                conversation_parts.append(f"Observation: {step.get('action_result')}")
            if step.get("answer"):
                conversation_parts.append(f"Answer: {step.get('answer')}")
        
        conversation_log = "\n\n".join(conversation_parts)
        
        # full_response 필드가 없으면 생성한 대화 로그 사용
        if "full_response" not in result:
            result["full_response"] = conversation_log
        
        # Sequential Thinking 정보 추가
        result["sequential_thinking_logs"] = self._get_sequential_thinking_logs(session_dir)
        
        # meta.json 파일 저장
        self._save_meta(session_dir, prompt, language, result, session_id)
        
        error_message = None
        
        # 오류 메시지 확인
        if result.get("error_messages"):
            error_message = "; ".join(result.get("error_messages"))
        
        return final_answer, thoughts_actions, result.get("full_response", conversation_log), error_message
    
    def _get_sequential_thinking_logs(self, log_dir: Path) -> List[Dict]:
        """
        Sequential Thinking 로그 파일들을 읽어 리스트로 반환합니다.
        
        Args:
            log_dir: 로그 디렉토리 경로
            
        Returns:
            List[Dict]: Sequential Thinking 로그 목록
        """
        sequential_logs = []
        
        try:
            for log_file in log_dir.glob("*_sequential_thinking.json"):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)
                        sequential_logs.append(log_data)
                except Exception as e:
                    logger.error(f"Failed to read Sequential Thinking log file {log_file}: {e}")
        except Exception as e:
            logger.error(f"Failed to read Sequential Thinking logs from {log_dir}: {e}")
        
        # 생각 번호로 정렬
        sequential_logs.sort(key=lambda x: x.get("sequential_step", 0))
        
        return sequential_logs
    
    def _save_meta(self, log_dir: Path, prompt: str, language: str, result: Dict, session_id: str) -> None:
        """
        ReAct 세션의 메타 정보를 저장합니다.
        Sequential Thinking 로그를 포함합니다.
        
        Args:
            log_dir: 로그 디렉토리 경로
            prompt: 사용자 프롬프트
            language: 언어 코드
            result: 처리 결과 딕셔너리
            session_id: 세션 ID
        """
        try:
            # 사용 가능한 도구 목록 가져오기
            available_tools = {}
            try:
                mcp_tools = self.mcp_service.get_all_tools()
                for server_name, tools in mcp_tools.items():
                    available_tools[server_name] = list(tools.keys())
            except Exception as e:
                logger.error(f"Failed to get MCP tools: {e}")
            
            # Sequential Thinking 기록 가져오기
            sequential_thinking_instance = get_sequential_thinking()
            # 주의: 현재 싱글톤 인스턴스이므로, 다른 세션의 기록이 섞일 수 있음
            # 추후 세션별 인스턴스 관리 또는 ID 기반 필터링 필요
            sequential_history = sequential_thinking_instance.thought_history
            
            # 메타 데이터 구성
            meta_data = {
                "start_time": datetime.now().isoformat(),
                "prompt": prompt,
                "model_path": str(self.llm_interface.model_path),
                "session_id": session_id,
                "start_language": language,
                "available_tools": available_tools,
                "mode": "Enhanced_ReAct_Sequential_2Step", # 모드 이름 수정
                "end_time": datetime.now().isoformat(),
                "success": not bool(result.get("error")),
                "reason": "completed" if not result.get("error") else "error",
                "iterations_completed": result.get("steps", 0),
                "final_error": result.get("error"),
                "structured_react_trace": result.get("trace", []),
                "react_history": result.get("full_response", ""),
                "sequential_thinking_details": [
                    {
                        "step": data.get("thoughtNumber", idx + 1),
                        "total_thoughts": data.get("totalThoughts"),
                        "thought": data.get("thought"),
                        "is_revision": data.get("isRevision"),
                        "revises_thought": data.get("revisesThought"),
                        "branch_from": data.get("branchFromThought"),
                        "branch_id": data.get("branchId"),
                        "next_needed": data.get("nextThoughtNeeded")
                    }
                    for idx, data in enumerate(sequential_history) if isinstance(data, dict)
                ]
            }
            
            # 메타 파일 저장
            meta_file_path = log_dir / "meta.json"
            with open(meta_file_path, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved meta.json for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to save meta.json: {e}", exc_info=True)

    def _validate_action_format(self, action: str, available_tools: List[str]) -> bool:
        """Validate the format of an action string."""
        if action == "null":
            return True
            
        # 기본 형식 확인: server/tool::args
        if "::" not in action:
            return False
            
        # 툴 이름 형식 확인
        tool_part = action.split("::", 1)[0]
        if "/" not in tool_part:
            return False
            
        # 사용 가능한 도구인지 확인
        if tool_part not in available_tools:
            return False
            
        return True

    async def generate_thought(self, prompt: str, tools: List[str], error_messages: List[str], current_step: int, session_id: str) -> Tuple[str, Optional[str]]:
        """
        사고 단계를 생성합니다. 내장 SequentialThinking을 사용하여 고정된 2단계 사고를 수행합니다.
        """
        try:
            # 내장형 Sequential Thinking 인스턴스 가져오기
            sequential_thinking = get_sequential_thinking()
            
            # 도구 이름 목록을 도구 상세 정보 딕셔너리로 변환
            tool_details = {}
            try:
                for tool_path in tools:
                    if "/" in tool_path:
                        server_name, tool_name = tool_path.split("/", 1)
                        if server_name not in tool_details:
                            tool_details[server_name] = {}
                        # MCP 서비스에서 도구 상세 정보 가져오기
                        tool_info = self.mcp_service.get_tool_details(server_name, tool_name)
                        if tool_info:
                            tool_details[server_name][tool_name] = tool_info
                        else:
                            # 상세 정보를 가져올 수 없는 경우 기본 설명 제공
                            tool_details[server_name][tool_name] = {"description": "No detailed description available"}
            except Exception as e:
                logger.error(f"Error converting tools to tool_details: {e}")
                # 오류 발생 시 기본 설명 제공
                tool_details = {
                    "unknown": {
                        "unknown_tool": {"description": "Error fetching tool details"}
                    }
                }
            
            # Sequential Thinking 사용 (고정된 2단계 사고)
            max_sequential_steps = 2  # 고정된 2단계
            thought_number = 1
            thoughts_history = []
            
            # 첫 번째 사고 단계 프롬프트 생성
            initial_thought_prompt = self.prompt_manager.build_thought_prompt(
                prompt=prompt,
                tools=tool_details,  # 도구 상세 정보 딕셔너리 전달
                error_messages=error_messages,
                previous_steps=current_step - 1,
                use_sequential_thinking=True
            )
            
            logger.debug(f"Sequential Thinking - Step {thought_number} Prompt (len:{len(initial_thought_prompt)}): {initial_thought_prompt[:200]}...")
            
            # 첫 번째 LLM 호출
            last_llm_response_text = await self.llm_interface.call_llm(initial_thought_prompt)
            logger.debug(f"Sequential Thinking - Step {thought_number} LLM Response: {last_llm_response_text[:200]}...")
            
            # Sequential Thinking 반복 (thought_number ~ max_sequential_steps)
            while thought_number <= max_sequential_steps:
                logger.info(f"Sequential Thinking - Processing thought {thought_number}/{max_sequential_steps}")
                
                # LLM 응답 파싱 (JSON 형식 기대)
                current_thought = ""
                try:
                    json_str, extract_error = clean_response_json(last_llm_response_text)
                    if extract_error:
                        logger.warning(f"Could not extract JSON in seq step {thought_number}. Using raw text. Error: {extract_error}")
                        current_thought = last_llm_response_text 
                        next_thought_needed = (thought_number < max_sequential_steps) 
                    else:
                        parsed_data = json.loads(json_str)
                        if "thought" not in parsed_data or not parsed_data["thought"]:
                             logger.warning(f"'thought' field missing or empty in seq step {thought_number}. JSON: {json_str}")
                             current_thought = json_str
                             next_thought_needed = (thought_number < max_sequential_steps) 
                        else:
                            current_thought = parsed_data["thought"]
                            next_thought_needed = parsed_data.get("nextThoughtNeeded", (thought_number < max_sequential_steps)) 
                            logger.debug(f"Parsed thought {thought_number}: {current_thought[:100]}... Next needed: {next_thought_needed}")
                
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON in seq step {thought_number}: {e}. Using raw text.")
                    current_thought = last_llm_response_text
                    next_thought_needed = (thought_number < max_sequential_steps)
                except Exception as e:
                     logger.error(f"Unexpected error parsing thought {thought_number}: {e}", exc_info=True)
                     current_thought = f"Error processing thought {thought_number}: {e}"
                     next_thought_needed = False 
                
                thoughts_history.append(current_thought)
                
                try:
                    thought_input_data = {
                        "thought": current_thought,
                        "thoughtNumber": thought_number,
                        "totalThoughts": max_sequential_steps,
                        "nextThoughtNeeded": next_thought_needed
                    }
                    sequential_thinking.process_thought(thought_input_data)
                except Exception as st_e:
                    logger.error(f"Error calling sequential_thinking.process_thought: {st_e}")
                
                if not next_thought_needed or thought_number == max_sequential_steps:
                    break
                
                try:
                    next_thought_prompt = self.prompt_manager.build_sequential_thought_prompt(
                        prompt=prompt,
                        previous_thoughts=thoughts_history,
                        tools=tool_details,  # 도구 상세 정보 딕셔너리 전달
                        error_messages=error_messages,
                        current_thought_number=thought_number + 1,
                        total_thoughts=max_sequential_steps
                    )
                    logger.debug(f"Sequential Thinking - Step {thought_number + 1} Prompt (len:{len(next_thought_prompt)}): {next_thought_prompt[:200]}...")
                    last_llm_response_text = await self.llm_interface.call_llm(next_thought_prompt)
                    logger.debug(f"Sequential Thinking - Step {thought_number + 1} LLM Response: {last_llm_response_text[:200]}...")
                except Exception as e:
                    logger.error(f"Error during sequential thought {thought_number + 1} LLM call: {e}", exc_info=True)
                    break
                
                thought_number += 1
            
            final_combined_thought = "\n\n".join([
                f"Step {i+1}: {thought_text}"
                for i, thought_text in enumerate(thoughts_history)
            ])
            logger.info(f"Sequential Thinking (2 steps) completed. Final combined thought generated.")
            
            final_result_json = json.dumps({"thought": final_combined_thought})
            return final_result_json, None

        except Exception as e:
            logger.error(f"사고 생성 중 예상치 못한 오류: {e}", exc_info=True)
            return "", str(e)

    async def generate_action_or_answer(self, prompt: str, thought: str, tools: List[str], error_messages: List[str], current_step: int) -> Tuple[str, Optional[str]]:
        """
        행동 단계 또는 최종 응답 생성. 
        
        이 단계에서는 'thought'의 결론을 바탕으로 실행할 action 또는 최종 answer를 결정합니다.
        
        Args:
            prompt: 사용자 프롬프트
            thought: 생성된 사고 과정
            tools: 사용 가능한 도구 목록
            error_messages: 이전 오류 메시지 목록
            current_step: 현재 단계 수
            
        Returns:
            Tuple[str, Optional[str]]: 생성된 JSON 문자열, 오류 메시지(있을 경우)
        """
        try:
            # 도구 이름 목록을 도구 상세 정보 딕셔너리로 변환
            tool_details = {}
            for tool_path in tools:
                if "/" in tool_path:
                    server_name, tool_name = tool_path.split("/", 1)
                    if server_name not in tool_details:
                        tool_details[server_name] = {}
                    # MCP 서비스에서 도구 상세 정보 가져오기
                    tool_info = self.mcp_service.get_tool_details(server_name, tool_name)
                    if tool_info:
                        tool_details[server_name][tool_name] = tool_info
                    else:
                        # 상세 정보를 가져올 수 없는 경우 기본 설명 제공
                        tool_details[server_name][tool_name] = {"description": "No detailed description available"}
            
            # 행동 프롬프트 생성
            action_prompt = self.prompt_manager.build_action_prompt(
                prompt=prompt,
                thought=thought,
                tools=tool_details,  # 도구 상세 정보 딕셔너리 전달
                error_messages=error_messages,
                previous_steps=current_step - 1
            )
            
            # LLM 호출하여 행동 생성
            llm_response = await self.llm_interface.call_llm(action_prompt)
            
            return llm_response, None
        except Exception as e:
            logger.error(f"행동 생성 중 예상치 못한 오류: {e}", exc_info=True)
            return "", str(e)

    async def process(self, user_input: str, mcp_config: Optional[Dict] = None, session_id: Optional[str] = None) -> Dict:
        """
        Process a user input and generate a response using the ReAct pattern.
        """
        start_time = time.time()
        trace: List[Dict] = []
        language = detect_language(user_input)
        
        session_id = session_id or str(uuid.uuid4().int)
        session_dir = self.react_logs_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # MCP Service에서 도구 목록 가져오기
        tools = []
        try:
            mcp_tools = self.mcp_service.get_all_tools()
            for server_name, server_tools in mcp_tools.items():
                for tool_name in server_tools.keys():
                    tools.append(f"{server_name}/{tool_name}")
        except Exception as e:
            logger.error(f"Error getting tools from MCP service: {e}")
            tools = []
        
        # Previous answers cache to detect loops
        previous_responses = set()
        
        # Initialize state for this conversation
        state = {
            "user_input": user_input,
            "tools": tools,
            "language": language,
            "mcp_config": mcp_config or {}
        }
        
        # Generate first prompt
        if not self.initial_prompt_template:
            logger.error("Initial prompt template not loaded")
            return {"error": "Initial prompt template not loaded"}
        
        try:
            # 템플릿 포맷팅 시도
            prompt = self.initial_prompt_template
            
            # 문자열 내에 {user_input}, {tools}, {language} 등의 키워드가 있다면 직접 치환
            prompt = prompt.replace("{user_input}", user_input)
            prompt = prompt.replace("{language}", language)
            prompt = prompt.replace("{tools}", ", ".join(tools))
        except Exception as e:
            logger.error(f"Error formatting prompt template: {e}", exc_info=True)
            # 포맷팅 실패시 템플릿 그대로 사용
            prompt = self.initial_prompt_template
        
        # Main ReAct loop
        max_steps = self.max_iterations
        step = 0
        result = None
        
        step_logs = []
        
        logger.info(f"ReAct process starting for session {session_id}") # 시작 로그 추가
        
        while step < max_steps:
            step += 1
            logger.info(f"Step {step}/{max_steps}")
            
            current_error_messages = []
            if trace:
                last_step = trace[-1]
                if isinstance(last_step, dict) and isinstance(last_step.get('observation'), dict) and last_step['observation'].get('error'):
                    error_detail = last_step['observation']['error']
                    # 이전 단계의 오류 메시지를 다음 생각 생성에 전달
                    current_error_messages.append(f"Previous step error: {error_detail}") 
                    logger.warning(f"Error from previous step passed to thought generation: {error_detail}")
                    
            # --- 1. Generate Thought --- 
            thought_json_str, thought_error = await self.generate_thought(
                prompt=user_input, 
                tools=tools,
                error_messages=current_error_messages, # 수정된 오류 메시지 전달
                current_step=step,
                session_id=session_id # 세션 ID 전달 확인
            )

            if thought_error:
                logger.error(f"Step {step}: Error during thought generation: {thought_error}")
                # 오류 발생 시 fallback 처리 또는 루프 중단
                result = {"error": f"Thought generation failed: {thought_error}", "answer": "An error occurred during processing. Please try again."}
                break
            
            # Parse thought
            try:
                thought_data = json.loads(thought_json_str)
                current_thought = thought_data.get("thought")
                if not current_thought:
                    raise ValueError("Thought field is missing or empty")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Step {step}: Failed to parse thought JSON or missing thought: {e}. Raw: {thought_json_str}")
                result = {"error": f"Invalid thought format: {e}", "answer": "An error occurred during processing. Please try again."}
                break

            # Log the thought
            step_log = {
                "step": step,
                "input": user_input if step == 1 else None,
                "thought": current_thought,
                "action": None,
                "action_result": None,
                "answer": None
            }
            
            # --- 2. Generate Action or Answer (based on thought) --- 
            action_answer_json_str, action_answer_error = await self.generate_action_or_answer(
                prompt=user_input, 
                thought=current_thought,
                tools=tools,
                error_messages=current_error_messages, # 오류 메시지 전달
                current_step=step
            )
            
            if action_answer_error:
                logger.error(f"Step {step}: Error during action/answer generation: {action_answer_error}. Thought was: {current_thought}")
                 # 오류 발생 시 fallback 처리 또는 루프 중단
                result = {"error": f"Action/Answer generation failed: {action_answer_error}", "answer": "An error occurred during processing. Please try again."}
                break
            
            # Parse action/answer
            try:
                # JSON 응답 추출
                json_str, extract_error = clean_response_json(action_answer_json_str)
                if extract_error:
                    error_msg = f"Failed to extract JSON from LLM response: {extract_error}"
                    logger.error(f"Step {step}: {error_msg}")
                    current_error_messages.append(error_msg)
                    continue  # 다음 단계로 진행
                
                # JSON 파싱
                parsed_data = json.loads(json_str)
                
                # 새로운 필드 should_end_conversation 확인
                should_end_conversation = parsed_data.get("should_end_conversation", False)
                
                # action과 answer 필드 검증
                action_value = parsed_data.get("action")
                answer_value = parsed_data.get("answer")
                
                # 필수 필드 검증
                if action_value is None and answer_value is None:
                    error_msg = "Both 'action' and 'answer' fields are null. This should be prevented by the prompt."
                    logger.error(error_msg)
                    
                    # 대화를 종료해야 한다고 판단되었을 경우 기본 응답 제공
                    if should_end_conversation:
                        logger.info(f"Step {step}: Conversation should end but no answer provided. Using default response.")
                        result = {"answer": "I'm here to help with your requests.", "should_end_conversation": True}
                        step_log["answer"] = result["answer"]
                        trace.append({"thought": current_thought, "answer": result["answer"]})
                        step_logs.append(step_log)
                        break  # 루프 종료
                    
                    current_error_messages.append(error_msg)
                    continue  # 다음 단계로 진행
                
                # should_end_conversation이 true인데 action이 null이 아니거나 answer가 null인 경우
                if should_end_conversation:
                    if action_value is not None or answer_value is None:
                        error_msg = "When should_end_conversation is true, action must be null and answer must be non-null."
                        logger.error(f"Step {step}: {error_msg}")
                        current_error_messages.append(error_msg)
                        continue  # 다음 단계로 진행
                    
                    # 대화 종료 처리
                    logger.info(f"Step {step}: Conversation ended with answer: {answer_value}")
                    result = {"answer": answer_value, "should_end_conversation": True}
                    step_log["answer"] = answer_value
                    trace.append({"thought": current_thought, "answer": answer_value})
                    step_logs.append(step_log)
                    break  # 루프 종료
                
                # should_end_conversation이 false인데 action이 null이거나 answer가 null이 아닌 경우
                elif not should_end_conversation:
                    if action_value is None or answer_value is not None:
                        error_msg = "When should_end_conversation is false, action must be non-null and answer must be null."
                        logger.error(f"Step {step}: {error_msg}")
                        current_error_messages.append(error_msg)
                        continue  # 다음 단계로 진행
                
                # action과 answer가 둘 다 non-null인 경우(구버전 호환성을 위한 처리)
                if action_value is not None and answer_value is not None:
                    logger.warning(f"Step {step}: Both 'action' and 'answer' fields are non-null. Prioritizing 'answer'.")
                    action_value = None  # action 무시
                
                # 최종 응답이 있는 경우 처리
                if answer_value is not None:
                    logger.info(f"Step {step}: Received final answer.")
                    result = {"answer": answer_value, "should_end_conversation": True}
                    step_log["answer"] = answer_value
                    trace.append({"thought": current_thought, "answer": answer_value})
                    step_logs.append(step_log)
                    break  # 루프 종료
                
                # action이 있는 경우 도구 실행
                elif action_value is not None:
                    step_log["action"] = action_value # Log the action string itself
                    
                    # Validate action format (tool name and arguments)
                    action_parts = str(action_value).split('::', 1)
                    if len(action_parts) != 2:
                        logger.error(f"Step {step}: Invalid action format: {action_value}. Expected 'server/tool::json_args'.")
                        action_result = {"error": f"Invalid action format: {action_value}."}
                        current_error_messages.append(f"Invalid action format: {action_value}. Expected 'server/tool::json_args'.")
                    else:
                        tool_name_part, args_json_str = action_parts
                        
                        # Validate tool exists
                        if tool_name_part not in tools:
                            error_msg = f"Tool '{tool_name_part}' not found. Available tools: {tools}"
                            logger.error(f"Step {step}: {error_msg}")
                            action_result = {"error": error_msg}
                            current_error_messages.append(error_msg)
                        else:
                            # Parse arguments
                            try:
                                arguments = json.loads(args_json_str)
                                if not isinstance(arguments, dict):
                                    raise ValueError("Arguments must be a JSON object/dict")
                                
                                # Execute the tool
                                try:
                                    logger.info(f"Step {step}: Executing tool '{tool_name_part}' with args: {arguments}") # INFO 유지
                                    server_name, actual_tool_name = tool_name_part.split("/", 1)
                                    action_result = await self.mcp_service.execute_tool(
                                        server_name=server_name,
                                        tool_name=actual_tool_name,
                                        arguments=arguments
                                    )
                                except Exception as tool_exec_e:
                                    logger.error(f"Step {step}: Error executing tool '{tool_name_part}': {tool_exec_e}", exc_info=True)
                                    action_result = {"error": f"Error executing tool '{tool_name_part}': {str(tool_exec_e)}"}
                                    current_error_messages.append(f"Error executing tool '{tool_name_part}': {str(tool_exec_e)}")
                            
                            except json.JSONDecodeError as arg_e:
                                logger.error(f"Step {step}: Invalid JSON arguments for action '{action_value}': {arg_e}")
                                action_result = {"error": f"Invalid JSON arguments provided for action '{tool_name_part}': {args_json_str}"}
                                current_error_messages.append(f"Invalid JSON arguments for action '{action_value}': {arg_e}")
                            except ValueError as arg_val_e:
                                logger.error(f"Step {step}: Arguments must be a dict for action '{action_value}': {arg_val_e}")
                                action_result = {"error": f"Arguments for action '{tool_name_part}' must be a JSON object: {args_json_str}"}
                                current_error_messages.append(f"Arguments must be a dict for action '{action_value}': {arg_val_e}")
                            except Exception as other_e: # Catch unexpected errors during parsing/split
                                logger.error(f"Step {step}: Error processing action '{action_value}': {other_e}", exc_info=True)
                                action_result = {"error": f"Unexpected error processing action: {str(other_e)}"}
                                current_error_messages.append(f"Error processing action '{action_value}': {other_e}")
                    
                    # Format and log the action result
                    formatted_result = format_tool_result(action_result)
                    step_log["action_result"] = formatted_result
                    
                    # Add the step to the trace for the next iteration's context
                    step_data = {
                        "thought": current_thought,
                        "action": action_value, # Store the action string
                        "observation": formatted_result
                    }
                    trace.append(step_data)
                    step_logs.append(step_log)
                
                # 이 경우는 이미 위에서 검증했으므로 도달하지 않을 것임
                else:
                    logger.error(f"Step {step}: Response has neither non-null 'action' nor non-null 'answer'. Raw: {action_answer_json_str}")
                    current_error_messages.append("Invalid LLM response format (neither action nor answer).")

            except json.JSONDecodeError as e:
                logger.error(f"Step {step}: Failed to parse JSON from action/answer LLM response: {e}")
                logger.error(f"Raw JSON string: {action_answer_json_str}")
                result = {"error": f"Invalid JSON response: {e}", "answer": "An error occurred during processing. Please try again."}
                break
            except Exception as e:
                logger.error(f"Step {step}: Unexpected error processing response: {e}", exc_info=True)
                result = {"error": f"Unexpected error: {e}", "answer": "An error occurred during processing. Please try again."}
                break
            
            # Save the step log to the list (NO FILE IO HERE)
            step_logs.append(step_log)
            # try:
            #     # 로그 디렉토리 설정 -> process 메서드 시작 시 정의된 session_dir 사용
            #     # self._format_save_step_log(session_dir, step, f"{step}_step", step_log) # 제거
            # except Exception as e:
            #     logger.error(f"Failed to save step log for step {step} in {session_dir}: {e}", exc_info=True)
            
            # logger.debug(f"Step {step}: Thought generated successfully.") # 제거 (너무 상세)
            # logger.debug(f"Step {step}: Action/Answer generated successfully.") # 제거 (너무 상세)
        
        # If we've exhausted steps without an answer, provide a fallback
        if not result:
            logger.warning(f"No result after {step} steps, providing fallback response")
            result = {
                "answer": "I wasn't able to complete this task successfully. Please try to rephrase your request or break it down into smaller steps."
            }
        
        execution_time = time.time() - start_time
        logger.info(f"ReAct process completed in {execution_time:.2f} seconds after {step} steps for session {session_id}") # INFO 유지 (세션 ID 추가)
        
        result["trace"] = trace
        result["steps"] = step
        result["execution_time"] = execution_time
        result["step_logs"] = step_logs # 최종 결과에 step_logs 포함
        
        return result

    async def _generate_thought_action(self, prompt: str, trace: str) -> Any:
        """
        LLM을 호출하여 현재 스텝에 대한 응답을 생성합니다.
        DEPRECATED: This method is no longer used. Use generate_thought and generate_action_or_answer instead.
        """
        logger.warning("_generate_thought_action is deprecated and should not be called.")
        raise NotImplementedError("_generate_thought_action is deprecated")

    # def _format_save_step_log(self, session_dir: Path, step: int, step_type: str, data: dict) -> None: # 메서드 제거
    #     """
    #     단계별 로그를 저장하는 헬퍼 메서드입니다.
    #     """
    #     try:
    #         save_step_log(session_dir, step, step_type, data)
    #     except Exception as e:
    #         logger.error(f"Failed to save step log: {e}", exc_info=True)

