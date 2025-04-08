import asyncio
import logging
import re
import json
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple, List, Dict, Optional
from fastapi import Depends
from llama_cpp import Llama
from transformers import AutoTokenizer, PreTrainedTokenizer
from huggingface_hub import hf_hub_download
import httpx
import uuid

from app.core.config import Settings, get_settings
from app.services.mcp_service import MCPService
from app.mcp_client.client import MCPError
from app.services.llm_interface import LLMInterface
from app.services.prompt_manager import PromptManager
from app.services.react_processor import ReactProcessor

logger = logging.getLogger(__name__)

# Define the base model identifier for tokenizer loading
BASE_MODEL_ID = "google/gemma-3-1b-it"
ACTION_TOOL_PATTERN = re.compile(r"([\w\-]+)/([\w\-]+)\((.*)\)") # Action 문자열에서 도구/인자 추출용 (유지)

# <<<--- 프롬프트 디렉토리 경로 추가 --- >>>
PROMPT_DIR = Path(__file__).parent.parent / "prompts"

class InferenceService:
    def __init__(self,
                 settings: Settings = Depends(get_settings),
                 mcp_service: MCPService = Depends(),
                 llm_interface: Optional[LLMInterface] = None,
                 prompt_manager: Optional[PromptManager] = None,
                 react_processor: Optional[ReactProcessor] = None
                ):
        logger.info("Initializing InferenceService...")
        self.settings = settings
        self.mcp_service = mcp_service
        
        # 컴포넌트 초기화
        self.llm_interface = llm_interface or LLMInterface(settings=settings)
        self.prompt_manager = prompt_manager or PromptManager()
        self.react_processor = react_processor or ReactProcessor(
            settings=settings, 
            mcp_service=mcp_service,
            llm_interface=self.llm_interface,
            prompt_manager=self.prompt_manager
        )
        
        # 로그 디렉토리 설정
        self.logs_dir = Path('/app/logs') if settings.is_docker else Path('./logs')
        self.react_logs_dir = self.logs_dir / "react_logs"
        self.react_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # LLM 초기화 트리거
        loop = asyncio.get_event_loop()
        if not loop.is_running():
            loop.run_until_complete(self.initialize())
        else:
            asyncio.create_task(self.initialize())
        
        logger.info("InferenceService initialized.")

    async def initialize(self):
        """컴포넌트 초기화를 위한 메서드"""
        logger.info("Initializing LLM...")
        await self.llm_interface.initialize()
        logger.info("LLM initialization completed.")

    # GGUF 다운로드 함수 복원
    async def _download_model(self):
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            model_url_str = str(self.model_url)
            logger.info(f"GGUF 모델 다운로드 시도: {model_url_str} -> {self.model_path}")
            success = False
            
            # 1. huggingface_hub 라이브러리 사용 시도
            try:
                if "huggingface.co" in model_url_str and "/resolve/" in model_url_str:
                    parts = model_url_str.split("/")
                    repo_idx = parts.index("huggingface.co")
                    if len(parts) > repo_idx + 2:
                        owner_model = parts[repo_idx + 1] + "/" + parts[repo_idx + 2]
                        resolve_idx = -1
                        for i, part in enumerate(parts):
                            if part == "resolve":
                                resolve_idx = i
                                break
                        if resolve_idx != -1 and len(parts) > resolve_idx + 2:
                            repo_id = owner_model
                            filename = "/".join(parts[resolve_idx+2:]).split("?")[0]
                            logger.info(f"Extracted from URL: repo_id={repo_id}, filename={filename}")

                            def _sync_download():
                                try:
                                    logger.info(f"Calling hf_hub_download for repo={repo_id}, filename={filename} into dir={self.model_path.parent}")
                                    downloaded_path_str = hf_hub_download(
                                        repo_id=repo_id,
                                        filename=filename,
                                        cache_dir=self.model_path.parent,
                                        local_dir=self.model_path.parent,
                                        local_dir_use_symlinks=False,
                                        force_download=False,
                                        resume_download=True
                                    )
                                    downloaded_path = Path(downloaded_path_str)
                                    logger.info(f"hf_hub_download returned path: {downloaded_path}")

                                    expected_final_path = self.model_path
                                    if expected_final_path.is_file():
                                        logger.info(f"Verified file exists at expected final path: {expected_final_path}")
                                        return True
                                    else: 
                                        logger.warning(f"File NOT found at expected path {expected_final_path} after hf_hub_download returned {downloaded_path}")
                                        if downloaded_path.is_file() and downloaded_path.parent == expected_final_path.parent and downloaded_path != expected_final_path:
                                            logger.info(f"Attempting to rename {downloaded_path} to {expected_final_path}")
                                            try:
                                                downloaded_path.rename(expected_final_path)
                                                if expected_final_path.is_file():
                                                    logger.info(f"Successfully renamed to {expected_final_path}")
                                                    return True
                                                else:
                                                    logger.error(f"Rename appeared successful but {expected_final_path} still not found.")
                                                    return False
                                            except Exception as rename_e:
                                                logger.error(f"Failed to rename {downloaded_path} to {expected_final_path}: {rename_e}")
                                                return False
                                        else:
                                            logger.error(f"Cannot rename: Source file does not exist, is in wrong directory, or is the same as target.")
                                            return False

                                except Exception as e:
                                    logger.error(f"hf_hub_download execution failed: {e}", exc_info=True)
                                    return False
                            
                            loop = asyncio.get_event_loop()
                            success = await loop.run_in_executor(None, _sync_download)
                            if success:
                                logger.info(f"huggingface_hub로 GGUF 모델 다운로드/확인 성공: {self.model_path}")
                            else:
                                logger.warning("hf_hub_download failed or did not result in the expected file. Falling back to httpx.")
            except Exception as hf_e:
                 logger.warning(f"huggingface_hub 방식 처리 중 예외 발생 (URL 파싱 등): {hf_e}")
                 success = False # hf_hub 방식 실패로 간주하고 httpx 실행

            # 2. httpx 직접 다운로드 (hf_hub 실패 시 또는 URL 형식이 다를 때)
            if not success:
                logger.info("httpx를 사용한 직접 다운로드 시도...")
                headers = {}
                hf_token = os.environ.get("HUGGING_FACE_TOKEN")
                if hf_token: headers["Authorization"] = f"Bearer {hf_token}"
                
                async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
                    async with client.stream("GET", model_url_str, headers=headers) as response:
                        response.raise_for_status()
                        total_size = int(response.headers.get("content-length", 0))
                        chunk_size = 8192
                        downloaded_size = 0
                        logger.info(f"GGUF 모델 다운로드 중 ({total_size / (1024*1024):.2f} MB) -> {self.model_path}...")
                        with open(self.model_path, "wb") as f:
                            async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                                f.write(chunk)
                                downloaded_size += len(chunk)
                                if downloaded_size % (chunk_size * 512) == 0 or downloaded_size == total_size:
                                    progress = (downloaded_size / total_size) * 100 if total_size else 0
                                    logger.debug(f"다운로드 진행: {downloaded_size / (1024*1024):.2f} MB ({progress:.1f}%)")
            logger.info("httpx로 GGUF 모델 다운로드 완료")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP 오류: {e.response.status_code} - {e.response.text[:200]}...")
            if self.model_path.exists(): self.model_path.unlink()
            raise
        except Exception as e:
            logger.error(f"GGUF 모델 다운로드 실패: {e}", exc_info=True)
            if self.model_path.exists(): self.model_path.unlink()
            raise

    # 언어 감지 함수는 그대로 사용
    def _detect_language(self, text: str) -> str:
        if not text or len(text.strip()) < 2:
            return "en"
        korean_char_count = len([c for c in text if ord('가') <= ord(c) <= ord('힣') or 
                                 ord('ㄱ') <= ord(c) <= ord('ㅎ') or
                                 ord('ㅏ') <= ord(c) <= ord('ㅣ')])
        if korean_char_count > 0 and korean_char_count / len(text.strip()) > 0.1:
            return "ko"
        return "en"

    # <<<--- 프롬프트 로드 헬퍼 함수 추가 --- >>>
    def _load_prompt_template(self, template_name: str) -> str:
        """Loads a prompt template file from the prompts directory."""
        if template_name in self.prompt_templates:
            return self.prompt_templates[template_name]
        
        file_path = PROMPT_DIR / f"{template_name}.txt"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.prompt_templates[template_name] = content
                return content
        except FileNotFoundError:
            logger.error(f"Prompt template file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompt template {file_path}: {e}", exc_info=True)
            raise

    def _load_all_prompt_templates(self):
        """Loads all required prompt templates into the cache."""
        # <<<--- 수정: 새로운 통합 파일 이름 사용 --- >>>
        required_templates = [
            "react_system_core", 
            "react_system_examples",
            "react_system_final_reminder"
        ]
        for name in required_templates:
            self._load_prompt_template(name)
    
    def _build_react_system_prompt(self, tool_details: Dict[str, Dict], iteration: int = 1) -> str:
        """Builds the simplified ReAct system prompt using consolidated templates."""
        
        # <<<--- 수정: 새로운 통합 템플릿 로드 --- >>>
        core_prompt = self._load_prompt_template("react_system_core")
        examples = self._load_prompt_template("react_system_examples")
        final_reminder = self._load_prompt_template("react_system_final_reminder")
        
        # 사용 가능한 도구 포맷팅 (기존 로직 유지)
        tools_str = "No tools available." if not tool_details else ""
        if tool_details:
            formatted_tools = []
            for server, tools_info in tool_details.items():
                # 수정: tools_info가 비어있지 않은 dict인지 확인
                if isinstance(tools_info, dict) and tools_info: 
                    for tool_name, tool_spec in tools_info.items():
                        description = tool_spec.get('description', 'No description available')
                        formatted_tools.append(f"- `{server}/{tool_name}`: {description}")
                # 수정: 경고 메시지 개선
                elif not isinstance(tools_info, dict):
                     logger.warning(f"_build_react_system_prompt: Expected dict for tools_info of server '{server}', but got {type(tools_info)}. Skipping server.")
                else: # 비어있는 dict인 경우
                    logger.warning(f"_build_react_system_prompt: Tools dictionary for server '{server}' is empty. Skipping server.")
            tools_str = "\n".join(formatted_tools)

        # <<<--- 수정: 통합된 템플릿 조합 --- >>>
        # 핵심 프롬프트에 도구 목록 삽입
        core_prompt_formatted = core_prompt.format(tools=tools_str.strip())
        
        # 프롬프트 조합 (핵심 -> 예시 -> 최종 리마인더)
        prompt_parts = [
            core_prompt_formatted,
            examples,
            final_reminder
        ]
        
        formatted_prompt = "\n\n".join(prompt_parts)
        return formatted_prompt

    # *** _call_llm 함수 복구: create_completion 사용 ***
    async def _call_llm(self, prompt: str, stop_tokens: Optional[List[str]] = None) -> str:
        """LLM 호출을 수행하고 응답 텍스트를 반환합니다. (create_completion 사용)"""
        if not self.model:
            logger.error("LLM call failed: Model not loaded.")
            try: 
                await self.initialize()
            except Exception as init_e:
                 logger.error(f"Failed to re-initialize model during LLM call: {init_e}", exc_info=True)
                 raise RuntimeError("Model is not loaded and could not be reloaded.") from init_e
            if not self.model:
                 raise RuntimeError("Model is not loaded and could not be reloaded after re-initialization attempt.")

        start_time = time.time()
        logger.info(f"Calling LLM... (Prompt length: {len(prompt)})")

        try:
            # <<<--- 복구: self.model.create_completion 사용 --->
            response = self.model.create_completion(
                prompt=prompt,
                max_tokens=self.max_new_tokens,
                temperature=self.default_temperature,
                top_p=self.default_top_p,
                stop=stop_tokens or ["User:"], # <<<--- stop 토큰 목록 수정: ``` 제외
                stream=False # Non-streaming completion
            )
            
            # <<<--- 복구: create_completion 응답 구조 사용 --->
            generated_text = response['choices'][0]['text'].strip()
            
            end_time = time.time()
            logger.info(f"LLM call completed in {end_time - start_time:.2f} seconds. Generated {len(generated_text)} characters.")
            return generated_text
            
        except Exception as e:
            logger.error(f"Error during LLM call (create_completion): {e}", exc_info=True)
            # <<<--- 복구: 오류 메시지 원복 --->
            raise RuntimeError(f"Model completion failed: {str(e)}") from e

    # *** generate 함수 단순화: 무조건 process_react_pattern 호출 ***
    async def generate(self, text: str) -> dict:
        """사용자 요청을 ReactProcessor에 위임하여 처리"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        logger.info(f"Received generation request for Session ID: {session_id}, User prompt: '{text[:50]}...'")

        # LLM 초기화 확인
        try:
            if not self.llm_interface.model or not self.llm_interface.tokenizer:
                logger.warning(f"LLM not initialized for Session ID: {session_id}. Attempting initialization.")
                await self.llm_interface.initialize()
                if not self.llm_interface.model or not self.llm_interface.tokenizer:
                    logger.error(f"LLM initialization failed for Session ID: {session_id}")
                    return {
                        "response": "Model initialization failed.",
                        "error": "Initialization failed",
                        "log_session_id": session_id,
                        "log_path": None,
                        "thoughts_and_actions": None,
                        "full_response": None
                    }
        except Exception as init_e:
            logger.error(f"Initialization check failed for Session ID: {session_id}: {init_e}", exc_info=True)
            return {
                "response": f"Initialization error: {init_e}",
                "error": str(init_e),
                "log_session_id": session_id,
                "log_path": None,
                "thoughts_and_actions": None,
                "full_response": None
            }

        # ReactProcessor에 처리 위임
        try:
            final_answer, thoughts_actions, conversation_log, error_message = \
                await self.react_processor.process_react_pattern(text, session_id)

            log_path_obj = self.react_logs_dir / session_id / "meta.json"
            log_path = str(log_path_obj) if log_path_obj.exists() else None

            return {
                "response": final_answer,
                "thoughts_and_actions": thoughts_actions,
                "full_response": conversation_log,
                "error": error_message,
                "log_session_id": session_id,
                "log_path": log_path
            }
        except Exception as react_e:
            logger.error(f"Error during React processing for Session ID: {session_id}: {react_e}", exc_info=True)
            return {
                "response": f"Error during processing: {react_e}",
                "thoughts_and_actions": None,
                "full_response": None,
                "error": str(react_e),
                "log_session_id": session_id,
                "log_path": None
            }

    # --- _clean_response 함수 수정: 다단계 추출 로직 ---
    def _clean_response(self, text: str) -> Tuple[Optional[str], Optional[Exception]]:
        """
        Attempts to extract a JSON object from the text using multiple strategies.
        Returns a tuple: (json_string, None) on success,
                        (None, exception) on failure.
        """
        logger.debug(f"_clean_response: Input text (length: {len(text)}). First 200 chars: {text[:200]}...")
        potential_json_str: Optional[str] = None
        last_exception: Optional[Exception] = None # Store the last encountered exception

        # Strategy 1: Look for ```json ... ``` block
        logger.debug("_clean_response: Attempting Strategy 1: ```json block...")
        match_md = re.search(r"```json\s*({.*?})\s*```", text, re.DOTALL | re.IGNORECASE)
        if match_md:
            potential_json_str = match_md.group(1).strip()
            logger.debug(f"_clean_response S1: Found potential JSON in ```json block: {potential_json_str[:200]}...")
            try:
                json.loads(potential_json_str)
                logger.info("_clean_response S1: Successfully parsed JSON from ```json block.")
                return potential_json_str, None # Success
            except json.JSONDecodeError as e:
                last_exception = e # Store the exception
                logger.warning(f"_clean_response S1: Failed to parse JSON from ```json block: {e}. String (repr): {repr(potential_json_str)}")
            except Exception as e_md:
                last_exception = e_md # Store the exception
                logger.error(f"_clean_response S1: Unexpected error processing ```json block: {e_md}", exc_info=True)
            # Fall through on error

        # Strategy 2: Find outermost curly braces {}
        logger.debug("_clean_response: Attempting Strategy 2: Outermost braces...")
        potential_json_str = None # Reset for this strategy
        try:
            first_brace_index = text.find('{')
            last_brace_index = text.rfind('}')

            if first_brace_index != -1 and last_brace_index != -1 and last_brace_index > first_brace_index:
                potential_json_str = text[first_brace_index : last_brace_index + 1]
                logger.debug(f"_clean_response S2: Found potential JSON between braces: {potential_json_str[:200]}...")
                json.loads(potential_json_str)
                logger.info("_clean_response S2: Successfully parsed JSON from outermost braces.")
                return potential_json_str, None # Success
            else:
                logger.debug("_clean_response S2: Could not find valid outermost braces.")
                # Fall through

        except json.JSONDecodeError as e:
            last_exception = e # Store the exception
            logger.warning(f"_clean_response S2: Failed to parse JSON from braces: {e}. String (repr): {repr(potential_json_str)}")
        except Exception as e_brace:
            last_exception = e_brace # Store the exception
            logger.error(f"_clean_response S2: Unexpected error processing braces: {e_brace}", exc_info=True)
        # Fall through on error

        # Strategy 3: Use json.JSONDecoder().raw_decode()
        logger.debug("_clean_response: Attempting Strategy 3: raw_decode...")
        text_to_decode = None # Initialize
        try:
            first_json_char_index = -1
            for i, char in enumerate(text):
                if char == '{' or char == '[':
                    first_json_char_index = i
                    break

            if first_json_char_index != -1:
                text_to_decode = text[first_json_char_index:].strip()
                logger.debug(f"_clean_response S3: Text to attempt raw_decode on: {text_to_decode[:200]}...")
                decoder = json.JSONDecoder()
                decoded_obj, end_index = decoder.raw_decode(text_to_decode)
                result_json_str = text_to_decode[:end_index]
                # Quick validation by reloading
                json.loads(result_json_str)
                logger.info(f"_clean_response S3: Successfully parsed JSON using raw_decode. Detected end index: {end_index}")
                return result_json_str, None # Success
            else:
                logger.debug("_clean_response S3: No starting JSON character ({ or [) found for raw_decode.")
                # Fall through (no explicit exception here, but no success either)

        except json.JSONDecodeError as e:
            last_exception = e # Store the exception
            logger.warning(f"_clean_response S3: raw_decode failed: {e}. Text was (repr): {repr(text_to_decode)}")
        except Exception as e_raw:
            last_exception = e_raw # Store the exception
            logger.error(f"_clean_response S3: Unexpected error during raw_decode: {e_raw}", exc_info=True)
        # Fall through on error


        # If all strategies fail
        logger.error(f"_clean_response: Failed to extract AND parse valid JSON using all strategies. Last exception: {last_exception}")
        return None, last_exception # Return None and the last exception encountered

    async def _translate_text(self, text: str, target_lang: str, source_lang: str = "en") -> str:
        """Translates text using a simple LLM call."""
        if not text or not self.model:
            return text # 번역할 텍스트가 없거나 모델이 없으면 원본 반환

        # 언어 코드에 따라 프롬프트 설정 (간단하게)
        lang_map = {"ko": "Korean", "en": "English"}
        target_lang_full = lang_map.get(target_lang, target_lang)
        source_lang_full = lang_map.get(source_lang, source_lang)

        prompt = f"Translate the following {source_lang_full} text to {target_lang_full}. Output ONLY the translated text:\n\n{text}"
        logger.info(f"Attempting translation from {source_lang} to {target_lang} for text: {text[:50]}...")

        try:
            # _call_llm과 유사하게 completion 호출 (stop 토큰 없이)
            response = self.model.create_completion(
                prompt=prompt,
                max_tokens=len(text) * 3 + 50, # 원본 길이 기반으로 토큰 수 추정
                temperature=0.7, # 번역에는 약간 낮은 temperature
                top_p=1.0,
                stop=None, # 번역 결과가 잘리지 않도록 stop 없음
                stream=False
            )
            translated_text = response['choices'][0]['text'].strip()
            
            # <<<--- 추가: 번역 결과 로깅 및 비어있는지 확인 --->
            logger.debug(f"Raw translation result from LLM (repr): {repr(translated_text)}") # 정확한 원본 로깅
            if not translated_text:
                logger.warning("Translation result was empty or whitespace after stripping. Returning original text.")
                return text # 빈 문자열 대신 원본 영어 반환
            # <<<--- 로깅 및 확인 로직 끝 --->
            
            logger.info(f"Translation successful: {translated_text[:50]}...")
            return translated_text
        except Exception as e:
            logger.error(f"Translation failed: {e}", exc_info=True)
            return text # 오류 시 원본 텍스트 반환

    def _format_tool_result(self, result: Any) -> str: # 클래스 멤버 함수로 들여쓰기 수정
        """MCP 도구 결과를 문자열로 포맷팅합니다."""
        if result is None:
            return "No result returned from tool."
        
        try:
            # 구조화된 콘텐츠 처리
            if isinstance(result, dict) and "content" in result:
                content = result["content"]
                
                # 리스트 형태의 콘텐츠는 각 부분을 조합
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict):
                            # 텍스트 부분 추출
                            if "text" in item:
                                parts.append(str(item["text"]))
                            # 다른 유형의 콘텐츠 처리 (필요시 확장)
                        else:
                            parts.append(str(item))
                    
                    # 모든 부분을 합쳐서 반환
                    return " ".join(parts).strip()
                
                # 단일 콘텐츠 처리 
                if isinstance(content, dict) and "text" in content:
                    return str(content["text"]).strip()
                
                # 직접 문자열 또는 다른 형태의 콘텐츠 처리
                return str(content).strip()
            
            # 문자열 처리
            if isinstance(result, str):
                return result.strip()
            
            # 기타 타입 처리 (숫자, 불리언 등)
            if isinstance(result, (int, float, bool)):
                return str(result)
            
            # 일반적인 객체 문자열 변환
            return str(result).strip()
        
        except Exception as e:
            logger.error(f"Error formatting tool result: {e}", exc_info=True)
            return f"Error formatting tool result: {str(e)}"

    # --- process_react_pattern 함수 수정: JSON 파싱 기반 로직 --- 
    async def process_react_pattern(self, initial_prompt: str, session_id: str = None) -> Tuple[str, List[Dict], str, Optional[str]]:
        """
        Processes the user input using the ReAct pattern, handling language translation.
        
        Args:
            initial_prompt: The initial user prompt.
            session_id: Optional session ID for logging.
            
        Returns:
            Tuple[str, List[Dict], str, Optional[str]]: Final answer, trace log, full conversation, error message.
        """
        if not session_id:
            session_id = str(uuid.uuid4().int)
        logger.info(f"process_react_pattern called for session {session_id}")

        original_language = self._detect_language(initial_prompt)
        logger.info(f"Detected language: {original_language}")
        
        prompt_to_process = initial_prompt
        
        # Translate to English if the original language is not English
        if original_language != 'en':
            logger.debug(f"Translating prompt from {original_language} to en...")
            try:
                prompt_to_process = await self.llm_interface.translate_text(initial_prompt, target_lang='en', source_lang=original_language)
                logger.debug(f"Translated prompt to English: {prompt_to_process[:100]}...")
            except Exception as translate_e:
                logger.error(f"Translation to English failed: {translate_e}", exc_info=True)
                # Fallback: Process with the original prompt if translation fails
                prompt_to_process = initial_prompt
                logger.warning("Proceeding with original language prompt due to translation failure.")
        
        # Call ReactProcessor with the (potentially translated) English prompt
        try:
            result_dict = await self.react_processor.process(prompt_to_process) # Pass the processed prompt
        except Exception as react_e:
            logger.error(f"Error during React processing: {react_e}", exc_info=True)
            return "Sorry, an internal error occurred during processing.", [], initial_prompt, str(react_e)

        final_answer_en = result_dict.get("answer", "I couldn't determine a final answer.")
        trace_log = result_dict.get("trace", [])
        step_logs = result_dict.get("step_logs", []) # Assuming process returns step logs
        error_messages = result_dict.get("error_messages", []) # Assuming process returns errors

        # Translate the final answer back to the original language if needed
        final_answer_translated = final_answer_en
        if original_language != 'en':
            logger.debug(f"Translating final answer from en to {original_language}...")
            try:
                final_answer_translated = await self.llm_interface.translate_text(final_answer_en, target_lang=original_language, source_lang='en')
                logger.debug(f"Translated final answer to {original_language}: {final_answer_translated[:100]}...")
            except Exception as translate_back_e:
                logger.error(f"Translation back to {original_language} failed: {translate_back_e}", exc_info=True)
                # Fallback: Return the English answer if back-translation fails
                final_answer_translated = final_answer_en 
                logger.warning(f"Returning English answer due to back-translation failure.")

        # Construct full response log (consider adding translation steps)
        full_response_parts = [f"User ({original_language}): {initial_prompt}"]
        if original_language != 'en' and prompt_to_process != initial_prompt:
            full_response_parts.append(f"System (Translated to en): {prompt_to_process}")
            
        # Include ReAct trace in the full log
        for step_data in trace_log:
            if step_data.get("thought"):
                full_response_parts.append(f"Thought: {step_data['thought']}")
            if step_data.get("action"):
                full_response_parts.append(f"Action: {step_data['action']}") # Assuming action is string
            if step_data.get("observation"):
                full_response_parts.append(f"Observation: {step_data['observation']}")
                
        full_response_parts.append(f"Final Answer (en): {final_answer_en}")
        if original_language != 'en' and final_answer_translated != final_answer_en:
            full_response_parts.append(f"Final Answer ({original_language}): {final_answer_translated}")
        elif original_language == 'en':
            full_response_parts.append(f"Final Answer (en): {final_answer_translated}") # For consistency if already english
            
        full_response_log = "\n\n".join(full_response_parts)

        error_msg_str = "; ".join(error_messages) if error_messages else None
        
        # Save logs (consider adding translated prompts/answers to logs)
        session_dir = self.react_logs_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        self.react_processor._save_meta(session_dir, initial_prompt, original_language, result_dict, session_id)
        
        return final_answer_translated, trace_log, full_response_log, error_msg_str

    def _save_step_log(self, log_dir: Path, step: int, step_type: str, data: dict):
        """ReAct 단계별 로그를 JSON 파일로 저장합니다."""
        try:
            filename = f"{step:02d}_{step_type}.json"
            log_file_path = log_dir / filename
            with open(log_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save log file {log_file_path}: {e}")

def get_inference_service(
    settings: Settings = Depends(get_settings),
    mcp_service: MCPService = Depends()
) -> InferenceService:
    """
    InferenceService 인스턴스를 반환하는 의존성 함수.
    컴포넌트들은 app_state를 통해 공유되어 싱글톤으로 관리됩니다.
    """
    from app.main import app_state
    
    # 컴포넌트 객체들을 app_state에서 관리
    if "llm_interface" not in app_state:
        app_state["llm_interface"] = LLMInterface(settings=settings)
    
    if "prompt_manager" not in app_state:
        app_state["prompt_manager"] = PromptManager()
    
    if "react_processor" not in app_state:
        app_state["react_processor"] = ReactProcessor(
            settings=settings,
            mcp_service=mcp_service,
            llm_interface=app_state["llm_interface"],
            prompt_manager=app_state["prompt_manager"]
        )
    
    if "inference_service" not in app_state:
        logger.info("Creating InferenceService singleton instance.")
        app_state["inference_service"] = InferenceService(
            settings=settings,
            mcp_service=mcp_service,
            llm_interface=app_state["llm_interface"],
            prompt_manager=app_state["prompt_manager"],
            react_processor=app_state["react_processor"]
        )
    
    inference_service_instance = app_state["inference_service"]
    
    # LLM 초기화 상태 확인
    if not inference_service_instance.llm_interface.model or not inference_service_instance.llm_interface.tokenizer:
        logger.warning("get_inference_service: LLMInterface seems uninitialized.")
    
    return inference_service_instance 