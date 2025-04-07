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

from app.core.config import Settings, get_settings
from app.services.mcp_service import MCPService
from app.mcp_client.client import MCPError

logger = logging.getLogger(__name__)

# Define the base model identifier for tokenizer loading
BASE_MODEL_ID = "google/gemma-3-1b-it"
ACTION_TOOL_PATTERN = re.compile(r"([\w\-]+)/([\w\-]+)\((.*)\)") # Action 문자열에서 도구/인자 추출용 (유지)

# <<<--- 프롬프트 디렉토리 경로 추가 --- >>>
PROMPT_DIR = Path(__file__).parent.parent / "prompts"

class InferenceService:
    def __init__(self, settings: Settings = Depends(get_settings), mcp_service: MCPService = Depends()):
        self.settings = settings
        self.mcp_service = mcp_service
        
        # --- 모델 정보 재확인 및 수정 (abliterated Q8_0) ---
        self.new_model_repo = "mradermacher/gemma3-4b-it-abliterated-GGUF"
        self.new_model_filename = "gemma3-4b-it-abliterated.Q8_0.gguf" # Q8_0 버전 재확인
        self.new_tokenizer_base_id = "google/gemma-3-4b-it" # 베이스 토크나이저 유지
        
        # GGUF 관련 속성 (Q8_0 파일명 사용 확인)
        self.model_path = Path('/app/models') / self.new_model_filename if settings.is_docker else Path('./models') / self.new_model_filename
        self.model_url = f"https://huggingface.co/{self.new_model_repo}/resolve/main/{self.new_model_filename}"
        
        self.tokenizer_path = Path('/app/tokenizer') if settings.is_docker else Path('./tokenizer')
        self.base_model_id_for_tokenizer = self.new_tokenizer_base_id 
        # --- 모델 정보 수정 완료 ---

        # 모델 및 토크나이저 속성 초기화
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model: Optional[Llama] = None

        # 추론 관련 설정
        self.max_new_tokens = 8192
        self.default_temperature = 1.0
        self.default_top_p = 1.0
        self.max_tokens = 131072
        
        # 로그 저장 디렉토리 설정
        self.logs_dir = Path('/app/logs') if settings.is_docker else Path('./logs')
        self.react_logs_dir = self.logs_dir / "react_logs"
        self.react_logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_iterations = 10
        self.lock = asyncio.Lock() # DEBUG -> 주석 해제
        # self._download_task = None # DEBUG: 주석 처리 (다운로드 로직은 initialize 내에서 처리)
        
        # 즉시 초기화 실행 (로그 레벨 INFO로 변경)
        loop = asyncio.get_event_loop()
        if not loop.is_running():
            loop.run_until_complete(self.initialize())
        else:
            asyncio.create_task(self.initialize())
            # logger.info("Model initialization scheduled as async task") # INFO: 유지

        self.prompt_templates: Dict[str, str] = {}
        self._load_all_prompt_templates() 

    async def initialize(self):
        # logger.debug(...) -> 주석 처리 또는 제거
        async with self.lock: # lock은 유지
            if self.model and self.tokenizer:
                logger.info("Model and tokenizer already initialized.") # INFO: 유지
                return

            if not self.model_path.is_file():
                logger.warning(f"GGUF 모델 파일을 찾을 수 없습니다: {self.model_path}. 다운로드를 시도합니다.") # WARNING: 유지
                # ... (다운로드 로직, 내부 logger.info/warning/error 유지)
            else:
                logger.info(f"GGUF 모델 파일 확인됨: {self.model_path}") # INFO: 유지

            try:
                # 토크나이저 로드
                try:
                    logger.info(f"Loading tokenizer from HuggingFace Hub: {self.base_model_id_for_tokenizer}") # INFO: 유지
                    self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id_for_tokenizer)
                    logger.info("Tokenizer loaded successfully from Hub.") # INFO: 유지
                except Exception as hub_e:
                    logger.warning(f"Hub에서 토크나이저 로드 실패: {hub_e}. 로컬 경로 시도: {self.tokenizer_path}") # WARNING: 유지
                    # ... (로컬 로드 로직, 내부 logger.info/error 유지)

                # GGUF 모델 로드
                logger.info(f"Loading GGUF model: {self.model_path}") # INFO: 유지
                start_time = time.time()
                try:
                    logger.info("Attempting to load model with GPU acceleration...") # INFO: 유지
                    self.model = Llama(
                        model_path=str(self.model_path),
                        n_ctx=8192,
                        n_gpu_layers=-1,  # 가능한 모든 레이어를 GPU로
                        verbose=False
                    )
                    logger.info(f"GGUF model loaded with GPU acceleration. (Took {time.time() - start_time:.2f} seconds)") # INFO: 유지
                except Exception as gpu_e:
                    logger.warning(f"GPU 가속으로 모델 로드 실패: {gpu_e}. CPU로 전환합니다.") # WARNING: 유지
                    self.model = Llama(
                        model_path=str(self.model_path),
                        n_ctx=8192,
                        n_gpu_layers=0, # CPU 폴백
                        verbose=False
                    )
                    logger.info(f"GGUF model loaded with CPU. (Took {time.time() - start_time:.2f} seconds)") # INFO: 유지

                logger.info("Inference service (GGUF) initialized successfully.") # INFO: 유지

            except Exception as e:
                logger.error(f"GGUF 초기화 실패: {e}", exc_info=True) # ERROR: 유지
                self.model = None
                self.tokenizer = None # 토크나이저도 초기화
                raise RuntimeError(f"GGUF 추론 서비스 초기화 실패: {e}") from e

    # GGUF 다운로드 함수 복원
    async def _download_model(self):
        # 내부 logger.info/warning/error 유지
        # logger.debug(...) -> 제거 또는 주석 처리
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
        required_templates = [
            "react_base_simple", "react_json_structure_simple", "react_rules_simple", 
            "react_example_simple_tool", "react_example_simple_direct",
            "react_final_reminder" # 최종 리마인더는 유지
        ]
        for name in required_templates:
            self._load_prompt_template(name)
    
    # <<<--- 수정: _build_react_system_prompt 함수 단순화 --- >>>
    def _build_react_system_prompt(self, tool_details: Dict[str, Dict], iteration: int = 1) -> str:
        """Builds the simplified ReAct system prompt."""
        
        # 단순화된 프롬프트 조각 로드
        base = self._load_prompt_template("react_base_simple")
        json_structure = self._load_prompt_template("react_json_structure_simple")
        rules = self._load_prompt_template("react_rules_simple")
        example_tool = self._load_prompt_template("react_example_simple_tool")
        example_direct = self._load_prompt_template("react_example_simple_direct")
        final_reminder = self._load_prompt_template("react_final_reminder")
        
        # 사용 가능한 도구 포맷팅
        tools_str = "No tools available." if not tool_details else ""
        if tool_details: # 도구가 있을 경우에만 포맷팅 진행
            formatted_tools = []
            for server, tools_info in tool_details.items(): # tools_info는 이제 딕셔너리
                # --- 수정: tools_info가 딕셔너리라고 가정하고 각 도구 정보에 접근 ---
                if isinstance(tools_info, dict): # 형식 확인
                    for tool_name, tool_spec in tools_info.items():
                        description = tool_spec.get('description', 'No description available') # 설명 추출 (없으면 기본값)
                        formatted_tools.append(f"- `{server}/{tool_name}`: {description}") # 설명 포함
            else:
                     logger.warning(f"_build_react_system_prompt: Expected dict for tools_info of server '{server}', but got {type(tools_info)}. Skipping.")
            tools_str = "\\n".join(formatted_tools) # 줄바꿈으로 합침

        # 프롬프트 조합 (단순하게)
        prompt_parts = [
            base.format(tools=tools_str.strip()),
            json_structure,
            rules,
            example_tool,
            example_direct,
            final_reminder
        ]
        
        formatted_prompt = "\n\n".join(prompt_parts) # 파트 간 간격 추가
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
        """Handles user requests using the ReAct pattern ONLY."""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        logger.info(f"Starting ReAct generation for Session ID: {session_id}, User prompt: '{text[:50]}...'" )

        # 모델 초기화 확인
        try:
            if self.model is None or self.tokenizer is None:
                await self.initialize()
                if self.model is None or self.tokenizer is None:
                    logger.error(f"Model/Tokenizer still not initialized for Session ID: {session_id}")
                    # 오류 시에도 dict 형태로 반환
                    return {"response": "Model initialization failed.", "error": "Initialization failed", "log_session_id": session_id, "log_path": None, "thoughts_and_actions": None, "full_response": None}
        except Exception as init_e:
             logger.error(f"Initialization check failed for Session ID: {session_id}: {init_e}", exc_info=True)
             return {"response": f"Initialization error: {init_e}", "error": str(init_e), "log_session_id": session_id, "log_path": None, "thoughts_and_actions": None, "full_response": None}

        # 무조건 ReAct 처리
        try:
            final_answer, thoughts_actions, conversation_log, error_message = await self.process_react_pattern(text, session_id)
            log_path = str(self.react_logs_dir / session_id) if (self.react_logs_dir / session_id).exists() else None
            return {
                "response": final_answer,
                "thoughts_and_actions": thoughts_actions,
                "full_response": conversation_log,
                "error": error_message,
                "log_session_id": session_id,
                "log_path": log_path
            }
        except Exception as react_e:
            logger.error(f"Error during ReAct processing for Session ID: {session_id}: {react_e}", exc_info=True)
            # 오류 시에도 dict 형태로 반환
            return {
                "response": f"Error during processing: {react_e}", 
                "thoughts_and_actions": None,
                "full_response": None,
                "error": str(react_e),
                "log_session_id": session_id,
                "log_path": None
            }

    # --- _clean_response 함수 수정: 예외 처리 및 반환 로직 명확화 --- 
    def _clean_response(self, text: str) -> Optional[str]: 
        """Attempts to extract the *first* JSON object found, returns it explicitly."""
        logger.debug(f"_clean_response: Input text (length: {len(text)}). First 200 chars: {text[:200]}...")
        result_json: Optional[str] = None # 최종 반환할 변수

        # 1. ```json ... ``` 블록 검색
        logger.debug("_clean_response: Searching for ```json block...")
        match_md = re.search(r"```json\s*({.*?})\s*```", text, re.DOTALL | re.IGNORECASE)
        if match_md:
            logger.debug("_clean_response: Found ```json block.")
            json_content_raw = None # 변수 초기화
            json_content_stripped = None
            try:
                json_content_raw = match_md.group(1)
                json_content_stripped = json_content_raw.strip()
                logger.debug(f"_clean_response: Stripped content (repr): {repr(json_content_stripped)}") # repr() 추가
                logger.debug("_clean_response: Attempting initial json.loads()...")
                # --- 상세 로깅 추가 --- 
                logger.debug(f"_clean_response: Trying to parse (initial): {json_content_stripped[:200]}...")
                parsed_data = json.loads(json_content_stripped) 
                # --- 상세 로깅 추가 끝 ---
                result_json = json_content_stripped # <<<--- 성공 시 변수에 할당
            
            except json.JSONDecodeError as e:
                # --- 상세 로깅 추가 --- 
                logger.warning(f"_clean_response: Initial json.loads() failed: {e}. String was (repr): {repr(json_content_stripped)}")
                # --- 상세 로깅 추가 끝 ---
                # "Extra data" 오류 처리 시도 (json_content_stripped가 None이 아닐 때만)
                if json_content_stripped is not None and "Extra data" in str(e) and json_content_stripped.endswith('}'):
                    logger.warning("_clean_response: Trying to remove trailing brace due to 'Extra data' error.")
                    try:
                        cleaned_part = json_content_stripped[:-1].strip()
                        logger.debug(f"_clean_response: Retrying json.loads() after removing brace... Content (repr): {repr(cleaned_part)}") # repr() 추가
                        # --- 상세 로깅 추가 --- 
                        logger.debug(f"_clean_response: Trying to parse (after brace removal): {cleaned_part[:200]}...")
                        parsed_data_retry = json.loads(cleaned_part)
                        logger.info(f"_clean_response: Successfully parsed after removing trailing brace. Parsed type: {type(parsed_data_retry)}")
                        # --- 상세 로깅 추가 끝 ---
                        result_json = cleaned_part # <<<--- 성공 시 변수에 할당
                    except json.JSONDecodeError as e2:
                         # --- 상세 로깅 추가 --- 
                         logger.warning(f"_clean_response: Still failed after removing trailing brace: {e2}. String was (repr): {repr(cleaned_part)}")
                         # --- 상세 로깅 추가 끝 ---
                    except Exception as inner_e: # 추가적인 내부 예외 처리
                         logger.error(f"_clean_response: Error during trailing brace removal/parsing: {inner_e}", exc_info=True)
                else:
                    logger.warning("_clean_response: JSONDecodeError was not 'Extra data' or could not apply fix.")
            except Exception as group_e: # group(1) 추출 등 다른 예외
                 logger.error(f"_clean_response: Error processing matched ```json block (e.g., group extraction): {group_e}", exc_info=True)
            
            # ```json 블록 처리 시도 후 결과 반환 (성공/실패 무관)
            # result_json이 설정되었으면 성공, 아니면 실패(None) 반환됨
            if result_json is not None:
                logger.debug(f"_clean_response: Returning result from ```json block processing (length: {len(result_json)})")
            else:
                logger.error("_clean_response: Failed to extract/parse from ```json block. Returning None.")
            return result_json # <<<--- 여기서 첫 번째 블록 처리 결과 반환

        # --- ```json 블록이 없을 경우의 폴백 로직 --- 
        # result_json이 None일 때만 실행됨 (위에서 return되지 않은 경우)
        logger.debug("_clean_response: No ```json block found or processed. Falling back to other patterns...")
        # 1. ``` ... ``` 블록 (Non-greedy)
        match_plain_md = re.search(r"```\s*({.*?})\s*```", text, re.DOTALL)
        if match_plain_md:
            try: 
                json_part = match_plain_md.group(1).strip(); 
                json.loads(json_part); 
                result_json = json_part 
                logger.debug("Found and parsed JSON in plain ``` block.")
            except Exception as e: 
                logger.warning(f"Failed to parse plain ``` block content: {e}")

        # 2. 가장 바깥쪽 중괄호 {} (폴백 1 실패 시)
        if result_json is None:
            match_outer_braces = re.search(r"^\s*({.*?})\s*$", text, re.DOTALL) 
            if match_outer_braces:
                try: 
                    json_part = match_outer_braces.group(1).strip(); 
                    json.loads(json_part); 
                    result_json = json_part 
                    logger.debug("Found and parsed JSON with outer braces.")
                except Exception as e: 
                    logger.warning(f"Failed to parse outer braces content: {e}")

        # 3. 전체 텍스트 (폴백 2 실패 시)
        if result_json is None:
            stripped_text = text.strip()
            if stripped_text.startswith("{") and stripped_text.endswith("}"):
                try: 
                    json.loads(stripped_text); 
                    result_json = stripped_text 
                    logger.debug("Parsed stripped full text as JSON.")
                except Exception as e:
                    logger.warning(f"Failed to parse stripped full text: {e}")
        
        # --- 최종 반환 --- 
        if result_json is not None:
             pass # 성공 시 별도 로그 불필요
        else:
             logger.error(f"_clean_response: Could not extract AND parse valid JSON object using any method. Returning None.") # <<<--- 에러 로그는 유지
        return result_json

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
    async def process_react_pattern(self, initial_prompt: str, session_id: str) -> Tuple[str, List[Dict], str, Optional[str]]:
        start_lang = self._detect_language(initial_prompt)
        logger.info(f"Starting ReAct Process (JSON Mode) for Session ID: {session_id}, Language: {start_lang}")
        session_log_dir = self.react_logs_dir / session_id
        session_log_dir.mkdir(parents=True, exist_ok=True)

        # --- 수정: 올바른 메소드를 사용하여 사용 가능한 도구 정보 로드 --- 
        all_available_tools = {}
        tool_details_for_prompt = {}
        try:
            # 실행 중인 서버 이름 목록 가져오기
            running_server_names = self.mcp_service.get_running_servers()
            logger.info(f"Detected running MCP servers: {running_server_names}")
            
            # 각 서버에서 도구 정보 가져오기 (get_server_tools 사용)
            for server_name in running_server_names:
                tools_info = self.mcp_service.get_server_tools(server_name) # 캐시된 정보 사용
                if tools_info:
                    all_available_tools[server_name] = list(tools_info.keys())
                    # --- 수정: 도구 이름 목록 대신 전체 tools_info(dict)를 전달 ---
                    tool_details_for_prompt[server_name] = tools_info
                else:
                    logger.warning(f"No tools found or cached for running server: {server_name}")
        except Exception as tool_err:
            logger.error(f"Error detecting available tools: {tool_err}", exc_info=True)
            # 도구 감지 실패 시 빈 목록으로 진행
            all_available_tools = {}
            tool_details_for_prompt = {}
        # --- 도구 정보 로드 로직 수정 끝 --- 
            
        # 세션 메타데이터 초기화
        session_meta = {
            "start_time": datetime.now().isoformat(), "prompt": initial_prompt,
            "model_path": str(self.model_path),
            "session_id": session_id,
            "start_language": start_lang,
            "available_tools": all_available_tools,
            "mode": "JSON_ReAct" # 모드 명시
        }
        structured_react_trace = [] # 변경: trace 구조를 json 객체 리스트로 변경
        error_message = None
        final_answer_for_user = None
        react_history = f"User: {initial_prompt}\n"
        iteration = 0
        conversation_log = react_history

        # ReAct 루프 시작
        for i in range(self.max_iterations):
            iteration = i + 1
            logger.info(f"--- ReAct Iteration {iteration}/{self.max_iterations} ---")
            
            # --- 수정: 매 반복마다 시스템 프롬프트 생성 --- 
            # 현재 반복 횟수에 따라 다른 프롬프트를 생성할 수 있음
            current_system_prompt = self._build_react_system_prompt(tool_details_for_prompt, iteration=iteration)
            
            # 현재 프롬프트 생성
            current_prompt = f"{current_system_prompt}\n\n{react_history}"
            
            logger.debug(f"Current prompt for LLM call (Iteration {iteration}):\n{current_prompt}")
            
            # LLM 호출 (복구된 _call_llm 사용)
            response_raw = None # 변수 초기화
            try:
                response_raw = await self._call_llm(current_prompt) # stop_tokens 인자 제거하여 기본값 사용
                
                # --- response_raw 직접 검사 로그 추가 (한 줄로 수정) --- 
                logger.debug(f"process_react_pattern: Raw response received (repr): {repr(response_raw)}")
                self._save_step_log(session_log_dir, iteration, "react_prompt", {"prompt": current_prompt})
                self._save_step_log(session_log_dir, iteration, "react_response", {"response": response_raw}) # 저장 전에 로그 찍기
            except Exception as e:
                logger.error(f"LLM call failed during ReAct iteration {iteration}: {e}", exc_info=True)
                error_message = f"LLM call failed: {str(e)}"
                structured_react_trace.append({"step": iteration, "error": error_message})
                break
                
            # --- JSON 추출 및 파싱 시도 (try 블록 분리) --- 
            json_response_str = None
            parsed_json: Optional[Dict[str, Any]] = None
            try: 
                json_response_str = self._clean_response(response_raw)

                # --- _clean_response 반환 값 및 조건문 결과 확인 (repr 추가) --- 
                response_type = type(json_response_str)
                response_repr = repr(json_response_str) # repr() 로 정확한 값 확인
                condition_result = bool(json_response_str) # 조건문 결과 미리 계산
                logger.debug(f"process_react_pattern: Received from _clean_response -> Type={response_type}, Value(repr)={response_repr}, ConditionResult={condition_result}")
                # --- 로그 추가 끝 --- 

                if condition_result: # 미리 계산된 조건 사용
                    logger.debug("process_react_pattern: Entering JSON parsing block...")
                    parsed_json = json.loads(json_response_str) # 여기서 오류 발생 가능성
                    # 필수 필드 검사 (수정: action 또는 answer는 null일 수 있음)
                    if not all(k in parsed_json for k in ["thought"]):
                        raise ValueError("Missing required field 'thought'")
                    if "action" not in parsed_json and "answer" not in parsed_json:
                         raise ValueError("Missing both required fields 'action' and 'answer'")
                    logger.debug("process_react_pattern: JSON parsed and basic validation passed.")
                else:
                    # json_response_str이 False로 평가되는 경우 (None 또는 빈 문자열)
                    logger.warning(f"process_react_pattern: _clean_response returned falsy value. Skipping JSON parsing.")
                    error_message = f"Invalid format in iteration {iteration}: Could not extract JSON."
                    structured_react_trace.append({"step": iteration, "raw_response": response_raw, "error": error_message})
                    break # JSON 없으면 루프 중단
            
            except (json.JSONDecodeError, ValueError) as json_e:
                # JSON 파싱 또는 구조 오류 처리
                logger.warning(f"process_react_pattern: JSON parsing/validation failed: {json_e}. JSON string attempted: {json_response_str}")
                error_message = f"Invalid format in iteration {iteration}: {json_e}"
                structured_react_trace.append({"step": iteration, "raw_response": response_raw, "error": error_message})
                break
            except Exception as clean_parse_e: # 예상치 못한 다른 오류
                logger.error(f"process_react_pattern: Unexpected error during clean/parse: {clean_parse_e}", exc_info=True)
                error_message = f"Unexpected error during response processing: {clean_parse_e}"
                structured_react_trace.append({"step": iteration, "raw_response": response_raw, "error": error_message})
                break
                
            # --- 오류 처리 로직 추가 (이전과 동일) --- 
            thought = parsed_json.get("thought", "") # 기본값 추가
            action_value = parsed_json.get("action") 
            answer_value = parsed_json.get("answer")
            
            # --- 최종 값으로 규칙 위반 확인 및 처리 --- 
            if action_value is not None and answer_value is not None:
                 logger.warning(f"Rule violation in iteration {iteration}: Both 'action' and 'answer' are non-null after processing. Prioritizing 'answer'.")
                 action_value = None # 답변 우선
                 
            # ----- MCP Tool 호출 또는 최종 답변 처리 ----- 
            if action_value is not None: # action 호출
                logger.info(f"Action determined: {action_value}")
                action_str = str(action_value) # 명시적 문자열 변환
                react_history += f"Action: {action_str}\n"
                
                # Initialize observation for potential errors
                observation = None 
                current_step_trace = { # 기본 트레이스 구조 미리 생성
                    "step": iteration,
                    "raw_response": response_raw,
                    "thought": thought,
                    "action": action_value,
                    "answer": answer_value,
                    "error": None, 
                    "observation": None 
                }

                try:
                    # 수정: action_str 파싱 강화
                    tool_call_match = re.match(r"([\w\-]+)/([\w\-]+)\s*\((.*)\)\s*$", action_str.strip()) # 서버명/도구명 추출
                    if not tool_call_match:
                        raise ValueError(f"Invalid action format: Could not match server/tool(...). Input: {action_str}")
                    
                    # 서버명, 도구명, 인자 문자열 분리
                    server_name = tool_call_match.group(1)
                    actual_tool_name = tool_call_match.group(2)
                    args_part_str = tool_call_match.group(3).strip() 
                    arguments_dict = None
                    logger.debug(f"Parsed Action: Server='{server_name}', Tool='{actual_tool_name}', ArgsStr='{args_part_str}'")

                    if args_part_str: # 인자 부분이 비어있지 않다면
                        try:
                            # 추가: 바깥쪽 중괄호 제거 시도
                            if args_part_str.startswith('{') and args_part_str.endswith('}'):
                                args_part_str_inner = args_part_str[1:-1]
                                logger.debug(f"Removed outer braces: {args_part_str_inner}")
                            else:
                                args_part_str_inner = args_part_str # 중괄호 없으면 그대로 사용
                            
                            # 추가: json.loads 전에 이스케이프 처리 (inner 사용)
                            unescaped_args_str = args_part_str_inner.replace('\\"', '"').replace('\\\\', '\\') # \" -> ", \\ -> \
                            logger.debug(f"Unescaped action arguments string (inner): {unescaped_args_str}")

                            # 괄호 안 내용을 JSON으로 파싱 시도 (이스케이프 처리된 문자열 사용)
                            arguments_dict = json.loads(unescaped_args_str)
                            if not isinstance(arguments_dict, dict):
                                raise ValueError(f"Parsed action arguments are not a dictionary: {arguments_dict}")
                            logger.debug(f"Successfully parsed action arguments for {server_name}/{actual_tool_name}: {arguments_dict}")
                        except json.JSONDecodeError as json_e:
                            # 파싱 실패 시 ValueError 발생시켜 에러 처리
                            logger.error(f"Failed to parse unescaped action arguments JSON for {server_name}/{actual_tool_name}: {json_e}. Original: {args_part_str}, Inner: {args_part_str_inner}, Unescaped: {unescaped_args_str}")
                            raise ValueError(f"Invalid JSON in action arguments (after processing): {unescaped_args_str}") from json_e
                    else:
                         # 인자 부분이 비어있으면 빈 딕셔너리 전달
                         arguments_dict = {}
                         logger.debug(f"No arguments provided for action {server_name}/{actual_tool_name}. Passing empty dict.")

                    # MCP 서비스 호출 (분리된 server_name, actual_tool_name, arguments_dict 전달)
                    raw_observation = await self.mcp_service.call_mcp_tool(server_name, actual_tool_name, arguments_dict)
                    # <<<--- 수정: 결과 포맷팅 및 변수 이름 명확화 --- >>>
                    observation_str = self._format_tool_result(raw_observation) 
                    logger.info(f"Observation from tool {server_name}/{actual_tool_name}: {observation_str[:100]}...") 
                    react_history += f"Observation: {observation_str}\n" # <<<--- 포맷팅된 문자열 사용
                    current_step_trace["observation"] = observation_str # <<<--- 포맷팅된 문자열 사용
                
                except (ValueError, KeyError, MCPError, ConnectionError) as e: # JSONDecodeError는 위에서 처리
                    logger.error(f"Error executing action \'{action_str}\' in iteration {iteration}: {e}", exc_info=True)
                    error_observation_str = f"Error executing action: {e}"
                    react_history += f"Observation: {error_observation_str}\n" # <<<--- 에러 문자열 사용
                    current_step_trace["error"] = f"Action execution failed: {e}" 
                    current_step_trace["observation"] = error_observation_str # <<<--- 에러 문자열 사용
                
                #structured_react_trace 업데이트는 try-except 블록 밖에서 한 번만 수행
                structured_react_trace.append(current_step_trace)
                
            elif answer_value is not None: # answer 반환 및 종료
                 logger.info(f"Final Answer determined: {answer_value}")
                 final_answer_for_user = str(answer_value)
                 break # 루프 종료
            
            else: # action과 answer 둘 다 null인 경우 (처리 후)
                logger.error(f"Critical Error: Both action and answer are null after processing in iteration {iteration}. This shouldn't happen.")
                error_message = f"Internal Error: Invalid state in iteration {iteration} (both action/answer are null)."
                # structured_react_trace에 에러 추가
                current_step_trace = {
                    "step": iteration,
                    "raw_response": response_raw,
                    "thought": thought,
                    "action": action_value,
                    "answer": answer_value,
                    "error": error_message,
                    "observation": None 
                }
                structured_react_trace.append(current_step_trace)
                break # <<<--- 오류로 간주하고 즉시 루프 중단

            # 최대 반복 횟수 도달 시
            if iteration == self.max_iterations:
                logger.warning("Reached max iterations.")
                error_message = "Agent stopped: Max iterations reached."
                final_error = error_message # 최종 에러 메시지 설정
                break

        # --- 루프 종료 후 처리 --- 
        is_final_answer = final_answer_for_user is not None
        if not is_final_answer and not error_message:
            logger.warning(f"Reached max iterations ({self.max_iterations}) without a final answer. Session ID: {session_id}")
            error_message = f"Reached max iterations ({self.max_iterations}) without finding an answer."
            if structured_react_trace:
                 # 마지막 스텝에 observation 추가 또는 error 메시지 추가
                 if "observation" not in structured_react_trace[-1]:
                      structured_react_trace[-1]["observation"] = f"(Agent stopped: {error_message})"
                 else:
                      structured_react_trace[-1]["error"] = structured_react_trace[-1].get("error", "") + f" (Agent stopped: {error_message})"
            else: 
                structured_react_trace.append({"step": 0, "error": f"(Agent stopped: {error_message})"})
        
        if not final_answer_for_user:
             final_answer_for_user = f"(Agent stopped: {error_message or 'Unknown error'})"

        # --- 최종 답변 번역 (한국어 입력 시) ---
        if start_lang == "ko" and final_answer_for_user and not error_message:
            # 오류 메시지가 아닌 실제 답변만 번역 시도
            if not final_answer_for_user.startswith("(Agent stopped:"):
                logger.info(f"Translating final answer to Korean for session {session_id}...")
                try:
                    # 영어 -> 한국어 번역 호출
                    translated_answer = await self._translate_text(final_answer_for_user, target_lang="ko", source_lang="en")
                    if translated_answer != final_answer_for_user: # 번역 성공 시 (원본과 다를 때)
                        logger.info("Translation applied.")
                        final_answer_for_user = translated_answer
                    else:
                        logger.warning("Translation returned original text or failed. Using original English answer.")
                except Exception as trans_e:
                    logger.error(f"Error during final translation step: {trans_e}", exc_info=True)
                    # 번역 실패 시 영어 답변 유지
            else:
                 logger.info("Skipping translation for agent stop message.")

        # --- 최종 메타데이터 저장 --- 
        session_meta["end_time"] = datetime.now().isoformat()
        session_meta["success"] = error_message is None and is_final_answer
        session_meta["reason"] = "completed" if session_meta["success"] else (error_message or "max_iterations_reached")
        session_meta["iterations_completed"] = iteration
        session_meta["final_error"] = error_message
        session_meta["structured_react_trace"] = structured_react_trace # JSON 객체 리스트
        session_meta["react_history"] = react_history

        logger.info(f"Attempting to save final meta.json for session {session_id} to {session_log_dir}")
        try:
            meta_file_path = session_log_dir / "meta.json"
            with open(meta_file_path, "w", encoding="utf-8") as f:
                # logger.debug(f"Final session_meta content for {session_id}: {json.dumps(session_meta, indent=2, ensure_ascii=False)}")
                json.dump(session_meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Fatal error saving final meta.json for session {session_id} to {session_log_dir}: {e}", exc_info=True)

        # 반환값 형식 유지
        return final_answer_for_user, structured_react_trace, conversation_log, error_message

    def _save_step_log(self, log_dir: Path, step: int, step_type: str, data: dict):
        """ReAct 단계별 로그를 JSON 파일로 저장합니다."""
        try:
            filename = f"{step:02d}_{step_type}.json"
            log_file_path = log_dir / filename
            with open(log_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save log file {log_file_path}: {e}")

# Dependency function 수정 불필요
def get_inference_service(settings: Settings = Depends(get_settings), mcp_service: MCPService = Depends()) -> 'InferenceService':
    from app.main import app_state
    if "inference_service" not in app_state:
         logger.critical("InferenceService가 app_state에 없습니다. 애플리케이션 초기화 실패 가능성.")
         raise RuntimeError("InferenceService not initialized. Check application startup.")
    inference_service_instance = app_state["inference_service"]
    if not inference_service_instance.model or not inference_service_instance.tokenizer:
        logger.warning("get_inference_service 호출 시 모델 또는 토크나이저가 로드되지 않음.")
    return inference_service_instance 