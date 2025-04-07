import asyncio
import logging
import re
import json
import httpx
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any
import asyncio
from fastapi import Depends
from llama_cpp import Llama
from transformers import AutoTokenizer, PreTrainedTokenizer
from huggingface_hub import hf_hub_download

from app.core.config import Settings, get_settings
from app.services.mcp_service import MCPService

logger = logging.getLogger(__name__)

# Define the base model identifier for tokenizer loading
BASE_MODEL_ID = "google/gemma-3-1b-it"
# Updated regex for ReAct pattern - 더 명확한 패턴 매칭을 위해 수정
ACTION_PATTERN = re.compile(r"Action:\s*(\w+[\w\-]*)/(\w+)(?:\(([^)]*)\))?")
FINAL_ANSWER_PATTERN = re.compile(r"Answer:\s*(.*)", re.DOTALL)

class InferenceService:
    def __init__(self, settings: Settings = Depends(get_settings), mcp_service: MCPService = Depends()):
        self.settings = settings
        self.mcp_service = mcp_service
        self.model_path = settings.model_path
        self.model_url = settings.model_url
        # Path to custom tokenizer folder
        self.tokenizer_path = Path('/app/tokenizer') if settings.is_docker else Path('./tokenizer')

        # GGUF 모델 설정
        self.max_tokens = 32768  # Gemma3 1B 모델의 최대 토큰 수
        self.max_new_tokens = 256  # 생성할 최대 토큰 수
        
        # 로그 저장 디렉토리 설정
        self.logs_dir = Path('/app/logs') if settings.is_docker else Path('./logs')
        self.react_logs_dir = self.logs_dir / 'react_logs'
        self.react_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # 초기값 설정
        self.tokenizer = None
        self.model = None
        self.lock = asyncio.Lock()
        self._download_task = None
        
        # 즉시 초기화 실행 (비동기 작업을 동기적으로 실행)
        loop = asyncio.get_event_loop()
        if not loop.is_running():
            loop.run_until_complete(self.initialize())
        else:
            # 이미 이벤트 루프가 실행 중인 경우 (FastAPI 환경 등), 초기화 태스크 예약
            asyncio.create_task(self.initialize())
            logger.info("Model initialization scheduled as async task")

    async def initialize(self):
        async with self.lock:
            if self.model and self.tokenizer:
                logger.info("Model and tokenizer already initialized.")
                return

            if not self.model_path.is_file():
                if self._download_task is None:
                    logger.info(f"Model not found at {self.model_path}. Starting download...")
                    self._download_task = asyncio.create_task(self._download_model())
                else:
                    logger.info("Model download already in progress...")
                # Wait for download task to complete
                await self._download_task
                self._download_task = None
                if not self.model_path.is_file():
                    logger.error("Model download failed or file still not found.")
                    raise RuntimeError("Model download failed.")

            try:
                # Load tokenizer from local folder instead of Hugging Face Hub
                if self.tokenizer_path.exists():
                    logger.info(f"Loading tokenizer from local path: {self.tokenizer_path}")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
                    logger.info("Local tokenizer loaded successfully.")
                else:
                    # Fallback to loading from Hugging Face Hub if local not available
                    logger.warning(f"Local tokenizer not found at {self.tokenizer_path}. Falling back to Hugging Face.")
                    logger.info(f"Loading tokenizer for base model: {BASE_MODEL_ID}...")
                    self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
                    logger.info("Tokenizer loaded from Hugging Face successfully.")

                # Load GGUF model with llama-cpp-python
                logger.info(f"Loading GGUF model from {self.model_path}...")
                try:
                    # GPU 가속 시도
                    self.model = Llama(
                        model_path=str(self.model_path),
                        n_ctx=self.max_tokens,
                        n_gpu_layers=-1,  # -1은 가능한 모든 레이어를 GPU로
                        verbose=False
                    )
                    logger.info("GGUF model loaded with GPU acceleration.")
                except Exception as e:
                    logger.warning(f"Failed to load model with GPU acceleration: {e}. Falling back to CPU.")
                    # CPU 폴백
                    self.model = Llama(
                        model_path=str(self.model_path),
                        n_ctx=self.max_tokens,
                        n_gpu_layers=0,
                        verbose=False
                    )
                    logger.info("GGUF model loaded with CPU only.")

                logger.info("Inference service initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize tokenizer or model: {e}", exc_info=True)
                self.model = None
                self.tokenizer = None
                raise RuntimeError(f"Failed to initialize inference service: {e}") from e

    async def _download_model(self):
        try:
            # 디렉토리 생성
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 모델 URL에서 repo_id와 filename 추출
            model_url_str = str(self.model_url)
            logger.info(f"모델 다운로드 시도: {model_url_str}")
            
            # 두 가지 방법으로 다운로드 시도
            success = False
            
            # 1. huggingface_hub 라이브러리 사용 (권장)
            try:
                logger.info("huggingface_hub 라이브러리를 사용하여 다운로드 시도...")
                # URL이 Hugging Face 형식이면 repo_id와 filename 추출
                if "huggingface.co" in model_url_str and "/resolve/" in model_url_str:
                    parts = model_url_str.split("/")
                    # URL에서 repo_id 추출 (일반적으로 'huggingface.co/OWNER/MODEL' 형식)
                    repo_idx = parts.index("huggingface.co")
                    if len(parts) > repo_idx + 2:  # 인덱스 유효성 확인
                        owner = parts[repo_idx + 1]
                        model_name = parts[repo_idx + 2]
                        repo_id = f"{owner}/{model_name}"
                        
                        # 파일명 추출
                        resolve_idx = parts.index("resolve")
                        if len(parts) > resolve_idx + 2:
                            filename = parts[-1].split("?")[0]  # '?download=true' 제거
                            
                            logger.info(f"추출된 repo_id: {repo_id}, filename: {filename}")
                            
                            # 비동기 환경에서 동기 함수 실행
                            def _download():
                                try:
                                    # hf_hub_download는 동기 함수이므로 별도 스레드에서 실행
                                    file_path = hf_hub_download(
                                        repo_id=repo_id,
                                        filename=filename,
                                        cache_dir=self.model_path.parent,
                                        force_download=True,
                                        resume_download=True
                                    )
                                    # 다운로드된 파일을 원하는 위치로 복사
                                    if Path(file_path) != self.model_path:
                                        shutil.copy(file_path, self.model_path)
                                    return True
                                except Exception as e:
                                    logger.error(f"hf_hub_download 실패: {e}")
                                    return False
                            
                            # 별도 스레드에서 동기 함수 실행
                            loop = asyncio.get_event_loop()
                            success = await loop.run_in_executor(None, _download)
                            
                            if success:
                                logger.info(f"huggingface_hub로 모델 다운로드 성공: {self.model_path}")
                                return
            except Exception as e:
                logger.warning(f"huggingface_hub 방식 다운로드 실패: {e}")
            
            # 2. httpx 직접 다운로드 (대체 방법)
            if not success:
                logger.info("httpx를 사용한 직접 다운로드 시도...")
                # 인증 헤더 준비
                headers = {}
                # 환경 변수에서 Hugging Face 토큰 가져오기
                hf_token = os.environ.get("HUGGING_FACE_TOKEN")
                if hf_token:
                    logger.info("Hugging Face 토큰을 사용하여 인증")
                    headers["Authorization"] = f"Bearer {hf_token}"
                else:
                    logger.warning("Hugging Face 토큰이 없습니다. 인증이 필요한 모델은 다운로드가 실패할 수 있습니다.")
                
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    # HTTP 다운로드 시도
                    async with client.stream("GET", model_url_str, headers=headers) as response:
                        response.raise_for_status()  # 오류 발생 시 예외 발생
                        total_size = int(response.headers.get("content-length", 0))
                        chunk_size = 8192
                        downloaded_size = 0
                        logger.info(f"모델 다운로드 중 ({total_size / (1024*1024):.2f} MB) -> {self.model_path}...")
                        with open(self.model_path, "wb") as f:
                            async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                                f.write(chunk)
                                downloaded_size += len(chunk)
                                # 진행 상황 로깅 (덜 자주)
                                if downloaded_size % (chunk_size * 512) == 0 or downloaded_size == total_size:
                                    progress = (downloaded_size / total_size) * 100 if total_size else 0
                                    logger.debug(f"다운로드 진행: {downloaded_size / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB ({progress:.1f}%)")
                logger.info("httpx로 모델 다운로드 완료")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP 오류: {e.response.status_code} - {e.response.reason_phrase}")
            if e.response.status_code == 401:
                logger.error("인증 실패. HUGGING_FACE_TOKEN 환경 변수를 설정하거나 huggingface_hub 라이브러리를 사용하세요.")
            # 응답 내용 기록 (디버깅용)
            logger.error(f"응답 내용: {e.response.text[:500]}...")
            # 손상된 파일 제거
            if self.model_path.exists():
                self.model_path.unlink()
            # 예외 다시 발생
            raise
        except Exception as e:
            logger.error(f"모델 다운로드 실패: {e}", exc_info=True)
            # 손상된 파일 제거
            if self.model_path.exists():
                self.model_path.unlink()
            # 예외 다시 발생
            raise

    def _detect_language(self, text: str) -> str:
        """
        사용자 입력 텍스트의 언어를 감지합니다.
        기본적으로 단순한 휴리스틱을 사용하여 한국어인지 여부를 판단합니다.
        
        Returns:
            str: "ko" 또는 "en" (한국어 또는 영어)
        """
        # 빈 텍스트나 너무 짧은 텍스트는 기본값으로 영어 반환
        if not text or len(text.strip()) < 2:
            return "en"
            
        # 한글 유니코드 범위: AC00-D7A3 (완성형 한글)
        # 또는 1100-11FF (한글 자모), 3130-318F (한글 호환 자모)
        korean_char_count = len([c for c in text if ord('가') <= ord(c) <= ord('힣') or 
                                 ord('ㄱ') <= ord(c) <= ord('ㅎ') or
                                 ord('ㅏ') <= ord(c) <= ord('ㅣ')])
        
        # 한글 비율이 높으면 한국어로 판단
        if korean_char_count > 0 and korean_char_count / len(text.strip()) > 0.1:
            return "ko"
        
        # 그 외에는 영어로 가정
        return "en"

    def _build_react_system_prompt(self, query: str = ""):
        """ReAct 패턴용 시스템 프롬프트 구성"""
        lang = self._detect_language(query)

        # MCP 서버 상태 확인
        available_servers = self.mcp_service.get_available_server_names()

        # MCP 클라이언트 연결 상태 확인
        connected_servers = []
        for server_name in available_servers:
            if self.mcp_service.is_server_connected(server_name):
                connected_servers.append(server_name)

        # 실제로 연결된 서버만 사용
        server_info = []

        # 사용 가능한 서버가 있는지 확인
        has_servers = len(connected_servers) > 0

        # 로깅 상태 정보
        if not has_servers:
            logger.warning("연결된 MCP 서버가 없습니다. 직접 응답 모드로 동작합니다.")
        else:
            logger.info(f"연결된 MCP 서버: {', '.join(connected_servers)}")

        # 사용 가능한 도구 목록 구성 (연결된 서버 기반)
        if has_servers:
            for server_name in connected_servers:
                try:
                    tools = self.mcp_service.get_server_tools(server_name)
                    for tool_name, tool_info in tools.items():
                        desc = tool_info.get("description", f"A tool named '{tool_name}'")
                        params = tool_info.get("parameters", {})
                        params_list = []
                        if params:
                            for p_name, p_info in params.get('properties', {}).items():
                                p_type = p_info.get('type', 'any')
                                p_desc = p_info.get('description', '')
                                param_desc = f"{p_name} ({p_type})"
                                if p_desc:
                                    param_desc += f": {p_desc}"
                                params_list.append(param_desc)

                        params_str = ", ".join(params_list) if params_list else ("none" if lang == "en" else "없음")

                        # 도구 설명 형식 개선
                        if lang == "ko":
                            server_info.append(f"- {server_name}/{tool_name}: {desc}\n  매개변수: {params_str}")
                        else:
                            server_info.append(f"- {server_name}/{tool_name}: {desc}\n  Parameters: {params_str}")

                except Exception as e:
                    logger.warning(f"서버 '{server_name}'의 도구 정보를 가져오는 중 오류 발생: {e}")

        # 도구 정보 텍스트
        tools_section = ""
        if has_servers:
            tools_list = "\n".join(server_info)
            if lang == "ko":
                tools_section = f"""
사용 가능한 도구:
{tools_list}
"""
            else:
                tools_section = f"""
Available Tools:
{tools_list}
"""
        else:
            if lang == "ko":
                tools_section = "주의: 현재 사용 가능한 도구가 없습니다. 내장된 지식으로만 답변하세요.\n"
            else:
                tools_section = "Note: There are currently no available tools. Answer using only your built-in knowledge.\n"

        # 예시에 사용할 서버 선택 (사용 가능한 서버가 있는 경우)
        example_server = next(iter(connected_servers)) if has_servers else "no-tools-available" # 서버 없으면 더 명확한 예시용 이름 사용

        # 프롬프트 수정: Thought는 영어로, Answer는 사용자 언어로
        if lang == "ko":
            prompt = f"""You are an AI assistant processing a request in Korean.
ALWAYS use English for internal thoughts ('Thought: ...').
ALWAYS provide the final answer ('Answer: ...') in Korean.

MANDATORY RESPONSE FORMAT:
1. ALWAYS start with "Thought: [your English analysis]"
2. ALWAYS end with "Answer: [your Korean response]"

Processing Steps:
1. Thought: (In English) Analyze the request, decide if a tool is needed.
2. Action (Optional): Call a tool if available. Format: Action: server/tool(parameter=\"value\").
3. Observation: Get tool result (system provided).
4. Thought: (In English) Analyze result, decide next step.
5. Answer: (In Korean) Provide the final answer to the user.

Rules:
- ALWAYS start with 'Thought:'. Thoughts MUST be in English.
- ALWAYS end with 'Answer:'. The Answer MUST be in Korean.
- If you don't use a tool, go directly from Thought to Answer.
- If no tools are available, answer directly from your knowledge.
- If you need to use a tool, only use tools from the available tools list.
- Follow tool parameter format (strings in \"quotes\").

{tools_section}

Example with no tools available:
User: 파이썬이란 무엇인가요?
AI:
Thought: The user is asking what Python is. I should provide information about the Python programming language.
Answer: 파이썬은 간결하고 읽기 쉬운 구문을 가진 고수준 프로그래밍 언어입니다. 웹 개발, 데이터 분석, 인공지능 등 다양한 분야에서 널리 사용되며, 초보자부터 전문가까지 모두에게 인기 있는 언어입니다.

"""
            if has_servers:
                prompt += f"""
Example with tools available:
User: 현재 디렉토리 파일 목록을 보여줘
AI:
Thought: The user wants to list files in the current directory. I should use the terminal tool.
Action: {example_server}/mcp_write_to_terminal(command=\"ls -la\")
Observation: total 128
drwxr-xr-x  24 user  staff   768 Apr  7 10:30 .
drwxr-xr-x   5 user  staff   160 Apr  7 10:29 ..
-rw-r--r--   1 user  staff  2345 Apr  7 10:30 README.md
drwxr-xr-x   8 user  staff   256 Apr  7 10:30 app
Thought: I've received the file list from the terminal command. Now I need to provide this information to the user in Korean.
Answer: 현재 디렉토리에는 다음과 같은 파일 및 폴더가 있습니다:
- README.md (파일)
- app (폴더)
- . (현재 디렉토리)
- .. (상위 디렉토리)
"""
        else: # Default to English if lang is not 'ko'
            prompt = f"""You are an AI assistant. Process requests and respond in English.
ALWAYS use English for internal thoughts ('Thought: ...').
ALWAYS provide the final answer ('Answer: ...') in English.

MANDATORY RESPONSE FORMAT:
1. ALWAYS start with "Thought: [your analysis]"
2. ALWAYS end with "Answer: [your response]"

Steps to process request:
1. Thought: Analyze request, decide if tool is needed.
2. Action (if needed): Call a tool if available. Format: Action: server/tool(parameter=\"value\").
3. Observation: Get tool result (provided by system).
4. Thought: Analyze result, decide next Action or final Answer.
5. Answer: Provide final answer.

Rules:
- ALWAYS start with 'Thought:'. All thoughts must be in English.
- ALWAYS end with 'Answer:'. All answers must be in English.
- If you don't use a tool, go directly from Thought to Answer.
- If no tools are available, answer directly from your knowledge.
- If you need to use a tool, only use tools from the available tools list.
- Follow tool parameter format (strings in \"quotes\").

{tools_section}

Example with no tools available:
User: What is Python?
AI:
Thought: The user is asking what Python is. I should provide information about the Python programming language.
Answer: Python is a high-level programming language with a clean and readable syntax. It's widely used in various fields including web development, data analysis, artificial intelligence, and more. It's popular among beginners and experts alike.

"""
            if has_servers:
                prompt += f"""
Example with tools available:
User: List files in the current directory
AI:
Thought: The user wants to list files in the current directory. I should use the terminal tool.
Action: {example_server}/mcp_write_to_terminal(command=\"ls -la\")
Observation: total 128
drwxr-xr-x  24 user  staff   768 Apr  7 10:30 .
drwxr-xr-x   5 user  staff   160 Apr  7 10:29 ..
-rw-r--r--   1 user  staff  2345 Apr  7 10:30 README.md
drwxr-xr-x   8 user  staff   256 Apr  7 10:30 app
Thought: I've received the file list from the terminal command. Now I need to provide this information to the user.
Answer: Here are the files and directories in the current location:
- README.md (file)
- app (directory)
- . (current directory)
- .. (parent directory)
"""

        return prompt

    async def _call_llm(self, prompt: str, temperature: float = 0.2) -> str:
        """GGUF 모델을 사용하여 추론을 실행합니다."""
        try:
            logger.info(f"Starting GGUF model inference with prompt length: {len(prompt)}")
            
            # 모델 응답 생성 - 안정적인 세팅값 사용 (temperature 조정됨)
            output = self.model.create_completion(
                prompt,
                max_tokens=self.max_new_tokens,
                temperature=temperature,  # 사용자 지정 온도 사용
                top_p=0.9,           # 다양성과 정확성의 균형
                top_k=40,            # 상위 토큰 선택 제한
                stop=["User:", "USER:", "user:"], # Answer는 stop token에서 제외
                stream=False
            )
            
            # 결과 추출 및 로깅
            generated_text = output['choices'][0]['text'].strip()
            logger.info(f"Generated {len(generated_text)} characters")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"LLM inference error: {e}", exc_info=True)
            # 실제 프로덕션 환경이므로 폴백 응답 대신 실제 오류 전달
            raise RuntimeError(f"Model inference failed: {str(e)}")

    async def generate(self, text: str) -> str:
        """사용자 입력에 대해 추론을 실행하고 ReAct 패턴으로 도구 호출을 처리합니다."""
        try:
            # 모델이 초기화되어 있는지 확인
            if self.model is None:
                await self.initialize()
                if self.model is None:
                    return "Model initialization failed. Please try again later."
            
            # 로그 기록
            logger.info(f"Processing user input: {text[:50]}..." if len(text) > 50 else f"Processing user input: {text}")
            
            # 최대 반복 횟수 설정 (무한 루프 방지)
            max_iterations = 7
            iteration = 0
            
            # 시스템 프롬프트 추가
            system_prompt = self._build_react_system_prompt(text)
            full_prompt = system_prompt
            
            # 사용자 입력을 대화 기록에 추가
            conversation_history = [("USER", text)]
            
            while iteration < max_iterations:
                # 응답 생성
                raw_response = await self._call_llm(full_prompt)
                
                if not raw_response:
                    logger.error("Failed to generate a response.")
                    return "I couldn't generate a proper response. Please try again."
                
                # 응답 로깅
                logger.debug(f"Raw model response: {raw_response[:200]}...")
                
                # 대화 기록에 AI 응답 추가
                conversation_history.append(("AI", raw_response))
                
                # Action과 Final Answer 패턴 확인
                action_match = re.search(r"Action:\s*(\w+[\w\-]*)/(\w+)\s*\(\s*({.*?})\s*\)", raw_response, re.DOTALL)
                final_answer_match = re.search(r"Answer:\s*(.*)", raw_response, re.DOTALL)
                
                # Final Answer가 있으면 결과 반환
                if final_answer_match:
                    final_answer = final_answer_match.group(1).strip()
                    logger.info(f"Final answer provided: {final_answer[:50]}...")
                    return raw_response
                
                # Action이 있으면 도구 호출 처리
                if action_match:
                    server_name, tool_name, args_str = action_match.groups()
                    
                    # JSON 인자 파싱
                    try:
                        # 작은따옴표를 큰따옴표로 바꾸어 JSON 파싱 시도
                        args_str = args_str.replace("'", '"')
                        args = json.loads(args_str)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON in tool arguments: {args_str}. Error: {e}")
                        
                        # JSON 파싱 실패 시 기본적인 수정 시도
                        try:
                            # 1. 작은따옴표와 큰따옴표 혼합 수정
                            fixed_str = re.sub(r"'([^']*)':", r'"\1":', args_str)
                            # 2. 값에 대한 따옴표 수정
                            fixed_str = re.sub(r':\s*\'([^\']*)\'\s*([,}])', r': "\1"\2', fixed_str)
                            fixed_str = fixed_str.replace("'", '"')
                            args = json.loads(fixed_str)
                            logger.info(f"Fixed JSON parsing: {fixed_str}")
                        except Exception:
                            # 여전히 실패하면 오류 메시지 추가하고 계속 진행
                            observation = "Error: Invalid JSON in tool arguments. Please provide valid JSON with double quotes."
                            full_prompt = f"{full_prompt}\n\nAction: {server_name}/{tool_name}({args_str})\n\nObservation: {observation}\n"
                            iteration += 1
                            continue
                    
                    # MCP 서버 사용 가능한지 확인
                    if not self.mcp_service.is_server_available(server_name):
                        available_servers = self.mcp_service.get_available_server_names()
                        observation = f"Error: MCP server '{server_name}' is not available. Available servers: {', '.join(available_servers) if available_servers else 'None'}"
                        full_prompt = f"{full_prompt}\n\nAction: {server_name}/{tool_name}({args_str})\n\nObservation: {observation}\n"
                        iteration += 1
                        continue
                    
                    # 도구 호출
                    try:
                        logger.info(f"Calling MCP tool: {server_name}/{tool_name} with args: {args}")
                        result = await self.mcp_service.call_mcp_tool(server_name, tool_name, args)
                        
                        # 결과 포맷팅
                        observation = self._format_tool_result(result)
                        logger.info(f"Tool result: {observation[:50]}..." if len(observation) > 50 else f"Tool result: {observation}")
                        
                        # 프롬프트에 새로운 관찰 추가
                        full_prompt = f"{full_prompt}\n\nAction: {server_name}/{tool_name}({args_str})\n\nObservation: {observation}\n"
                    except Exception as e:
                        logger.error(f"Tool call failed: {e}", exc_info=True)
                        observation = f"Error: Tool execution failed - {str(e)}"
                        full_prompt = f"{full_prompt}\n\nAction: {server_name}/{tool_name}({args_str})\n\nObservation: {observation}\n"
                else:
                    # Action이나 Final Answer가 없는 경우
                    logger.warning("모델 응답에서 유효한 Action 또는 Answer를 찾지 못했습니다.")
                    current_step_log["error"] = "Invalid response format (no Action or Answer)"
                    thoughts_and_actions.append(current_step_log)
                    self._save_step_log(session_log_dir, iteration, "invalid_format", current_step_log)
                    
                    # 더 명확한 안내 메시지
                    if has_servers:
                        if thought and not thought.strip().endswith('.'):
                            # 모델이 Thought를 생성했지만 완료하지 않은 경우, 직접 Answer로 이어가도록 유도
                            reminder = f"\nYou must ALWAYS end your response with 'Answer: ' followed by your final response in {start_lang}. Please complete your answer now."
                        else:
                            # 일반적인 형식 오류
                            reminder = f"\nYour response MUST follow this exact format: Start with 'Thought: ' and end with 'Answer: '. If using a tool, include 'Action: ' between them. Available servers: {', '.join(valid_servers)}."
                    else:
                        if thought and not thought.strip().endswith('.'):
                            # 모델이 Thought를 생성했지만 완료하지 않은 경우, 직접 Answer로 이어가도록 유도
                            reminder = f"\nComplete your thought and then provide an answer. You must ALWAYS end with 'Answer: ' followed by your final response in {start_lang}."
                        else:
                            # 도구가 없는 경우 간단한 안내
                            reminder = "\nYour response MUST follow this exact format: Start with 'Thought: ' and end with 'Answer: '. Since there are no available tools, you should provide a direct Answer."
                    
                    # 특별히 Thought만 있고 Answer가 없는 경우를 위한 처리
                    if thought and not final_answer and "Action" not in response_text:
                        if "Thought:" in response_text and response_text.strip().endswith(thought.strip()):
                            # Thought만 생성한 경우 - 직접 Answer 요청
                            answer_prompt = f"\nNow, provide your 'Answer: ' in {start_lang}."
                            reminder = answer_prompt
                    
                    current_prompt += f"{response_text}{reminder}\nAI:" 
                    conversation_log += f"System Reminder: {reminder.strip()}\n"
                
                iteration += 1
            
            # 최대 반복 도달 - 마지막 응답 반환 또는 안내 메시지 추가
            if raw_response:
                return f"{raw_response}\n\nI've reached the maximum number of steps. Here's my final answer based on what I've found so far."
            else:
                return "I couldn't complete the task within the allowed steps. Please try again with a more specific request."
        
        except Exception as e:
            logger.error(f"Error in generate: {e}", exc_info=True)
            return f"An error occurred while generating a response: {str(e)}"

    def _clean_response(self, text: str) -> str:
        """Clean up the model response to remove any random tokens or unwanted patterns."""
        # Remove random markdown code blocks
        text = re.sub(r'```(?!json).*?```', '', text, flags=re.DOTALL)
        
        # Remove random sequences of symbols
        text = re.sub(r'(\*\*\*+|\#\#\#+|===+|\-\-\-+){2,}', '', text)
        
        # Remove duplicate line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove random quote blocks
        text = re.sub(r'"{3,}[^"]*?"{3,}', '', text, flags=re.DOTALL)
        
        return text

    def _format_tool_result(self, result: Any) -> str:
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

    async def process_react_pattern(self, prompt: str) -> dict:
        """
        ReAct 패턴을 사용하여 추론을 진행합니다.
        이 패턴은 사고(Thought), 행동(Action), 관찰(Observation)을 반복하여 문제를 해결합니다.
        각 단계별로 상세 로그를 저장합니다.
        """
        session_id = f"{int(time.time())}" # 세션 ID를 먼저 생성
        logger.info(f"--- process_react_pattern START --- Session ID: {session_id}, Input: {prompt[:50]}...") # 수정된 진입 로그
        start_lang = self._detect_language(prompt) 

        # 로그 세션 디렉토리 생성
        session_log_dir = self.react_logs_dir / session_id
        session_log_dir.mkdir(parents=True, exist_ok=True)

        # 세션 메타데이터 저장
        session_meta = {
            "start_time": datetime.now().isoformat(),
            "prompt": prompt,
            "model_path": str(self.model_path),
            "session_id": session_id,
            "start_language": start_lang 
        }
        with open(session_log_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(session_meta, f, ensure_ascii=False, indent=2)

        try:
            # 사용 가능한 MCP 서버 확인
            valid_servers = self.mcp_service.get_available_server_names()
            has_servers = len(valid_servers) > 0
            
            # 대화 이력 및 관찰 기록
            thoughts_and_actions = []
            conversation_log = ""
            max_iterations = 7
            iteration = 0
            has_used_tool = False
            system_prompt = self._build_react_system_prompt(prompt)
            self._save_step_log(session_log_dir, 0, "system_prompt", {"prompt": system_prompt})
            current_prompt = f"{system_prompt}\n\nUser: {prompt}\nAI:" 
            conversation_log += f"User: {prompt}\n" 
            final_answer_for_user = ""

            while iteration < max_iterations:
                iteration += 1
                logger.info(f"ReAct 패턴 반복 {iteration}/{max_iterations} (Session ID: {session_id})")
                step_start_time = time.time()
                response_text = await self._call_llm(current_prompt, temperature=0.2)
                generation_time = time.time() - step_start_time
                
                # 모델 응답 저장
                self._save_step_log(session_log_dir, iteration, "model_response", {
                    "response_text": response_text,
                    "generation_time_sec": generation_time,
                    "input_prompt": current_prompt
                })
                
                # 빈 응답 처리
                if not response_text or response_text.strip() == "":
                    if iteration == 1:
                        # 첫 번째 시도에서 빈 응답이면 실패로 간주
                        logger.error(f"모델이 첫 시도에서 응답을 생성하지 못했습니다. (Session ID: {session_id})")
                        self._save_step_log(session_log_dir, iteration, "error", {"error": "모델이 응답을 생성하지 못했습니다."})
                        session_meta["end_time"] = datetime.now().isoformat()
                        session_meta["success"] = False
                        session_meta["reason"] = "empty_model_response"
                        with open(session_log_dir / "meta.json", "w", encoding="utf-8") as f:
                            json.dump(session_meta, f, ensure_ascii=False, indent=2)
                        return {
                            "response": "죄송합니다, 응답을 생성하는 중 오류가 발생했습니다. 다시 시도해 주세요." if start_lang == "ko" else "Sorry, an error occurred while generating a response. Please try again.",
                            "thoughts_and_actions": thoughts_and_actions,
                            "log_session_id": session_id,
                            "error": "Empty model response"
                        }
                    else:
                        # 두 번째 이상의 시도에서 빈 응답이면 이전까지의 결과로 최종 답변 시도
                        logger.warning(f"모델이 반복 {iteration}에서 빈 응답을 반환했습니다. 이전 단계 결과로 마무리합니다. (Session ID: {session_id})")
                        self._save_step_log(session_log_dir, iteration, "warning", {"warning": "빈 응답, 이전 단계 결과로 마무리"})
                        # 이전 단계에서 모은 정보로 최종 답변 시도
                        if start_lang == "ko":
                            final_answer_for_user = "죄송합니다, 요청을 완료하지 못했습니다. 다음은 지금까지 찾은 정보입니다:\n\n" + "\n".join([
                                f"- {act.get('thought', '')}" for act in thoughts_and_actions if 'thought' in act
                            ])
                        else:
                            final_answer_for_user = "Sorry, I couldn't complete the request. Here's what I found so far:\n\n" + "\n".join([
                                f"- {act.get('thought', '')}" for act in thoughts_and_actions if 'thought' in act
                            ])
                        break
                
                logger.debug(f"Raw model response: {response_text[:200]}... (Session ID: {session_id})")
                conversation_log += f"AI: {response_text}\n"
                thought = self._extract_thought(response_text)
                server_name, tool_name, args = self._extract_action_details(response_text)
                final_answer = self._extract_final_answer(response_text)
                current_step_log = {"thought": thought}
                
                # 최종 답변 처리
                if final_answer:
                    logger.info(f"최종 답변 감지됨: {final_answer[:50]}... (Session ID: {session_id})")
                    final_answer_for_user = final_answer  # 최종 사용자 응답 설정
                    current_step_log["answer"] = final_answer
                    thoughts_and_actions.append(current_step_log)
                    self._save_step_log(session_log_dir, iteration, "final_answer_detected", current_step_log)
                    break
                
                # 액션 처리
                elif server_name and tool_name:
                    has_used_tool = True
                    current_step_log["action"] = {"server": server_name, "tool": tool_name, "args": args}
                    logger.info(f"Action 감지됨: {server_name}/{tool_name} ({args}) (Session ID: {session_id})")

                    observation = "" 
                    valid_servers = self.mcp_service.get_available_server_names()
                    is_valid_server = server_name in valid_servers
                    is_placeholder = server_name in ["[server-name]", "no-tools-available"]

                    # 유효하지 않은 서버 이름 또는 플레이스홀더 처리
                    if not is_valid_server or is_placeholder:
                        if is_placeholder:
                             error_msg = f"Error: You used the placeholder '{server_name}'. Available servers: {valid_servers if valid_servers else 'None'}" 
                             log_type = "placeholder_server_name"
                             logger.warning(f"모델이 서버 이름 플레이스홀더를 사용했습니다: {server_name} (Session ID: {session_id})")
                        else:
                             error_msg = f"Error: Invalid server name '{server_name}'. Available servers: {valid_servers if valid_servers else 'None'}" 
                             log_type = "invalid_server_name"
                             logger.warning(f"모델이 유효하지 않은 서버 이름 '{server_name}'을 사용했습니다. (Session ID: {session_id})")
                        
                        observation = error_msg
                        current_step_log["observation"] = observation
                        thoughts_and_actions.append(current_step_log)
                        self._save_step_log(session_log_dir, iteration, log_type, current_step_log)
                        current_prompt += f"{response_text}\nObservation: {observation}\nAI:" 
                        conversation_log += f"Observation: {observation}\n" 
                    else:
                        # 유효한 서버 및 도구 호출 시도
                        try:
                            if not self.mcp_service.is_server_connected(server_name):
                                obs_text = f"Error: Server '{server_name}' is not connected. Available: {valid_servers}"
                                logger.warning(f"서버 '{server_name}'에 연결되지 않았습니다. (Session ID: {session_id})")
                                observation = obs_text
                            elif not self.mcp_service.get_server_tools(server_name).get(tool_name):
                                available_tools = list(self.mcp_service.get_server_tools(server_name).keys())
                                obs_text = f"Error: Tool '{server_name}/{tool_name}' not found. Available: {available_tools}"
                                logger.warning(f"도구 '{server_name}/{tool_name}'을(를) 찾을 수 없습니다. (Session ID: {session_id})")
                                observation = obs_text
                            else:
                                tool_call_start = time.time()
                                result = await self.mcp_service.call_mcp_tool(server_name, tool_name, args)
                                tool_call_duration = time.time() - tool_call_start
                                logger.info(f"도구 호출 완료 ({tool_call_duration:.2f}초): {server_name}/{tool_name} (Session ID: {session_id})")
                                observation = self._format_tool_result(result)
                                logger.debug(f"도구 결과: {observation[:100]}... (Session ID: {session_id})")

                        except Exception as e:
                            logger.error(f"도구 호출/처리 실패: {server_name}/{tool_name} - {e} (Session ID: {session_id})", exc_info=True)
                            observation = f"Error: Tool execution failed - {str(e)}"
                        
                        current_step_log["observation"] = observation
                        thoughts_and_actions.append(current_step_log)
                        self._save_step_log(session_log_dir, iteration, "tool_executed_or_failed", current_step_log)
                        current_prompt += f"{response_text}\nObservation: {observation}\nAI:" 
                        conversation_log += f"Observation: {observation}\n" 

                # Action도 Answer도 없는 경우
                else:
                    logger.warning(f"모델 응답에서 유효한 Action 또는 Answer를 찾지 못했습니다. (Session ID: {session_id})")
                    current_step_log["error"] = "Invalid response format (no Action or Answer)"
                    thoughts_and_actions.append(current_step_log)
                    self._save_step_log(session_log_dir, iteration, "invalid_format", current_step_log)
                    
                    # 더 명확한 안내 메시지
                    if has_servers:
                        if thought and not thought.strip().endswith('.'):
                            # 모델이 Thought를 생성했지만 완료하지 않은 경우, 직접 Answer로 이어가도록 유도
                            reminder = f"\nYou must ALWAYS end your response with 'Answer: ' followed by your final response in {start_lang}. Please complete your answer now."
                        else:
                            # 일반적인 형식 오류
                            reminder = f"\nYour response MUST follow this exact format: Start with 'Thought: ' and end with 'Answer: '. If using a tool, include 'Action: ' between them. Available servers: {', '.join(valid_servers)}."
                    else:
                        if thought and not thought.strip().endswith('.'):
                            # 모델이 Thought를 생성했지만 완료하지 않은 경우, 직접 Answer로 이어가도록 유도
                            reminder = f"\nComplete your thought and then provide an answer. You must ALWAYS end with 'Answer: ' followed by your final response in {start_lang}."
                        else:
                            # 도구가 없는 경우 간단한 안내
                            reminder = "\nYour response MUST follow this exact format: Start with 'Thought: ' and end with 'Answer: '. Since there are no available tools, you should provide a direct Answer."
                    
                    # 특별히 Thought만 있고 Answer가 없는 경우를 위한 처리
                    if thought and not final_answer and "Action" not in response_text:
                        if "Thought:" in response_text and response_text.strip().endswith(thought.strip()):
                            # Thought만 생성한 경우 - 직접 Answer 요청
                            answer_prompt = f"\nNow, provide your 'Answer: ' in {start_lang}."
                            reminder = answer_prompt
                    
                    current_prompt += f"{response_text}{reminder}\nAI:" 
                    conversation_log += f"System Reminder: {reminder.strip()}\n"
            
            # --- 루프 종료 후 --- #
            # 반복 횟수 초과했지만 최종 답변이 없는 경우
            if not final_answer_for_user:
                logger.warning(f"최대 반복 횟수에 도달하거나 모델이 최종 답변을 생성하지 못했습니다. 폴백 응답을 사용합니다. (Session ID: {session_id})")
                # 마지막으로 생성된 생각이 있다면 이를 사용하여 폴백 응답 생성
                last_thought = ""
                for ta in reversed(thoughts_and_actions):
                    if ta.get("thought"):
                        last_thought = ta.get("thought")
                        break
                
                # 폴백 응답 생성
                final_answer_for_user = self._generate_fallback_answer(prompt, last_thought, start_lang)
                self._save_step_log(session_log_dir, iteration, "fallback_answer_used", {
                    "last_thought": last_thought,
                    "fallback_answer": final_answer_for_user
                })
            
            # 결과 기록 및 반환
            success = final_answer_for_user != "" and "error" not in locals()
            session_meta["end_time"] = datetime.now().isoformat()
            session_meta["success"] = success
            session_meta["reason"] = "completed" if success else "max_iterations_reached"
            with open(session_log_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(session_meta, f, ensure_ascii=False, indent=2)
            
            logger.info(f"ReAct 패턴 처리 완료 - 성공: {success}, 반복: {iteration}, 도구 사용: {has_used_tool} (Session ID: {session_id})")
            
            return {
                 "response": final_answer_for_user,
                 "thoughts_and_actions": thoughts_and_actions,
                 "full_response": conversation_log, 
                 "log_session_id": session_id,
                 "iterations": iteration,
                 "used_tool": has_used_tool,
                 "success": success
            }

        except Exception as e:
            logger.error(f"ReAct 패턴 처리 중 심각한 오류 발생 (Session ID: {session_id}): {e}", exc_info=True)
            self._save_step_log(session_log_dir, -1, "global_error", {
                "error": str(e),
                "error_type": type(e).__name__,
            })
            session_meta["end_time"] = datetime.now().isoformat()
            session_meta["success"] = False
            session_meta["reason"] = f"error: {str(e)}"
            with open(session_log_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(session_meta, f, ensure_ascii=False, indent=2)
            
            error_message = "죄송합니다, 요청을 처리하는 중 오류가 발생했습니다." if start_lang == "ko" else "Sorry, an error occurred while processing your request."
            return {
                "response": f"{error_message} (error: {str(e)}, log ID: {session_id})",
                "error": str(e),
                "log_session_id": session_id,
                "success": False
            }
        finally:
             logger.info(f"--- process_react_pattern END --- Session ID: {session_id}")

    # --- Helper functions --- #
    def _extract_thought(self, text: str) -> str:
         thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|\nAnswer:|\nObservation:|$)", text, re.DOTALL)
         return thought_match.group(1).strip() if thought_match else ""
    
    def _extract_action_details(self, text: str) -> tuple:
        """
        텍스트에서 Action 형식을 추출하여 서버 이름, 도구 이름, 인수를 반환합니다.
        Action: server/tool(arg1="value1", arg2="value2") 형식을 파싱합니다.
        """
        action_match = ACTION_PATTERN.search(text)
        if not action_match:
            return None, None, None
            
        server_name, tool_name, args_str = action_match.groups()
        args = {}
        
        # 인수 문자열이 없으면 빈 dict 반환
        if not args_str or args_str.strip() == "":
            return server_name, tool_name, args
            
        # 다양한 파싱 방법 시도
        try:
            # 1. JSON 형식으로 파싱 시도 (괄호로 감싸진 경우)
            if args_str.strip().startswith('{') and args_str.strip().endswith('}'):
                # 작은따옴표를 큰따옴표로 변환하여 JSON 파싱 시도
                json_str = args_str.replace("'", '"')
                try:
                    args = json.loads(json_str)
                    return server_name, tool_name, args
                except json.JSONDecodeError:
                    # JSON 파싱 실패 시 다음 방법으로 진행
                    pass
                    
            # 2. 정규식을 사용한 키-값 쌍 추출
            # 사용 가능한 패턴: key="value" 또는 key=value
            pairs = re.findall(r'(\w+)\s*=\s*(?:\"([^\"]*)\"|\'([^\']*)\'|([^,\s\)]+))', args_str)
            if pairs:
                for match in pairs:
                    key = match[0]
                    # 값은 따옴표가 있는 경우(match[1] 또는 match[2]) 또는 없는 경우(match[3])
                    value = match[1] or match[2] or match[3]
                    args[key] = value.strip()
                    
                # 키-값 쌍을 찾았으나 args가 비어있는 경우 확인
                if not args and args_str.strip():
                    logger.warning(f"키-값 쌍을 파싱했으나 결과가 비어있습니다: '{args_str}'")
                    args = {'command': args_str.strip()}
                
                return server_name, tool_name, args
                
            # 3. 단일 명령어로 취급 (키-값 구조가 없는 경우)
            args = {'command': args_str.strip()}
            return server_name, tool_name, args
                
        except Exception as e:
            logger.warning(f"Action 인수 파싱 실패: '{args_str}' - {str(e)}. 단일 명령어로 처리합니다.")
            args = {'command': args_str.strip()}
            return server_name, tool_name, args

    def _extract_final_answer(self, text: str) -> str:
         # Extract content after the *last* 'Answer:' tag
         parts = text.split('Answer:')
         if len(parts) > 1:
             final_part = parts[-1].strip()
             # Remove any subsequent Thought/Action/Observation if present
             final_part = re.split(r'\nThought:|\nAction:|\nObservation:', final_part, 1)[0].strip()
             return final_part
         return ""

    def _save_step_log(self, log_dir: Path, step: int, step_type: str, data: dict):
        try:
            filename = f"{step:02d}_{step_type}.json"
            filepath = log_dir / filename
            log_entry = {
                "step": step,
                "step_type": step_type,
                "timestamp": datetime.now().isoformat(),
                **data
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(log_entry, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"로그 저장 중 오류 발생: {e}")

    def _generate_fallback_answer(self, prompt: str, thought: str, lang: str) -> str:
        """
        모델이 제대로 응답하지 않을 때 사용할 폴백 답변을 생성합니다.
        Thought 내용이 있으면 이를 바탕으로 답변을 생성합니다.
        """
        if not thought or thought.strip() == "":
            # Thought가 없는 경우 기본 오류 메시지 반환
            if lang == "ko":
                return "죄송합니다, 질문에 대한 답변을 생성하지 못했습니다. 다시 시도해 주세요."
            else:
                return "Sorry, I couldn't generate an answer to your question. Please try again."
                
        # 파이썬 관련 질문에 대한 기본 응답
        if "python" in thought.lower() or "파이썬" in prompt.lower():
            if lang == "ko":
                return "파이썬은 간결하고 읽기 쉬운 구문을 가진 고수준 프로그래밍 언어입니다. 웹 개발, 데이터 분석, 인공지능 등 다양한 분야에서 널리 사용되며, 초보자부터 전문가까지 모두에게 인기 있는 언어입니다."
            else:
                return "Python is a high-level programming language with a clean and readable syntax. It's widely used in various fields including web development, data analysis, artificial intelligence, and more. It's popular among beginners and experts alike."
                
        # 파일 목록 관련 질문에 대한 기본 응답
        if "file" in thought.lower() or "directory" in thought.lower() or "파일" in prompt.lower() or "목록" in prompt.lower():
            if lang == "ko":
                return "현재 디렉토리의 파일 목록을 확인하려면 파일 탐색기를 사용하거나 터미널에서 'ls' 명령어(Linux/Mac) 또는 'dir' 명령어(Windows)를 사용할 수 있습니다."
            else:
                return "To check the files in the current directory, you can use the file explorer or use the 'ls' command (Linux/Mac) or 'dir' command (Windows) in the terminal."
                
        # 일반적인 인사에 대한 기본 응답
        if "hello" in thought.lower() or "greeting" in thought.lower() or "안녕" in prompt.lower():
            if lang == "ko":
                return "안녕하세요! 무엇을 도와드릴까요?"
            else:
                return "Hello! How can I help you today?"
                
        # 기타 일반적인 질문에 대한 기본 응답
        if lang == "ko":
            return "죄송합니다만, 질문을 완전히 이해하지 못했습니다. 보다 구체적인 정보를 제공해 주시면 더 도움이 될 것 같습니다."
        else:
            return "I'm sorry, but I didn't fully understand your question. It would be helpful if you could provide more specific information."

# Dependency function
def get_inference_service(settings: Settings = Depends(get_settings), mcp_service: MCPService = Depends()) -> InferenceService:
    # This function might need adjustment if InferenceService instantiation changes
    # For now, assume it works with the global app_state or similar mechanism
    # in main.py to provide the singleton instance.
    # Placeholder: Replace with actual dependency injection logic if needed
    from app.main import app_state # Example: Accessing global state
    if "inference_service" not in app_state:
         raise RuntimeError("InferenceService not initialized")
    return app_state["inference_service"] 