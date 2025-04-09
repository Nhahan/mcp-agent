import json
import logging
# import uuid # UUID 사용 안 함
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List # List 추가
import fcntl # 파일 잠금 위해 추가 (Unix-like)
import os # 파일 잠금 위해 추가

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# --- JSON 직렬화 헬퍼 (유지) ---
def default_serializer(obj):
    if isinstance(obj, BaseModel):
        # Pydantic 모델의 경우 .model_dump() 또는 .dict() 사용
        try:
            return obj.model_dump(mode='json') # Pydantic v2, JSON 호환 타입으로 직렬화
        except AttributeError:
            try:
                return obj.dict() # Pydantic v1
            except Exception:
                 return f"<unserializable pydantic: {type(obj).__name__}>"
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
         return obj.isoformat()
    # 다른 직렬화 불가능한 타입 처리
    try:
        # 먼저 직접 직렬화 시도 (기본 타입 등)
        json.dumps(obj)
        return obj
    except TypeError:
        try:
            return str(obj) # 실패 시 문자열 변환 시도
        except Exception:
            return f"<unserializable: {type(obj).__name__}>"

# --- 통합 로그 저장 함수 (Structured JSON) ---
def save_meta_log(log_root_dir: Path, session_id: str, data: Dict[str, Any]):
    """
    Saves a log event into a structured JSON file (meta.json) for the session.
    Reads the existing list, appends the new event, and writes back.
    Includes basic file locking for safety.
    """
    if not log_root_dir or not session_id:
        logger.warning("Log root directory or session ID not provided, skipping meta log saving.")
        return

    try:
        session_log_dir = log_root_dir / "api_logs" / session_id
        session_log_dir.mkdir(parents=True, exist_ok=True)

        # 로그 파일 경로 (.json 사용)
        log_file_path = session_log_dir / "meta.json"

        log_events: List[Dict[str, Any]] = []

        # --- File Locking --- 
        # 파일을 열고 잠금을 시도합니다.
        # "a+" 모드는 파일이 없으면 생성하고, 읽기/쓰기(추가)를 가능하게 합니다.
        # 여기서는 읽고 덮어쓰므로 "r+" 또는 "w" 후 "r" 조합이 더 적합할 수 있으나,
        # 생성과 잠금을 동시에 처리하기 위해 "a+" 후 seek(0) 사용 시도.
        # 더 안전한 방법: 파일 존재 확인 후 "r+" 또는 "w" 사용

        if log_file_path.exists():
            file_mode = "r+" # 읽고 쓰기 (덮어쓰기 위해)
        else:
            file_mode = "w+" # 새로 쓰기 (파일 생성)

        try:
            with open(log_file_path, file_mode, encoding="utf-8") as f:
                # 파일 잠금 ( flock 사용, Unix-like 시스템에서 동작 )
                # Windows에서는 다른 메커니즘 필요 (e.g., msvcrt or pywin32)
                try:
                    # Blocking exclusive lock
                    fcntl.flock(f, fcntl.LOCK_EX)
                    logger.debug(f"Acquired lock for {log_file_path}")

                    # 파일 내용 읽기 시도
                    if file_mode == "r+": # 읽기 모드로 열었을 경우
                        content = f.read()
                        if content:
                            try:
                                log_events = json.loads(content)
                                if not isinstance(log_events, list):
                                    logger.warning(f"Existing log file {log_file_path} is not a JSON list. Starting new list.")
                                    log_events = []
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to decode JSON from {log_file_path}. Starting new list.")
                                log_events = []
                        # 파일 처음으로 이동하여 덮어쓸 준비
                        f.seek(0)
                        f.truncate() # 기존 내용 삭제 (덮어쓰기 위해)
                    # else (w+ 모드): 파일이 비어 있으므로 log_events는 빈 리스트 유지

                    # 새 로그 데이터 추가
                    log_events.append(data)

                    # 업데이트된 리스트를 파일에 쓰기
                    json.dump(log_events, f, ensure_ascii=False, indent=2, default=default_serializer)

                finally:
                    # 파일 잠금 해제
                    fcntl.flock(f, fcntl.LOCK_UN)
                    logger.debug(f"Released lock for {log_file_path}")

            logger.debug(f"Saved/updated meta log ({data.get('event_type', 'unknown')}) to: {log_file_path}")

        except IOError as e:
             logger.error(f"File I/O error accessing {log_file_path}: {e}", exc_info=True)
        except ImportError:
             logger.warning("fcntl module not available (likely non-Unix system). File locking disabled.")
             # --- Locking 없이 진행 (간단한 버전) ---
             if log_file_path.exists():
                 try:
                     with open(log_file_path, "r", encoding="utf-8") as f_read:
                         log_events = json.load(f_read)
                         if not isinstance(log_events, list):
                             log_events = []
                 except (json.JSONDecodeError, IOError):
                     log_events = [] # 파일 읽기 실패 시 빈 리스트
             log_events.append(data)
             try:
                 with open(log_file_path, "w", encoding="utf-8") as f_write:
                     json.dump(log_events, f_write, ensure_ascii=False, indent=2, default=default_serializer)
                 logger.debug(f"Saved/updated meta log (no lock) ({data.get('event_type', 'unknown')}) to: {log_file_path}")
             except IOError as write_e:
                  logger.error(f"File write error (no lock) for {log_file_path}: {write_e}", exc_info=True)
             # --- Locking 없는 버전 끝 ---

    except Exception as e:
        event_type = data.get('event_type', 'unknown')
        logger.error(f"Failed to process meta log event '{event_type}' for session {session_id}: {e}", exc_info=True) 