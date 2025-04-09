import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import fcntl
import os
import asyncio

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# --- JSON 직렬화 헬퍼 ---
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

# 비동기 로그 저장 함수
async def async_save_meta_log(log_root_dir: Path, session_id: str, data: Dict[str, Any]):
    """
    비동기적으로 로그 이벤트를 저장합니다.
    파일 I/O 작업을 별도의 스레드에서 수행하여 메인 이벤트 루프를 차단하지 않습니다.
    """
    if not log_root_dir or not session_id:
        logger.warning("Log root directory or session ID not provided, skipping meta log saving.")
        return

    try:
        session_log_dir = log_root_dir / "api_logs" / session_id
        
        # 디렉토리 생성은 asyncio.to_thread로 래핑
        await asyncio.to_thread(lambda: session_log_dir.mkdir(parents=True, exist_ok=True))

        # 로그 파일 경로
        log_file_path = session_log_dir / "meta.json"

        # 파일 I/O 작업을 별도 스레드로 분리
        await asyncio.to_thread(
            _write_log_to_file, log_file_path, data
        )
        
        logger.debug(f"Asynchronously saved meta log ({data.get('event_type', 'unknown')}) to: {log_file_path}")
    
    except Exception as e:
        event_type = data.get('event_type', 'unknown')
        logger.error(f"Failed to process async meta log event '{event_type}' for session {session_id}: {e}", exc_info=True)

# 파일 I/O 작업을 수행하는 내부 함수 (별도 스레드에서 실행됨)
def _write_log_to_file(log_file_path: Path, data: Dict[str, Any]):
    """파일 잠금 및 로그 쓰기를 수행하는 내부 함수"""
    log_events: List[Dict[str, Any]] = []

    if log_file_path.exists():
        file_mode = "r+" # 읽고 쓰기
    else:
        file_mode = "w+" # 새로 쓰기

    try:
        with open(log_file_path, file_mode, encoding="utf-8") as f:
            try:
                # 파일 잠금
                fcntl.flock(f, fcntl.LOCK_EX)
                
                # 파일 내용 읽기 시도
                if file_mode == "r+":
                    content = f.read()
                    if content:
                        try:
                            log_events = json.loads(content)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to decode JSON from {log_file_path}. Starting new list.")
                            log_events = []
                    # 파일 처음으로 이동하여 덮어쓸 준비
                    f.seek(0)
                    f.truncate()

                # 새 로그 데이터 추가
                log_events.append(data)

                # 업데이트된 리스트를 파일에 쓰기
                json.dump(log_events, f, ensure_ascii=False, indent=2, default=default_serializer)

            finally:
                # 파일 잠금 해제
                fcntl.flock(f, fcntl.LOCK_UN)
                
    except IOError as e:
        logger.error(f"File I/O error accessing {log_file_path}: {e}", exc_info=True)
    except ImportError:
        logger.warning("fcntl module not available (likely non-Unix system). File locking disabled.")
        # --- Locking 없이 진행하는 코드 ---
        if log_file_path.exists():
            try:
                with open(log_file_path, "r", encoding="utf-8") as f_read:
                    log_events = json.load(f_read)
                    if not isinstance(log_events, list):
                        log_events = []
            except (json.JSONDecodeError, IOError):
                log_events = []
        log_events.append(data)
        try:
            with open(log_file_path, "w", encoding="utf-8") as f_write:
                json.dump(log_events, f_write, ensure_ascii=False, indent=2, default=default_serializer)
        except IOError as write_e:
            logger.error(f"File write error (no lock) for {log_file_path}: {write_e}", exc_info=True)