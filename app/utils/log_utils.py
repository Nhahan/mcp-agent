import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import fcntl
import os
import asyncio
from collections import OrderedDict

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# --- JSON 직렬화 헬퍼 ---
def default_serializer(obj):
    if isinstance(obj, BaseModel):
        # Pydantic 모델의 경우 .model_dump() 또는 .dict() 사용
        try:
            # Use exclude_none=True to avoid serializing None values unless explicitly set
            return obj.model_dump(mode='json', exclude_none=True)
        except AttributeError:
            try:
                # Pydantic v1 fallback
                return obj.dict(exclude_none=True)
            except Exception:
                 return f"<unserializable pydantic: {type(obj).__name__}>"
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
         return obj.isoformat()
    # 다른 직렬화 불가능한 타입 처리 (Set 등)
    if isinstance(obj, set):
        return list(obj) # Convert set to list for JSON
    try:
        # 먼저 직접 직렬화 시도 (기본 타입 등)
        json.dumps(obj)
        return obj
    except TypeError:
        try:
            # Try converting complex objects to string representation
            return str(obj)
        except Exception:
            # Final fallback for unserializable types
            return f"<unserializable: {type(obj).__name__}>"

# 비동기 로그 저장 함수
async def async_save_meta_log(session_log_dir: Path, session_id: str, event_data: Dict[str, Any]):
    """
    비동기적으로 로그 이벤트를 단일 세션 JSON 파일에 저장/업데이트합니다.
    파일 I/O 작업을 별도의 스레드에서 수행합니다.
    로그 레벨이 DEBUG 이하인 경우에만 저장합니다.
    """
    if not session_log_dir or not session_id:
        logger.warning("Session log directory or session ID not provided, skipping meta log saving.")
        return

    # 로그 레벨 체크는 그대로 유지
    root_logger = logging.getLogger()
    if root_logger.level > logging.DEBUG:
        # Optionally log that we are skipping due to level
        # logger.debug(f"Skipping meta log saving for session {session_id} due to log level.")
        return

    try:
        # 디렉토리 생성 (이제 session_log_dir 자체가 대상)
        # exist_ok=True ensures no error if directory already exists
        await asyncio.to_thread(lambda: session_log_dir.mkdir(parents=True, exist_ok=True))

        # 로그 파일 경로 (이제 session_log_dir 바로 아래)
        log_file_path = session_log_dir / "meta.json"

        # 파일 I/O 작업을 별도 스레드로 분리
        await asyncio.to_thread(
            _update_session_log_file, log_file_path, session_id, event_data
        )

        logger.debug(f"Asynchronously updated meta log ({event_data.get('event_type', 'unknown')}) in: {log_file_path}")

    except Exception as e:
        event_type = event_data.get('event_type', 'unknown')
        logger.error(f"Failed to process async meta log event '{event_type}' for session {session_id}: {e}", exc_info=True)


# 단일 세션 로그 파일 업데이트 함수 (read-modify-write)
def _update_session_log_file(log_file_path: Path, session_id: str, event_data: Dict[str, Any]):
    """파일 잠금 및 단일 JSON 객체 업데이트를 수행 (read-modify-write)"""
    try:
        # Use 'a+' for creation, then 'r+' for read/write after locking
        # Open with 'a+' first to ensure the file exists
        with open(log_file_path, "a", encoding="utf-8") as f_create:
            pass # Just ensure file exists

        # Now open with 'r+' for read/write operations
        with open(log_file_path, "r+", encoding="utf-8") as f:
            try:
                # 파일 잠금 (Exclusive lock)
                fcntl.flock(f, fcntl.LOCK_EX)

                # 파일 내용 읽기 시도
                content = f.read()
                session_data = OrderedDict() # Use OrderedDict to maintain insertion order

                if content:
                    try:
                        # Preserve order when loading if possible (though standard dict is ordered in Python 3.7+)
                        session_data = json.loads(content, object_pairs_hook=OrderedDict)
                        if not isinstance(session_data, (dict, OrderedDict)):
                            logger.warning(f"Log file {log_file_path} contained non-object data. Re-initializing.")
                            session_data = OrderedDict()
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode JSON from {log_file_path}. Re-initializing.")
                        session_data = OrderedDict()
                else:
                    # Initialize basic structure for new file
                    session_data['session_id'] = session_id
                    session_data['start_time'] = datetime.now().isoformat()
                    session_data['iterations'] = []
                    session_data['errors'] = []

                # --- Update session_data based on event_data ---
                event_type = event_data.get("event_type")

                # --- Explicitly ignore 'react_process' event type --- 
                if event_type == "react_process":
                    logger.warning(f"Ignoring deprecated event_type 'react_process' for session {session_id}")
                    # To prevent writing, we should ideally return here or ensure session_data isn't modified.
                    # Let's simply skip modifications for this event.
                    pass # Modifications skipped, file will be re-written with existing session_data
                else:
                    # --- Remove model info from non-api_request events --- 
                    # This is a safeguard against incorrect data being passed.
                    if event_type != "api_request":
                        if "model_info" in event_data:
                            del event_data["model_info"]
                            
                    # Ensure basic fields are present if loading existing data
                    if 'session_id' not in session_data: session_data['session_id'] = session_id
                    if 'iterations' not in session_data: session_data['iterations'] = []
                    if 'errors' not in session_data: session_data['errors'] = []
                    if 'start_time' not in session_data: session_data['start_time'] = datetime.now().isoformat() # Fallback start time


                    if event_type == "api_request":
                        # Store initial request details, add model info if available
                        session_data['initial_request'] = event_data.get("request_body")
                        session_data['language_detected'] = event_data.get("language_detected")
                        if "model_info" in event_data:
                             session_data['model_info'] = event_data["model_info"]
                        # Move timestamp to top level if not already set by start_time
                        if 'timestamp' in event_data and 'request_timestamp' not in session_data:
                             session_data['request_timestamp'] = event_data['timestamp']


                    elif event_type == "react_iteration":
                        # Append iteration data
                        iteration_content = event_data.get("iteration_data", {})
                        if iteration_content: # Ensure there's data to append
                            session_data['iterations'].append(iteration_content)

                    elif event_type == "tool_error" or event_data.get("error"):
                        # Log errors encountered
                        error_details = {
                            "timestamp": event_data.get("timestamp", datetime.now().isoformat()),
                            "event_type": event_type,
                            "details": event_data.get("error_details") or event_data.get("error")
                        }
                        # Add related iteration number if available
                        if "iteration_number" in event_data:
                            error_details["iteration"] = event_data["iteration_number"]
                        session_data['errors'].append(error_details)


                    elif event_type == "api_response":
                         # Store final response details
                         session_data['final_response'] = event_data.get("response_body")
                         session_data['end_time'] = event_data.get("timestamp", datetime.now().isoformat())
                         # If iterations were logged directly in response_body, ensure they are merged/handled correctly
                         # (Current design expects iterations via react_iteration events)
                         if "thoughts_and_actions" in event_data.get("response_body", {}):
                             # Potentially reconcile/add these if not logged via react_iteration
                             # For now, assume react_iteration covers this. If not, logic needed here.
                             pass


                    else:
                        # Generic update for other event types? Or log as unhandled?
                        # For now, let's store it under a generic key if type is unknown
                        if event_type:
                            if event_type not in session_data: session_data[event_type] = []
                            # Avoid adding redundant session_id etc. if already top-level
                            data_to_add = {k: v for k, v in event_data.items() if k not in ['session_id', 'event_type']}
                            if isinstance(session_data[event_type], list):
                                 session_data[event_type].append(data_to_add)
                            else: # If not a list, maybe overwrite or log error? Overwriting for simplicity now.
                                 session_data[event_type] = data_to_add
                        else:
                             logger.warning(f"Log event received without event_type for session {session_id}")
                             # Add to a specific 'unknown_events' list?
                             if 'unknown_events' not in session_data: session_data['unknown_events'] = []
                             session_data['unknown_events'].append(event_data)


                # 파일 처음으로 이동하여 덮어쓸 준비
                f.seek(0)
                f.truncate()

                # 업데이트된 단일 객체를 파일에 쓰기
                # Use indent for readability
                json.dump(session_data, f, ensure_ascii=False, indent=2, default=default_serializer)

            finally:
                # 파일 잠금 해제
                fcntl.flock(f, fcntl.LOCK_UN)

    except IOError as e:
        logger.error(f"File I/O error accessing {log_file_path}: {e}", exc_info=True)
    except ImportError:
        # Correctly format the warning string using str(log_file_path)
        logger.warning(f"fcntl module not available on this system (likely non-Unix). File locking disabled for {str(log_file_path)}. "
                       f"Concurrent writes may lead to corrupted log files.")
        # --- Non-locking fallback (RISKY for read-modify-write) ---
        # This fallback is inherently unsafe for read-modify-write patterns.
        # It's highly likely to cause data loss or corruption under concurrency.
        # A better non-locking strategy might involve writing events to separate small files
        # and consolidating them later, but that's much more complex.
        # Providing a simple, but risky, fallback for compatibility.
        session_data = OrderedDict()
        try:
            if log_file_path.exists():
                 with open(log_file_path, "r", encoding="utf-8") as f_read:
                     content = f_read.read()
                     if content:
                         try:
                             session_data = json.loads(content, object_pairs_hook=OrderedDict)
                             if not isinstance(session_data, (dict, OrderedDict)): session_data = OrderedDict()
                         except json.JSONDecodeError:
                             session_data = OrderedDict() # Start fresh if corrupt

            # --- Apply updates (Duplicated logic - consider refactor) ---
            event_type = event_data.get("event_type")

            # --- Explicitly ignore 'react_process' event type (Fallback) --- 
            if event_type == "react_process":
                logger.warning(f"Ignoring deprecated event_type 'react_process' for session {session_id} (non-locking fallback)")
                # Skip update logic, just rewrite existing data if any
                pass 
            else:
                # --- Remove model info from non-api_request events (Fallback) --- 
                if event_type != "api_request":
                    if "model" in event_data:
                        logger.warning(f"Removing unexpected 'model' key from event '{event_type}' (non-locking fallback)." )
                        del event_data["model"]
                    if "model_info" in event_data:
                        logger.warning(f"Removing unexpected 'model_info' key from event '{event_type}' (non-locking fallback)." )
                        del event_data["model_info"]
                        
                if 'session_id' not in session_data: session_data['session_id'] = session_id
                if 'iterations' not in session_data: session_data['iterations'] = []
                if 'errors' not in session_data: session_data['errors'] = []
                if 'start_time' not in session_data: session_data['start_time'] = datetime.now().isoformat()

                if event_type == "api_request":
                    session_data['initial_request'] = event_data.get("request_body")
                    session_data['language_detected'] = event_data.get("language_detected")
                    if "model_info" in event_data: session_data['model_info'] = event_data["model_info"]
                    if 'timestamp' in event_data and 'request_timestamp' not in session_data: session_data['request_timestamp'] = event_data['timestamp']
                elif event_type == "react_iteration":
                    iteration_content = event_data.get("iteration_data", {})
                    if iteration_content: session_data['iterations'].append(iteration_content)
                elif event_type == "tool_error" or event_data.get("error"):
                     error_details = {
                        "timestamp": event_data.get("timestamp", datetime.now().isoformat()),
                        "event_type": event_type,
                        "details": event_data.get("error_details") or event_data.get("error")
                     }
                     if "iteration_number" in event_data: error_details["iteration"] = event_data["iteration_number"]
                     session_data['errors'].append(error_details)
                elif event_type == "api_response":
                     session_data['final_response'] = event_data.get("response_body")
                     session_data['end_time'] = event_data.get("timestamp", datetime.now().isoformat())
                else:
                    # Handle unknown event types similarly
                    if event_type:
                         if event_type not in session_data: session_data[event_type] = []
                         data_to_add = {k: v for k, v in event_data.items() if k not in ['session_id', 'event_type']}
                         if isinstance(session_data[event_type], list):
                              session_data[event_type].append(data_to_add)
                         else:
                              session_data[event_type] = data_to_add
                    else:
                        if 'unknown_events' not in session_data: session_data['unknown_events'] = []
                        session_data['unknown_events'].append(event_data)
            # --- End of duplicated update logic ---

            # Overwrite the file (unsafe without lock)
            with open(log_file_path, "w", encoding="utf-8") as f_write:
                json.dump(session_data, f_write, ensure_ascii=False, indent=2, default=default_serializer)

        except IOError as write_e:
            logger.error(f"File write error (no lock) for {log_file_path}: {write_e}", exc_info=True)
        except Exception as e:
             logger.error(f"Unexpected error during non-locking log update for {log_file_path}: {e}", exc_info=True)

# Optional: Helper function to get session log directory path consistently
def get_session_log_directory(base_log_dir: Path, session_id: str) -> Path:
    """Constructs the standard path for a session's log directory."""
    return base_log_dir / "api_logs" / session_id