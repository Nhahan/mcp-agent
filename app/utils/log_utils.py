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

# Meta JSON Schema (로그 파일 스키마)
"""
{
  "type": "object",
  "properties": {
    "session_id": { "type": "string" },
    "start_time": { "type": "string", "format": "date-time" },
    "initial_request": { "type": "string" },
    "language_detected": { "type": "string" },
    "model_info": { 
      "type": "object", 
      "properties": {
        "path": { "type": "string" },
        "parameters": { "type": "object" }
      }
    },
    "iterations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "iteration": { "type": "integer" },
          "timestamp": { "type": "string", "format": "date-time" },
          "prompt": { "type": "string" },
          "response": { "type": "string" },
          "action": { "type": "object" },
          "observation": { "type": ["string", "object", "null"] },
          "error": { "type": ["string", "null"] }
        }
      }
    },
    "errors": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "timestamp": { "type": "string", "format": "date-time" },
          "event_type": { "type": "string" },
          "details": { "type": ["string", "object"] },
          "iteration": { "type": "integer" }
        }
      }
    },
    "request_timestamp": { "type": "string", "format": "date-time" },
    "end_time": { "type": "string", "format": "date-time" },
    "final_response": { "type": ["string", "object"] }
  },
  "required": ["session_id", "start_time"]
}
"""

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
async def async_save_meta_log(session_log_dir: Path, event_data: Dict[str, Any], session_id: str, merge: bool = False):
    """
    비동기적으로 로그 이벤트를 단일 세션 JSON 파일에 저장/업데이트합니다.
    파일 I/O 작업을 별도의 스레드에서 수행합니다.
    로그 레벨이 DEBUG 이하인 경우에만 저장합니다.
    
    Args:
        session_log_dir: 로그 디렉토리 경로
        event_data: 저장할 이벤트 데이터 (딕셔너리)
        session_id: 세션 ID
        merge: 기존 파일과 병합 여부 (기본값: False)
    """
    root_logger = logging.getLogger()
    if root_logger.level > logging.DEBUG:
        # Optionally log that we are skipping due to level
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

        # 이벤트 타입을 안전하게 가져오기 (event_data가 딕셔너리인지 확인)
        event_type = "unknown"
        if isinstance(event_data, dict):
            event_type = event_data.get('event_type', 'unknown')

    except Exception as e:
        # 이벤트 타입을 안전하게 가져오기 (event_data가 딕셔너리인지 확인)
        event_type = "unknown"
        if isinstance(event_data, dict):
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
                    # Initialize basic structure for new file with specified field order
                    session_data = OrderedDict()
                    session_data['session_id'] = session_id
                    session_data['start_time'] = datetime.now().isoformat()
                    session_data['request_timestamp'] = '' # Initialize explicitly
                    session_data['initial_request'] = ""
                    session_data['language_detected'] = ""
                    session_data['model_info'] = {}
                    session_data['iterations'] = []
                    session_data['errors'] = []
                    session_data['final_response'] = None # Initialize explicitly
                    session_data['end_time'] = None # Initialize explicitly
                    
                event_type = event_data.get("event_type") if isinstance(event_data, dict) else None

                # --- Remove model info from non-api_request events --- 
                if event_type != "api_request" and isinstance(event_data, dict):
                    if "model_info" in event_data:
                        del event_data["model_info"]
                        
                # Ensure basic fields are present if loading existing data
                if 'session_id' not in session_data: session_data['session_id'] = session_id
                if 'start_time' not in session_data: session_data['start_time'] = datetime.now().isoformat() # Fallback start time
                if 'initial_request' not in session_data: session_data['initial_request'] = ""
                if 'language_detected' not in session_data: session_data['language_detected'] = ""
                if 'model_info' not in session_data: session_data['model_info'] = {}
                if 'iterations' not in session_data: session_data['iterations'] = []
                if 'errors' not in session_data: session_data['errors'] = []

                if event_type == "api_request":
                    # Store initial request details directly from event_data
                    # Only set if not already present
                    if 'initial_request' not in session_data or not session_data['initial_request']:
                        session_data['initial_request'] = event_data.get('initial_request')
                    
                    # Handle language detection - update if provided
                    if "language_detected" in event_data:
                        session_data['language_detected'] = event_data["language_detected"]
                    
                    # Handle model info - update if provided
                    if "model_info" in event_data:
                        session_data['model_info'] = event_data["model_info"]
                    
                    # Use timestamp from event_data for request_timestamp if available and not set
                    if 'request_timestamp' in event_data and ('request_timestamp' not in session_data or not session_data['request_timestamp']):
                        session_data['request_timestamp'] = event_data['request_timestamp']
                    elif 'request_timestamp' not in session_data: # Ensure it exists
                        session_data['request_timestamp'] = datetime.now().isoformat() # Fallback

                elif event_type == "react_iteration":
                    # Append iteration data - ADAPTED FOR is_final FORMAT
                    iteration_content_raw = event_data.get("iteration_data", {})
                    if iteration_content_raw and isinstance(iteration_content_raw, dict):
                        # Create a new dict to store cleaned/structured iteration data
                        current_iteration_log = OrderedDict()
                        current_iteration_log['iteration'] = iteration_content_raw.get('iteration')
                        current_iteration_log['timestamp'] = iteration_content_raw.get('timestamp', datetime.now().isoformat())
                        current_iteration_log['llm_prompt_length'] = iteration_content_raw.get('llm_prompt_length')
                        current_iteration_log['llm_response'] = iteration_content_raw.get('llm_response') # Raw LLM response
                        
                        # Extract action details based on is_final from parsed_action
                        parsed_action = iteration_content_raw.get('parsed_action', {})
                        current_iteration_log['action'] = OrderedDict() # Initialize action sub-object
                        if isinstance(parsed_action, dict):
                            is_final = parsed_action.get('is_final') # Could be True, False, or None (on error)
                            current_iteration_log['action']['is_final'] = is_final
                            current_iteration_log['action']['thought'] = parsed_action.get('thought')
                            
                            if is_final is True:
                                current_iteration_log['action']['answer'] = parsed_action.get('answer')
                            elif is_final is False:
                                current_iteration_log['action']['tool_name'] = parsed_action.get('tool_name')
                                current_iteration_log['action']['arguments'] = parsed_action.get('arguments')
                            # If is_final is None, only log thought and is_final=None
                                
                        current_iteration_log['observation'] = iteration_content_raw.get('observation') # 올바른 키 'observation' 사용
                        current_iteration_log['error'] = iteration_content_raw.get('error') # Log iteration-specific errors
                        
                        session_data['iterations'].append(current_iteration_log)
                    else:
                         logger.warning(f"Received react_iteration event with invalid data: {iteration_content_raw}")

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
                     # Store final response details - always update these
                     session_data['final_response'] = event_data.get("response_body")
                     session_data['end_time'] = event_data.get("timestamp", datetime.now().isoformat())
                     
                     # Merge iterations if provided in the event (e.g., from final_meta)
                     if "iterations" in event_data and isinstance(event_data["iterations"], list):
                         # Avoid duplicating entire list if already present? For now, simple overwrite/append
                         # If merging is complex, ensure `inference_service` sends the complete final list.
                         session_data['iterations'] = event_data["iterations"]
                         
                     # Merge errors if provided in the event
                     if "errors" in event_data and isinstance(event_data["errors"], list):
                         if 'errors' not in session_data: session_data['errors'] = []
                         # Avoid duplicate errors? Could check hashes or specific content if needed.
                         # For now, append new errors from the event.
                         existing_errors_set = set(json.dumps(e, sort_keys=True) for e in session_data.get('errors', []))
                         for error_item in event_data["errors"]:
                             error_item_str = json.dumps(error_item, sort_keys=True)
                             if error_item_str not in existing_errors_set:
                                 session_data['errors'].append(error_item)
                                 existing_errors_set.add(error_item_str)
                                 
                     # Remove obsolete check for thoughts_and_actions
                     # if "thoughts_and_actions" in event_data.get("response_body", {}):
                     #     pass

                else:
                    # 이벤트 타입이 없는 경우: 경고 로깅 후 데이터 통합 시도하지 않음
                    # 이전의 복잡한 병합 로직은 데이터 손상 위험이 있어 제거.
                    if not event_type:
                         logger.warning(f"Received event data without 'event_type' for session {session_id}. Skipping merging of unknown data: {str(event_data)[:200]}...")
                    else:
                         logger.debug(f"Received unhandled event_type '{event_type}' for session {session_id}. Data: {str(event_data)[:200]}...")
                    # 기본 필드들에 매핑하여 병합 (중복 방지) - REMOVED
                    # unknown_data 처리 로직 - REMOVED

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
                # Fallback: Store initial request, language, model info if available
                if 'initial_request' not in session_data or not session_data['initial_request']:
                    session_data['initial_request'] = event_data.get('initial_request')
                session_data['language_detected'] = event_data.get("language_detected")
                if "model_info" in event_data: session_data['model_info'] = event_data["model_info"]
                if 'request_timestamp' in event_data and ('request_timestamp' not in session_data or not session_data['request_timestamp']):
                    session_data['request_timestamp'] = event_data['request_timestamp']
                elif 'request_timestamp' not in session_data: session_data['request_timestamp'] = datetime.now().isoformat()
            elif event_type == "react_iteration":
                iteration_content_raw = event_data.get("iteration_data", {})
                if iteration_content_raw:
                    # Create a new dict to store cleaned/structured iteration data
                    current_iteration_log = OrderedDict()
                    current_iteration_log['iteration'] = iteration_content_raw.get('iteration')
                    current_iteration_log['timestamp'] = iteration_content_raw.get('timestamp', datetime.now().isoformat())
                    current_iteration_log['llm_prompt_length'] = iteration_content_raw.get('llm_prompt_length')
                    current_iteration_log['llm_response'] = iteration_content_raw.get('llm_response') # Raw LLM response
                    
                    # Extract action details based on is_final from parsed_action
                    parsed_action = iteration_content_raw.get('parsed_action', {})
                    current_iteration_log['action'] = OrderedDict() # Initialize action sub-object
                    if isinstance(parsed_action, dict):
                        is_final = parsed_action.get('is_final') # Could be True, False, or None (on error)
                        current_iteration_log['action']['is_final'] = is_final
                        current_iteration_log['action']['thought'] = parsed_action.get('thought')
                        
                        if is_final is True:
                            current_iteration_log['action']['answer'] = parsed_action.get('answer')
                        elif is_final is False:
                            current_iteration_log['action']['tool_name'] = parsed_action.get('tool_name')
                            current_iteration_log['action']['arguments'] = parsed_action.get('arguments')
                        # If is_final is None, only log thought and is_final=None
                            
                    current_iteration_log['observation'] = iteration_content_raw.get('observation') # 올바른 키 'observation' 사용
                    current_iteration_log['error'] = iteration_content_raw.get('error') # Log iteration-specific errors
                    
                    session_data['iterations'].append(current_iteration_log)
                else:
                    logger.warning(f"Received react_iteration event with invalid data: {iteration_content_raw}")
            elif event_type == "tool_error" or event_data.get("error"):
                 error_details = {
                    "timestamp": event_data.get("timestamp", datetime.now().isoformat()),
                    "event_type": event_type,
                    "details": event_data.get("error_details") or event_data.get("error")
                 }
                 if "iteration_number" in event_data: error_details["iteration"] = event_data["iteration_number"]
                 session_data['errors'].append(error_details)
            elif event_type == "api_response":
                 # Fallback: Update final response and end time
                 session_data['final_response'] = event_data.get("response_body")
                 session_data['end_time'] = event_data.get("timestamp", datetime.now().isoformat())
                 # Fallback: Overwrite iterations and errors if present (less safe than locked version)
                 if "iterations" in event_data and isinstance(event_data["iterations"], list):
                     session_data["iterations"] = event_data["iterations"]
                 if "errors" in event_data and isinstance(event_data["errors"], list):
                     session_data["errors"] = event_data["errors"]
            else:
                # 이벤트 타입이 없는 경우 (Fallback): 경고만 로깅하고 병합 시도 안 함
                if not event_type:
                    logger.warning(f"(Fallback/No Lock) Received event data without 'event_type' for session {session_id}. Skipping merging.")
                else:
                    logger.debug(f"(Fallback/No Lock) Received unhandled event_type '{event_type}' for session {session_id}.")
                # 기본 필드들에 매핑하여 병합 (중복 방지) - REMOVED FROM FALLBACK
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