#!/usr/bin/env python3
# test_chat_api.py - 채팅 API 테스트 스크립트
#
# 실행 방법:
# 1. 먼저 API 서버 실행:
#    python -m app.main
#
# 2. 다른 터미널에서 테스트 스크립트 실행:
#    python test_scripts/test_chat_api.py
#
# 옵션:
#    --host HOST     : API 서버 호스트 (기본값: 127.0.0.1)
#    --port PORT     : API 서버 포트 (기본값: 8000)
#    --prompt 텍스트  : 사용자 지정 프롬프트
#
# 예시:
#    python test_scripts/test_chat_api.py --prompt "현재 날짜와 시간을 알려주세요"
#    python test_scripts/test_chat_api.py --port 8001
#
import os
import sys
import argparse
import requests
import json
import time
from pathlib import Path

# 경로 설정
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOG_DIR = PROJECT_ROOT / "logs" / "chat_tests"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 서버 설정
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
API_TIMEOUT = 120  # 초

def save_conversation_log(test_name, prompt, response):
    """대화 내용을 로그 파일에 저장합니다."""
    log_file = LOG_DIR / f"{test_name}_{int(time.time())}.json"
    
    log_data = {
        "test_name": test_name,
        "prompt": prompt,
        "response": response,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    print(f"대화 로그가 저장되었습니다: {log_file}")
    return log_file

def test_chat_endpoint(host, port, prompt):
    """채팅 엔드포인트를 테스트합니다."""
    print(f"\n=== 채팅 테스트 ===")
    print(f"프롬프트: {prompt}")
    
    url = f"http://{host}:{port}/api/v1/chat"
    payload = {"text": prompt}
    
    try:
        response = requests.post(url, json=payload, timeout=API_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        
        print(f"\n최종 응답:")
        print(f"{result.get('response', '')}")
        
        # 사고 및 행동 기록 출력
        thoughts = result.get('thoughts_and_actions', [])
        if thoughts:
            print(f"\n총 {len(thoughts)}개의 사고/행동 기록:")
            for i, thought in enumerate(thoughts):
                print(f"\n[{i+1}] 생각: {thought.get('thought', '')}")
                action = thought.get('action', {})
                action_str = json.dumps(action, ensure_ascii=False, indent=2) if isinstance(action, dict) else str(action)
                print(f"[{i+1}] 행동: {action_str}")
                print(f"[{i+1}] 관찰: {thought.get('observation', '')[:100]}...")
        
        # 로그 정보 출력
        log_session_id = result.get('log_session_id')
        log_path = result.get('log_path')
        
        if log_session_id:
            print(f"\n상세 로그 정보:")
            print(f"로그 세션 ID: {log_session_id}")
            if log_path:
                print(f"로그 디렉토리: {log_path}")
                # 절대 경로로 변환
                abs_log_path = str(PROJECT_ROOT / log_path)
                print(f"로그 위치(절대): {abs_log_path}")
        
        # 로그 저장
        log_file = save_conversation_log("chat", prompt, result)
        
        return True, result
    
    except Exception as e:
        print(f"오류 발생: {e}")
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description="채팅 API 테스트 스크립트")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"API 서버 호스트 (기본값: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"API 서버 포트 (기본값: {DEFAULT_PORT})")
    parser.add_argument("--prompt", help="사용자 지정 프롬프트")
    
    args = parser.parse_args()
    
    # 기본 프롬프트
    chat_prompt = args.prompt or "현재 디렉토리의 파일 목록을 보여주세요."
    
    # 테스트 실행
    success, result = test_chat_endpoint(args.host, args.port, chat_prompt)
    
    # 결과 요약
    print("\n=== 테스트 결과 요약 ===")
    status = "성공" if success else "실패"
    print(f"채팅 테스트: {status}")

if __name__ == "__main__":
    main() 