#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
import requests
from pathlib import Path
from datetime import datetime

# 루트 디렉토리 설정
ROOT_DIR = Path(__file__).parent.parent

def setup_logging():
    """로깅 디렉토리 설정"""
    log_dir = ROOT_DIR / "logs" / "e2e_tests"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

def call_chat_api(prompt, host="localhost", port=8000):
    """Chat API 호출"""
    url = f"http://{host}:{port}/api/v1/chat"
    payload = {
        "text": prompt
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API 호출 오류: {e}")
        return {"error": str(e)}

def save_test_result(log_dir, test_name, prompt, response):
    """테스트 결과 저장"""
    timestamp = int(time.time())
    log_path = log_dir / f"test_{test_name}_{timestamp}.json"
    
    result = {
        "test_name": test_name,
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "response": response
    }
    
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return log_path

def run_test_greeting(log_dir, host="localhost", port=8000):
    """인사 테스트 - 기본 인사 응답"""
    print("\n=== 테스트: 인사 ===")
    prompt = "안녕하세요. 반갑습니다."
    print(f"프롬프트: {prompt}")
    
    response = call_chat_api(prompt, host, port)
    log_path = save_test_result(log_dir, "greeting", prompt, response)
    
    success = True
    if "error" in response and response["error"]:
        print(f"오류: {response['error']}")
        success = False
    else:
        # 응답 확인 - API 응답 구조에 맞게 수정
        final_response = response.get("response", "")
        print(f"응답: {final_response}")
        
        # 단순히 응답이 있는지만 확인
        if not final_response or len(final_response.strip()) < 2:
            print("오류: 응답이 비어있거나 너무 짧습니다.")
            success = False
    
    print(f"로그 저장 위치: {log_path}")
    print(f"테스트 결과: {'성공' if success else '실패'}")
    return success

def run_test_korean(log_dir, host="localhost", port=8000):
    """한국어 테스트 - 한국어 응답"""
    print("\n=== 테스트: 한국어 ===")
    prompt = "한국어로 인공지능에 대해 간단하게 설명해주세요."
    print(f"프롬프트: {prompt}")
    
    response = call_chat_api(prompt, host, port)
    log_path = save_test_result(log_dir, "korean", prompt, response)
    
    success = True
    if "error" in response and response["error"]:
        print(f"오류: {response['error']}")
        success = False
    else:
        # 응답 확인 - API 응답 구조에 맞게 수정
        final_response = response.get("response", "")
        print(f"응답: {final_response}")
        
        # 한국어가 포함되어 있는지 확인
        korean_chars = set("가나다라마바사아자차카타파하")
        if not any(char in final_response for char in korean_chars):
            print("오류: 응답에 한국어가 포함되어 있지 않습니다.")
            success = False
    
    print(f"로그 저장 위치: {log_path}")
    print(f"테스트 결과: {'성공' if success else '실패'}")
    return success

def run_test_tool_use(log_dir, host="localhost", port=8000):
    """도구 사용 테스트 - 시간 확인"""
    print("\n=== 테스트: 도구 사용 ===")
    prompt = "현재 시간이 몇 시인지 알려주세요."
    print(f"프롬프트: {prompt}")
    
    response = call_chat_api(prompt, host, port)
    log_path = save_test_result(log_dir, "tool_use", prompt, response)
    
    success = True
    if "error" in response and response["error"]:
        print(f"오류: {response['error']}")
        success = False
    else:
        # 응답 확인 - API 응답 구조에 맞게 수정
        final_response = response.get("response", "")
        print(f"응답: {final_response}")
        
        # 시간 관련 단어 포함 확인
        time_keywords = ["time", "o'clock", "hour", "minute", "시간", "시", "분", "초"]
        if not any(keyword in final_response.lower() for keyword in time_keywords):
            print("오류: 응답에 시간 관련 단어가 포함되어 있지 않습니다.")
            success = False
    
    print(f"로그 저장 위치: {log_path}")
    print(f"테스트 결과: {'성공' if success else '실패'}")
    return success

def run_all_tests(host="localhost", port=8000):
    """모든 테스트 실행"""
    log_dir = setup_logging()
    
    # 테스트 실행
    results = []
    results.append(("인사 테스트", run_test_greeting(log_dir, host, port)))
    results.append(("한국어 테스트", run_test_korean(log_dir, host, port)))
    results.append(("도구 사용 테스트", run_test_tool_use(log_dir, host, port)))
    
    # 결과 요약
    print("\n=== 테스트 결과 요약 ===")
    all_success = True
    for test_name, result in results:
        status = "성공" if result else "실패"
        print(f"{test_name}: {status}")
        if not result:
            all_success = False
    
    return 0 if all_success else 1

def main():
    parser = argparse.ArgumentParser(description="E2E 테스트 실행")
    parser.add_argument("--host", default="localhost", help="API 서버 호스트 (기본값: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="API 서버 포트 (기본값: 8000)")
    
    args = parser.parse_args()
    
    print(f"API 서버 주소: http://{args.host}:{args.port}")
    return run_all_tests(args.host, args.port)

if __name__ == "__main__":
    sys.exit(main()) 