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
    log_dir = ROOT_DIR / "logs" / "reasoning_tests"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

def call_chat_api(prompt, host="localhost", port=8000):
    """Chat API 호출"""
    url = f"http://{host}:{port}/api/v1/chat"
    payload = {
        "text": prompt
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)  # 긴 시간 작업을 위해 타임아웃 확장
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API 호출 오류: {e}")
        return {"error": str(e)}

def save_test_result(log_dir, test_name, prompt, response, analysis):
    """테스트 결과 저장"""
    timestamp = int(time.time())
    log_path = log_dir / f"test_{test_name}_{timestamp}.json"
    
    result = {
        "test_name": test_name,
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "response": response,
        "analysis": analysis
    }
    
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return log_path

def check_reasoning_quality(response):
    """응답의 추론 품질 분석"""
    # 응답에서 필요한 정보 추출
    full_response = response.get("full_response", "")
    iterations = response.get("iterations", 0)
    used_tool = response.get("used_tool", False)
    thoughts_and_actions = response.get("thoughts_and_actions", [])
    final_response = response.get("response", "")
    
    # 추론 품질 점수 및 분석 결과
    quality_score = 0
    analysis = {
        "iterations": iterations,
        "used_tool": used_tool,
        "tool_usage_count": len(thoughts_and_actions),
        "has_thought_process": False,
        "has_reasoning": False,
        "response_quality": "poor"
    }
    
    # 생각 과정(Thought) 존재 여부 확인
    if "Thought:" in full_response:
        analysis["has_thought_process"] = True
        quality_score += 1
    
    # 실제 추론 과정 존재 여부 확인 (기본적인 추론 표현 검사)
    reasoning_indicators = [
        "because", "therefore", "thus", "since", "as a result", "consequently",
        "so", "이유는", "따라서", "그래서", "결과적으로", "그러므로"
    ]
    
    if any(indicator in final_response.lower() for indicator in reasoning_indicators):
        analysis["has_reasoning"] = True
        quality_score += 1
    
    # 도구 사용 횟수에 따른 점수
    if used_tool:
        tool_count = len(thoughts_and_actions)
        if tool_count > 0:
            quality_score += min(tool_count, 3)  # 최대 3점까지 추가
    
    # 최종 품질 평가
    if quality_score >= 4:
        analysis["response_quality"] = "excellent"
    elif quality_score >= 2:
        analysis["response_quality"] = "good"
    else:
        analysis["response_quality"] = "poor"
    
    analysis["quality_score"] = quality_score
    return analysis

def run_factual_query_test(log_dir, host="localhost", port=8000):
    """사실 기반 쿼리 테스트"""
    print("\n=== 테스트: 사실 기반 쿼리 ===")
    prompt = """리눅스에서 디렉토리 내용을 나열하는 명령어와 파일 크기를 확인하는 명령어는 무엇인지 알려주세요."""
    print(f"프롬프트: {prompt}")
    
    start_time = time.time()
    response = call_chat_api(prompt, host, port)
    elapsed_time = time.time() - start_time
    
    # 응답 분석
    analysis = check_reasoning_quality(response)
    analysis["elapsed_time"] = elapsed_time
    
    log_path = save_test_result(log_dir, "factual_query", prompt, response, analysis)
    
    success = True
    if "error" in response and response["error"]:
        print(f"오류: {response['error']}")
        success = False
    else:
        final_response = response.get("response", "")
        print(f"응답: {final_response}")
        print(f"응답 시간: {elapsed_time:.2f}초")
        print(f"추론 품질 점수: {analysis['quality_score']}/5 ({analysis['response_quality']})")
        print(f"도구 사용 여부: {'O' if analysis['used_tool'] else 'X'}")
        print(f"반복 횟수: {analysis['iterations']}")
    
    print(f"로그 저장 위치: {log_path}")
    print(f"테스트 결과: {'성공' if success else '실패'}")
    return success, analysis

def run_reasoning_query_test(log_dir, host="localhost", port=8000):
    """추론 기반 쿼리 테스트"""
    print("\n=== 테스트: 추론 기반 쿼리 ===")
    prompt = """프로그래밍 언어 Python과 JavaScript의 주요 차이점을 분석하고, 웹 개발에 어떤 언어가 더 적합한지 이유와 함께 설명해주세요."""
    print(f"프롬프트: {prompt}")
    
    start_time = time.time()
    response = call_chat_api(prompt, host, port)
    elapsed_time = time.time() - start_time
    
    # 응답 분석
    analysis = check_reasoning_quality(response)
    analysis["elapsed_time"] = elapsed_time
    
    log_path = save_test_result(log_dir, "reasoning_query", prompt, response, analysis)
    
    success = True
    if "error" in response and response["error"]:
        print(f"오류: {response['error']}")
        success = False
    else:
        final_response = response.get("response", "")
        print(f"응답: {final_response}")
        print(f"응답 시간: {elapsed_time:.2f}초")
        print(f"추론 품질 점수: {analysis['quality_score']}/5 ({analysis['response_quality']})")
        print(f"도구 사용 여부: {'O' if analysis['used_tool'] else 'X'}")
        print(f"반복 횟수: {analysis['iterations']}")
    
    print(f"로그 저장 위치: {log_path}")
    print(f"테스트 결과: {'성공' if success else '실패'}")
    return success, analysis

def run_creative_query_test(log_dir, host="localhost", port=8000):
    """창의적 쿼리 테스트"""
    print("\n=== 테스트: 창의적 쿼리 ===")
    prompt = """블록체인 기술을 활용하여 의료 기록 관리 시스템을 개발한다면 어떤 장단점이 있을지 분석해주세요."""
    print(f"프롬프트: {prompt}")
    
    start_time = time.time()
    response = call_chat_api(prompt, host, port)
    elapsed_time = time.time() - start_time
    
    # 응답 분석
    analysis = check_reasoning_quality(response)
    analysis["elapsed_time"] = elapsed_time
    
    log_path = save_test_result(log_dir, "creative_query", prompt, response, analysis)
    
    success = True
    if "error" in response and response["error"]:
        print(f"오류: {response['error']}")
        success = False
    else:
        final_response = response.get("response", "")
        print(f"응답: {final_response}")
        print(f"응답 시간: {elapsed_time:.2f}초")
        print(f"추론 품질 점수: {analysis['quality_score']}/5 ({analysis['response_quality']})")
        print(f"도구 사용 여부: {'O' if analysis['used_tool'] else 'X'}")
        print(f"반복 횟수: {analysis['iterations']}")
    
    print(f"로그 저장 위치: {log_path}")
    print(f"테스트 결과: {'성공' if success else '실패'}")
    return success, analysis

def main():
    parser = argparse.ArgumentParser(description="AI 모델 추론 품질 테스트")
    parser.add_argument("--host", default="localhost", help="API 서버 호스트 (기본값: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="API 서버 포트 (기본값: 8000)")
    parser.add_argument("--prompt", help="특정 프롬프트로 테스트합니다 (지정하면 해당 프롬프트만 테스트)")
    
    args = parser.parse_args()
    
    log_dir = setup_logging()
    
    print(f"API 서버 주소: http://{args.host}:{args.port}")
    
    # 사용자 지정 프롬프트가 있으면 해당 프롬프트로만 테스트
    if args.prompt:
        print("\n=== 테스트: 사용자 지정 쿼리 ===")
        print(f"프롬프트: {args.prompt}")
        
        start_time = time.time()
        response = call_chat_api(args.prompt, args.host, args.port)
        elapsed_time = time.time() - start_time
        
        # 응답 분석
        analysis = check_reasoning_quality(response)
        analysis["elapsed_time"] = elapsed_time
        
        log_path = save_test_result(log_dir, "custom_query", args.prompt, response, analysis)
        
        success = True
        if "error" in response and response["error"]:
            print(f"오류: {response['error']}")
            success = False
        else:
            final_response = response.get("response", "")
            print(f"응답: {final_response}")
            print(f"응답 시간: {elapsed_time:.2f}초")
            print(f"추론 품질 점수: {analysis['quality_score']}/5 ({analysis['response_quality']})")
            print(f"도구 사용 여부: {'O' if analysis['used_tool'] else 'X'}")
            print(f"반복 횟수: {analysis['iterations']}")
        
        print(f"로그 저장 위치: {log_path}")
        print(f"테스트 결과: {'성공' if success else '실패'}")
        return 0 if success else 1
    
    # 표준 테스트 실행
    results = []
    
    # 사실 기반 쿼리 테스트
    factual_success, factual_analysis = run_factual_query_test(log_dir, args.host, args.port)
    results.append(("사실 기반 쿼리", factual_success, factual_analysis))
    
    # 추론 기반 쿼리 테스트
    reasoning_success, reasoning_analysis = run_reasoning_query_test(log_dir, args.host, args.port)
    results.append(("추론 기반 쿼리", reasoning_success, reasoning_analysis))
    
    # 창의적 쿼리 테스트
    creative_success, creative_analysis = run_creative_query_test(log_dir, args.host, args.port)
    results.append(("창의적 쿼리", creative_success, creative_analysis))
    
    # 결과 요약
    print("\n=== 테스트 결과 요약 ===")
    total_score = 0
    max_score = 0
    for test_name, success, analysis in results:
        status = "성공" if success else "실패"
        quality = analysis["response_quality"]
        score = analysis["quality_score"]
        total_score += score
        max_score += 5  # 각 테스트의 최대 점수는 5
        print(f"{test_name} 테스트: {status} (품질: {quality}, 점수: {score}/5)")
    
    # 종합 평가
    overall_percentage = (total_score / max_score) * 100
    print(f"\n종합 추론 품질 점수: {total_score}/{max_score} ({overall_percentage:.1f}%)")
    
    if overall_percentage >= 70:
        print("종합 평가: 우수 - 모델이 대체로 좋은 추론 과정을 보여주고 있습니다.")
    elif overall_percentage >= 40:
        print("종합 평가: 양호 - 모델이 기본적인 추론 과정을 보여주지만 개선이 필요합니다.")
    else:
        print("종합 평가: 미흡 - 모델의 추론 과정이 부족하며 상당한 개선이 필요합니다.")
    
    # 모든 테스트가 성공했는지 확인
    all_success = all(success for _, success, _ in results)
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main()) 