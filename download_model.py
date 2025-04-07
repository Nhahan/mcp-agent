#!/usr/bin/env python
"""
Gemma 3 1B GGUF 모델 다운로드 스크립트
https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf/resolve/main/gemma-3-1b-it-q4_0.gguf?download=true 에서 다운로드
"""

import os
import sys
import shutil
from pathlib import Path
from tqdm import tqdm
import asyncio
import httpx

# huggingface_hub 가져오기 시도
try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    print("huggingface_hub 라이브러리가 설치되지 않았습니다. pip install huggingface_hub로 설치하세요.")
    HF_HUB_AVAILABLE = False

# 모델 URL 및 파일 경로 설정
MODEL_URL = "https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf/resolve/main/gemma-3-1b-it-q4_0.gguf?download=true"
MODEL_PATH = Path("test_model_cache/gemma-3-1b-it-q4_0.gguf")
MODEL_REPO_ID = "google/gemma-3-1b-it-qat-q4_0-gguf"
MODEL_FILENAME = "gemma-3-1b-it-q4_0.gguf"

def download_with_huggingface_hub():
    """huggingface_hub 라이브러리를 사용하여 다운로드"""
    print(f"\n=== huggingface_hub 라이브러리 다운로드 시도 ===")
    print(f"repo_id: {MODEL_REPO_ID}")
    print(f"filename: {MODEL_FILENAME}")
    print(f"저장 디렉토리: {MODEL_PATH.parent}")
    
    try:
        # 디렉토리 생성
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # hf_hub_download 사용
        file_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            cache_dir=MODEL_PATH.parent,
            force_download=False,
            resume_download=True
        )
        
        print(f"다운로드 완료: {file_path}")
        
        # 원하는 경로로 복사
        if Path(file_path) != MODEL_PATH:
            print(f"파일을 목적지로 복사: {MODEL_PATH}")
            shutil.copy(file_path, MODEL_PATH)
        
        print(f"파일 준비 완료: {MODEL_PATH}")
        print(f"파일 크기: {MODEL_PATH.stat().st_size / (1024*1024):.2f} MB")
        return True
    except Exception as e:
        print(f"huggingface_hub 다운로드 실패: {e}")
        return False

async def download_with_httpx():
    """httpx를 사용하여 다운로드"""
    print(f"\n=== httpx 라이브러리 다운로드 시도 ===")
    print(f"다운로드 URL: {MODEL_URL}")
    print(f"저장 경로: {MODEL_PATH}")
    
    # 디렉토리 생성
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Hugging Face 토큰 확인
    hf_token = os.environ.get("HUGGING_FACE_TOKEN")
    if not hf_token:
        print("경고: HUGGING_FACE_TOKEN 환경 변수가 설정되지 않았습니다. 인증이 필요한 모델 다운로드가 실패할 수 있습니다.")
    
    # 인증 헤더 준비
    headers = {}
    if hf_token:
        print("Hugging Face 토큰을 사용하여 인증합니다.")
        headers["Authorization"] = f"Bearer {hf_token}"
    
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            # Head 요청으로 상태 확인
            try:
                print(f"URL 유효성 확인 중: {MODEL_URL}")
                head_response = await client.head(MODEL_URL, headers=headers)
                head_response.raise_for_status()
                print(f"URL 유효성 확인 완료: HTTP {head_response.status_code}")
            except httpx.HTTPStatusError as e:
                print(f"URL 유효성 검사 실패: {e}")
                print(f"상태 코드: {e.response.status_code}")
                if e.response.status_code == 401:
                    print("인증 오류: 인증이 필요한 모델입니다.")
                    print("1. Hugging Face 계정으로 로그인하세요: https://huggingface.co/login")
                    print("2. 모델 라이센스에 동의하세요: https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf")
                    print("3. 토큰을 생성하세요: https://huggingface.co/settings/tokens")
                    print("4. 환경 변수로 설정하세요: export HUGGING_FACE_TOKEN=your_token_here")
                print(f"응답 내용: {e.response.text[:500] if hasattr(e.response, 'text') else 'No response text'}")
                return False
            except Exception as e:
                print(f"URL 유효성 검사 중 오류: {e}")
                return False
            
            # GET 요청으로 파일 다운로드
            try:
                print("GET 요청으로 파일 다운로드 시작...")
                async with client.stream("GET", MODEL_URL, headers=headers) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get("content-length", 0))
                    
                    # tqdm 진행 표시줄 설정
                    progress = tqdm(
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"다운로드 중: {MODEL_PATH.name}"
                    )
                    
                    # 파일에 저장
                    with open(MODEL_PATH, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            f.write(chunk)
                            progress.update(len(chunk))
                    
                    progress.close()
                    print(f"파일 다운로드 완료: {MODEL_PATH}")
                    print(f"파일 크기: {MODEL_PATH.stat().st_size / (1024*1024):.2f} MB")
                    return True
            except httpx.HTTPStatusError as e:
                print(f"다운로드 실패: HTTP {e.response.status_code}")
                print(f"응답 내용: {e.response.text[:500] if hasattr(e.response, 'text') else 'No response text'}")
                return False
            except Exception as e:
                print(f"다운로드 중 오류 발생: {e}")
                return False
    except Exception as e:
        print(f"예기치 않은 오류: {e}")
        return False

def main():
    """메인 함수"""
    if MODEL_PATH.exists():
        print(f"모델 파일이 이미 존재합니다: {MODEL_PATH}")
        print(f"파일 크기: {MODEL_PATH.stat().st_size / (1024*1024):.2f} MB")
        return True
    
    # 1. huggingface_hub 라이브러리 시도 (권장 방법)
    if HF_HUB_AVAILABLE:
        if download_with_huggingface_hub():
            return True
        else:
            print("huggingface_hub 다운로드 실패, httpx 방식으로 시도합니다.")
    
    # 2. httpx 다운로드 시도 (대체 방법)
    loop = asyncio.get_event_loop()
    success = loop.run_until_complete(download_with_httpx())
    
    if success:
        print(f"다운로드 성공! 파일 크기: {MODEL_PATH.stat().st_size / (1024*1024):.2f} MB")
        return True
    else:
        print("다운로드에 실패했습니다. HUGGING_FACE_TOKEN 환경 변수를 설정하고 다시 시도하세요.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 