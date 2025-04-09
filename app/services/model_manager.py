#!/usr/bin/env python3
"""
모델 관리 유틸리티

이 스크립트는 다양한 GGUF 모델을 다운로드하고 .env 파일을 자동으로 업데이트합니다.
사용자는 미리 정의된 모델 중에서 선택하거나 사용자 정의 HuggingFace 리포지토리와 파일명을 지정할 수 있습니다.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import shutil
import textwrap

from huggingface_hub import hf_hub_download, login
import httpx
import asyncio
from dotenv import load_dotenv, set_key

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_manager')

PREDEFINED_MODELS = {
    "gemma-3-1b-it-q4_0": {
        "repo_id": "google/gemma-3-1b-it-qat-q4_0-gguf",
        "filename": "gemma-3-1b-it-q4_0.gguf",
        "tokenizer": "google/gemma-3-1b-it",
        "description": "Gemma 3 1B-IT 4-bit Quantized (작은 규모, 빠른 추론)"
    },
    "gemma-3-4b-it-q4_0": {
        "repo_id": "google/gemma-3-4b-it-qat-q4_0-gguf",
        "filename": "gemma-3-4b-it-q4_0.gguf",
        "tokenizer": "google/gemma-3-4b-it",
        "description": "Gemma 3 4B-IT 4-bit Quantized (중간 규모, 높은 성능 밸런스)"
    },
    "gemma-3-8b-it-q4_0": {
        "repo_id": "google/gemma-3-8b-it-qat-q4_0-gguf",
        "filename": "gemma-3-8b-it-q4_0.gguf",
        "tokenizer": "google/gemma-3-8b-it",
        "description": "Gemma 3 8B-IT 4-bit Quantized (큰 규모, 고품질)"
    },
    "gemma-3-2b-it-q4_0": {
        "repo_id": "google/gemma-3-2b-it-qat-q4_0-gguf",
        "filename": "gemma-3-2b-it-q4_0.gguf",
        "tokenizer": "google/gemma-3-2b-it",
        "description": "Gemma 3 2B-IT 4-bit Quantized (작은 규모, 고성능 밸런스)"
    },
    "gemma3-4b-it-abliterated-q8_0": {
        "repo_id": "mradermacher/gemma3-4b-it-abliterated-GGUF",
        "filename": "gemma3-4b-it-abliterated.Q8_0.gguf",
        "tokenizer": "google/gemma-3-4b-it",
        "description": "Gemma 3 4B-IT Abliterated 8-bit Quantized"
    }
}

async def download_model_httpx(url, target_path, headers=None):
    """HTTP 클라이언트를 사용하여 모델을 다운로드합니다."""
    target_path = Path(target_path)
    temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    
    if temp_path.exists():
        temp_path.unlink()
    
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
            async with client.stream("GET", url, headers=headers or {}) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"Failed to download model. HTTP status: {response.status_code}")
                
                total_size = int(response.headers.get("content-length", 0))
                downloaded_size = 0
                
                print(f"Downloading model ({total_size / (1024*1024):.2f} MB) to {target_path}")
                
                with open(temp_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            if downloaded_size % (1024*1024*10) < 8192:  # 약 10MB마다
                                print(f"Progress: {progress:.2f}% ({downloaded_size/(1024*1024):.2f} MB / {total_size/(1024*1024):.2f} MB)")
                
                # 완료 후 최종 경로로 이동
                if temp_path.stat().st_size == 0:
                    raise RuntimeError("Downloaded file is empty")
                
                temp_path.replace(target_path)
                return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        if temp_path.exists():
            temp_path.unlink()
        if target_path.exists() and target_path.stat().st_size == 0:
            target_path.unlink()
        raise

def download_model_hf(repo_id, filename, target_path, token=None):
    """HuggingFace Hub에서 모델을 다운로드합니다."""
    try:
        print(f"Downloading {filename} from {repo_id}")
        target_path = Path(target_path)
        target_dir = target_path.parent
        
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=target_dir / ".cache",
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            force_download=False,
            resume_download=True,
            token=token
        )
        
        downloaded_path = Path(downloaded_path)
        
        # 파일 이름이 다른 경우 이동
        if downloaded_path != target_path and downloaded_path.exists():
            print(f"Moving {downloaded_path} to {target_path}")
            shutil.move(str(downloaded_path), str(target_path))
            
        return True
    except Exception as e:
        logger.error(f"HuggingFace download failed: {e}")
        raise

def update_env_file(model_info, env_path=".env"):
    """모델 정보로 .env 파일을 업데이트합니다."""
    env_path = Path(env_path)
    
    # .env 파일이 없으면 생성
    if not env_path.exists():
        with open(env_path, "w") as f:
            f.write("# AI 모델 설정\n")
    
    # 기존 .env 파일 로드
    load_dotenv(env_path)
    
    # 모델 설정 업데이트
    set_key(env_path, "MODEL_REPO_ID", model_info["repo_id"])
    set_key(env_path, "MODEL_FILENAME", model_info["filename"])
    set_key(env_path, "TOKENIZER_BASE_ID", model_info["tokenizer"])
    set_key(env_path, "AUTO_DOWNLOAD_MODEL", "true")
    
    print(f"\n.env 파일이 다음 설정으로 업데이트되었습니다:")
    print(f"MODEL_REPO_ID={model_info['repo_id']}")
    print(f"MODEL_FILENAME={model_info['filename']}")
    print(f"TOKENIZER_BASE_ID={model_info['tokenizer']}")

def list_available_models():
    """사용 가능한 사전 정의 모델 목록을 표시합니다."""
    print("\n사용 가능한 모델 목록:")
    print("-" * 80)
    for key, model in PREDEFINED_MODELS.items():
        print(f"{key}:")
        print(f"  설명: {model['description']}")
        print(f"  리포지토리: {model['repo_id']}")
        print(f"  파일명: {model['filename']}")
        print(f"  토크나이저: {model['tokenizer']}")
        print("-" * 80)

async def main():
    parser = argparse.ArgumentParser(description="모델을 다운로드하고 .env 파일을 업데이트하는 유틸리티")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="사용 가능한 모델 목록 보기")
    group.add_argument("--model", help="다운로드할 모델 이름 (사전 정의된 모델)")
    group.add_argument("--custom", action="store_true", help="사용자 정의 모델 지정")
    
    parser.add_argument("--repo-id", help="HuggingFace 리포지토리 ID (사용자 정의 모델용)")
    parser.add_argument("--filename", help="모델 파일 이름 (사용자 정의 모델용)")
    parser.add_argument("--tokenizer", help="토크나이저 ID (사용자 정의 모델용)")
    parser.add_argument("--model-dir", default="./models", help="모델 저장 디렉토리 (기본값: ./models)")
    parser.add_argument("--token", help="HuggingFace 토큰 (비공개 모델 접근용)")
    parser.add_argument("--env-path", default=".env", help=".env 파일 경로 (기본값: ./.env)")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        return 0
    
    # 모델 디렉토리 생성
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # HuggingFace 토큰 설정
    if args.token:
        print("HuggingFace 토큰을 설정합니다...")
        os.environ["HUGGING_FACE_TOKEN"] = args.token
        try:
            login(token=args.token)
        except Exception as e:
            print(f"경고: HuggingFace 로그인 실패 - {e}")
    
    # 모델 정보 결정
    if args.model:
        if args.model not in PREDEFINED_MODELS:
            print(f"오류: '{args.model}' 모델을 찾을 수 없습니다. --list 옵션으로 사용 가능한 모델을 확인하세요.")
            return 1
        
        model_info = PREDEFINED_MODELS[args.model]
        print(f"선택한 모델: {args.model} - {model_info['description']}")
    else:  # 사용자 정의 모델
        if not all([args.repo_id, args.filename, args.tokenizer]):
            print("오류: 사용자 정의 모델은 --repo-id, --filename, --tokenizer가 모두 필요합니다.")
            return 1
        
        model_info = {
            "repo_id": args.repo_id,
            "filename": args.filename,
            "tokenizer": args.tokenizer,
            "description": "사용자 정의 모델"
        }
        print(f"사용자 정의 모델: {args.repo_id}/{args.filename}")
    
    # 모델 파일 경로
    model_path = model_dir / model_info["filename"]
    
    # 모델 다운로드
    try:
        if model_path.exists():
            print(f"모델 파일이 이미 존재합니다: {model_path}")
            choice = input("다시 다운로드하시겠습니까? (y/N): ")
            if choice.lower() != 'y':
                print("다운로드를 건너뜁니다.")
                update_env_file(model_info, args.env_path)
                return 0
            model_path.unlink()
        
        print(f"모델 다운로드를 시작합니다: {model_info['repo_id']}/{model_info['filename']}")
        try:
            # 먼저 HuggingFace Hub로 시도
            success = download_model_hf(
                repo_id=model_info["repo_id"],
                filename=model_info["filename"],
                target_path=model_path,
                token=args.token
            )
        except Exception as e:
            print(f"HuggingFace 다운로드 실패: {e}")
            print("직접 다운로드로 시도합니다...")
            
            # 직접 URL로 시도
            model_url = f"https://huggingface.co/{model_info['repo_id']}/resolve/main/{model_info['filename']}"
            headers = {}
            if args.token:
                headers["Authorization"] = f"Bearer {args.token}"
            
            success = await download_model_httpx(
                url=model_url,
                target_path=model_path,
                headers=headers
            )
        
        if success:
            print(f"모델 다운로드 완료: {model_path}")
            update_env_file(model_info, args.env_path)
            return 0
        else:
            print("모델 다운로드 실패")
            return 1
    except Exception as e:
        print(f"오류: {e}")
        return 1

if __name__ == "__main__":
    # 사용 예시 출력
    if len(sys.argv) == 1:
        print(textwrap.dedent("""
        모델 매니저 - GGUF 모델 다운로드 및 .env 구성 도구
        
        사용 예시:
          # 사용 가능한 모델 목록 보기
          python model_manager.py --list
          
          # 사전 정의된 모델 다운로드
          python model_manager.py --model gemma-3-1b-it-q4_0
          
          # 사용자 정의 모델 다운로드
          python model_manager.py --custom --repo-id google/gemma-3-1b-it-qat-q4_0-gguf \\
                                 --filename gemma-3-1b-it-q4_0.gguf \\
                                 --tokenizer google/gemma-3-1b-it
          
          # HuggingFace 토큰과 함께 비공개 모델 다운로드
          python model_manager.py --model gemma-3-4b-it-q4_0 --token your_hf_token
        
        자세한 옵션:
          python model_manager.py --help
        """))
        sys.exit(0)
    
    # 메인 함수 실행
    sys.exit(asyncio.run(main())) 