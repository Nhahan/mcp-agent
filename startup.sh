#!/bin/bash

MODEL_PATH=${MODEL_PATH:-"test_model_cache/gemma-3-1b-it-q4_0.gguf"}
MODEL_URL=${MODEL_URL:-"https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf/resolve/main/gemma-3-1b-it-q4_0.gguf?download=true"}
MODEL_REPO_ID=${MODEL_REPO_ID:-"google/gemma-3-1b-it-qat-q4_0-gguf"}
MODEL_FILENAME=${MODEL_FILENAME:-"gemma-3-1b-it-q4_0.gguf"}

echo "모델 경로 확인: $MODEL_PATH"

# 디렉토리 확인 및 생성
mkdir -p $(dirname $MODEL_PATH)
mkdir -p logs

# 모델 파일 확인
if [ ! -f "$MODEL_PATH" ]; then
  echo "모델 파일이 없습니다. 다운로드를 시도합니다..."
  
  # Python 스크립트로 다운로드 시도 (huggingface_hub 라이브러리 사용)
  echo "Python 스크립트를 사용한 다운로드 시도..."
  python - <<EOF
import sys
from huggingface_hub import hf_hub_download
import shutil
from pathlib import Path
import os

MODEL_PATH = "${MODEL_PATH}"
MODEL_REPO_ID = "${MODEL_REPO_ID}"
MODEL_FILENAME = "${MODEL_FILENAME}"

print(f"huggingface_hub 라이브러리를 사용하여 다운로드 시도...")
print(f"repo_id: {MODEL_REPO_ID}")
print(f"filename: {MODEL_FILENAME}")
print(f"저장 경로: {MODEL_PATH}")

try:
    # 디렉토리 생성
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    # 다운로드
    file_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        cache_dir=Path(MODEL_PATH).parent,
        force_download=False,
        resume_download=True
    )
    
    print(f"다운로드 완료: {file_path}")
    
    # 원하는 경로로 복사
    if Path(file_path) != Path(MODEL_PATH):
        print(f"파일을 목적지로 복사: {MODEL_PATH}")
        shutil.copy(file_path, MODEL_PATH)
    
    print(f"파일 준비 완료: {MODEL_PATH}")
    sys.exit(0)
except Exception as e:
    print(f"huggingface_hub 다운로드 실패: {e}")
    sys.exit(1)
EOF

  # Python 스크립트 실패 시 curl 백업 방법 사용
  if [ $? -ne 0 ]; then
    echo "Python 다운로드 실패. curl을 사용한 다운로드 시도..."
    
    # Hugging Face 토큰 확인
    if [ -z "$HUGGING_FACE_TOKEN" ]; then
      echo "경고: HUGGING_FACE_TOKEN이 설정되지 않았습니다. 인증이 필요한 모델은 다운로드가 실패할 수 있습니다."
      curl -L -o "$MODEL_PATH" "$MODEL_URL"
    else
      echo "Hugging Face 토큰을 사용하여 다운로드합니다."
      curl -L -o "$MODEL_PATH" "$MODEL_URL" \
        -H "Authorization: Bearer $HUGGING_FACE_TOKEN"
    fi
    
    # 다운로드 결과 확인
    if [ $? -ne 0 ]; then
      echo "모델 다운로드 실패!"
      exit 1
    fi
  fi
  
  echo "모델 다운로드 완료"
fi

# 파일 크기 확인
file_size=$(stat -c%s "$MODEL_PATH" 2>/dev/null || stat -f%z "$MODEL_PATH")
if [ "$file_size" -lt 1000000 ]; then  # 1MB 미만이면 에러
  echo "모델 파일이 너무 작습니다: $file_size 바이트"
  echo "파일이 손상되었거나 잘못된 파일일 수 있습니다."
  exit 1
fi

echo "모델 파일 준비 완료: $MODEL_PATH ($file_size 바이트)"
echo "애플리케이션 시작..."

# 애플리케이션 실행
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 