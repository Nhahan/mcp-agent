#!/bin/bash

# MCP Agent 실행 스크립트
# 백엔드 서버와 프론트엔드 개발 서버를 동시에 실행합니다.

# 스크립트 종료 시 백그라운드 프로세스 정리
cleanup() {
  echo "프로세스 정리 중..."
  if [ ! -z "$BACKEND_PID" ]; then
    echo "백엔드 서버 종료 (PID: $BACKEND_PID)"
    kill $BACKEND_PID 2>/dev/null
  fi
  if [ ! -z "$FRONTEND_PID" ]; then
    echo "프론트엔드 서버 종료 (PID: $FRONTEND_PID)"
    kill $FRONTEND_PID 2>/dev/null
  fi
  exit 0
}

# Ctrl+C 등 인터럽트 신호 처리
trap cleanup SIGINT SIGTERM

# 필요한 디렉토리가 있는지 확인
if [ ! -d "app" ]; then
  echo "Error: 'app' 디렉토리를 찾을 수 없습니다. 루트 디렉토리에서 실행하세요."
  exit 1
fi

if [ ! -d "frontend" ]; then
  echo "Error: 'frontend' 디렉토리를 찾을 수 없습니다. 루트 디렉토리에서 실행하세요."
  exit 1
fi

# 로고 출력
echo "---------------------------------------------"
echo "  MCP Agent 개발 서버 시작"
echo "---------------------------------------------"

# 백엔드 서버 시작
echo "백엔드 서버 시작 중..."
python -m app.main &
BACKEND_PID=$!

# 백엔드 서버가 제대로 시작되었는지 확인
sleep 2
if ! ps -p $BACKEND_PID > /dev/null; then
  echo "Error: 백엔드 서버 실행 실패"
  cleanup
  exit 1
fi
echo "백엔드 서버 실행 중 (PID: $BACKEND_PID)"
echo "백엔드 API 주소: http://localhost:8000"

# 프론트엔드 서버 시작
echo "프론트엔드 서버 시작 중..."
cd frontend 

# npm install 추가 (필요한 경우에만 실행되도록 개선 가능)
echo "프론트엔드 의존성 확인 및 설치 중..."
npm install 

echo "프론트엔드 개발 서버 실행 중..."
npm run dev &
FRONTEND_PID=$!

# 프론트엔드 서버가 제대로 시작되었는지 확인
sleep 5
if ! ps -p $FRONTEND_PID > /dev/null; then
  echo "Error: 프론트엔드 서버 실행 실패"
  cleanup
  exit 1
fi
echo "프론트엔드 서버 실행 중 (PID: $FRONTEND_PID)"

# 모든 서버가 실행된 후 정보 출력
echo "---------------------------------------------"
echo "MCP Agent 서버 실행 완료!"
echo "프론트엔드: http://localhost:5173"
echo "백엔드 API: http://localhost:8000"
echo "백엔드 API 문서: http://localhost:8000/docs"
echo "종료하려면 Ctrl+C를 누르세요."
echo "---------------------------------------------"

# 백그라운드 프로세스가 종료될 때까지 대기
wait 