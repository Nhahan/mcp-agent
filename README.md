# MCP Agent: AI 에이전트 웹 컨테이너

이 프로젝트는 OS에 종속되지 않는 AI 에이전트를 웹 컨테이너 환경에서 실행하는 시스템입니다. [Gemma 3 1B](https://huggingface.co/google/gemma-3-1b-it) 모델을 사용하여 자연어 처리와 도구 호출 기능을 제공합니다.

## 주요 기능

- 웹 API를 통한 자연어 처리
- ReAct 패턴을 사용한 도구 호출 및 대화형 상호작용
- MCP(Model Context Protocol) 서버와의 연동
- 다국어 지원 (한국어 포함)
- 대화 로그 저장 및 분석

## 시스템 요구사항

- Python 3.10 이상
- 메모리: 최소 4GB (8GB 이상 권장)
- 디스크: 최소 2GB
- (선택) CUDA 지원 GPU (GPU 가속 시)

## 설치 방법

### 로컬 설치

1. 저장소 클론
   ```bash
   git clone https://github.com/yourusername/mcp-agent.git
   cd mcp-agent
   ```

2. 환경 설정
   ```bash
   cp .env.example .env
   # .env 파일 내용을 적절히 수정
   ```

3. 종속성 설치
   ```bash
   pip install -r requirements.txt
   ```

4. 모델 다운로드
   ```bash
   python download_model.py
   ```

### Docker 설치

1. Docker 이미지 빌드
   ```bash
   docker-compose build
   ```

2. Docker 컨테이너 실행
   ```bash
   docker-compose up
   ```

## 사용 방법

### API 서버 실행

```bash
python -m app.main
```

기본적으로 서버는 `http://localhost:8000`에서 실행됩니다.

### API 엔드포인트

- `/api/v1/chat`: 대화형 채팅 엔드포인트
  ```bash
  curl -X POST http://localhost:8000/api/v1/chat \
    -H "Content-Type: application/json" \
    -d '{"text": "안녕하세요"}'
  ```

- `/api/v1/status`: 시스템 상태 확인 엔드포인트
  ```bash
  curl http://localhost:8000/api/v1/status
  ```

### MCP 설정

MCP 서버를 추가하려면 `mcp.json` 파일을 편집하세요:

```json
{
  "mcpServers": {
    "iterm-mcp": {
      "command": "npx",
      "args": [
        "-y",
        "@smithery/cli@latest",
        "run",
        "iterm-mcp",
        "--key",
        "your-key-here"
      ]
    }
  }
}
```

## 테스트 실행

### 모델 직접 테스트

```bash
python test_scripts/test_model_direct.py
```

### 채팅 API 테스트

```bash
python test_scripts/test_chat_api.py --prompt "안녕하세요"
```

### E2E 테스트

```bash
python test_scripts/test_e2e_completion.py
```

## 로그 및 디버깅

로그 파일은 `logs` 디렉토리에 저장됩니다:

- `logs/react_logs/`: ReAct 패턴 처리 로그
- `logs/chat_tests/`: 채팅 테스트 로그
- `logs/e2e_tests/`: E2E 테스트 로그

## 문제해결

**Q: 모델 다운로드가 실패합니다.**
A: 네트워크 연결을 확인하고, 필요한 경우 Hugging Face 토큰을 설정하세요.

**Q: 메모리 부족 오류가 발생합니다.**
A: `max_tokens` 설정을 줄이거나 더 작은 모델로 전환하세요.

**Q: MCP 서버 연결이 실패합니다.**
A: `mcp.json` 파일의 설정을 확인하고 필요한 패키지가 설치되어 있는지 확인하세요.

## 라이선스

MIT

## 기여

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다.