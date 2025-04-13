# MCP Server (Model Context Protocol Server)

## 개요

MCP(Model Context Protocol) Server는 Anthropic에서 개발한 오픈소스 프로토콜인 Model Context Protocol을 기반으로 하는 서버입니다. 이 프로토콜은 대규모 언어 모델(LLM)과 같은 AI 시스템이 외부 데이터 소스, 도구, 프롬프트 등에 안전하고 표준화된 방식으로 접근할 수 있도록 설계되었습니다.

MCP Server는 AI 클라이언트(예: Claude Desktop, Cursor)와 1:1 연결을 맺고, 정의된 인터페이스를 통해 AI가 활용할 수 있는 다양한 기능과 정보를 제공합니다.

## 주요 기능 및 역할

*   **데이터 접근:** 파일 시스템, 데이터베이스(PostgreSQL, SQLite, MySQL 등), 클라우드 스토리지(Google Drive, AWS S3 등) 등 다양한 데이터 소스에 대한 접근 권한을 AI에게 제공합니다. 접근 제어 설정이 가능하여 보안을 유지할 수 있습니다.
*   **도구 제공:** AI가 특정 작업을 수행할 수 있도록 도구를 제공합니다. 예를 들어 Git 명령어 실행, GitHub/GitLab API 연동, 웹 검색(Brave Search), 웹 콘텐츠 가져오기(Fetch), 브라우저 자동화(Puppeteer), 이미지 생성(EverArt), 코드 실행 등의 기능을 도구 형태로 제공할 수 있습니다.
*   **프롬프트 제공:** AI와의 상호작용을 위한 템플릿화된 프롬프트를 제공할 수 있습니다.
*   **보안:** MCP Server는 자체적으로 리소스를 관리하고 접근 제어를 수행합니다. LLM 제공자에게 API 키 등을 공유할 필요 없이 명확한 시스템 경계를 유지하여 보안성을 높입니다.

## 구현 및 종류

MCP Server는 TypeScript, Python, Java, Go 등 다양한 언어로 구현될 수 있으며, 공식 SDK가 제공됩니다.

*   **참조 구현 (Reference Implementations):** Anthropic에서 제공하는 공식 예제 서버들로, MCP의 핵심 기능과 SDK 사용법을 보여줍니다. (예: Filesystem, Git, GitHub, PostgreSQL, Fetch, Brave Search 등)
*   **공식 통합 (Official Integrations):** 다양한 회사들이 자사 플랫폼을 위해 개발 및 유지보수하는 MCP 서버들입니다. (예: Axiom, Browserbase, Cloudflare, E2B, Neon, Stripe, Weaviate 등)
*   **커뮤니티 서버 (Community Servers):** 커뮤니티 개발자들이 다양한 목적으로 개발하고 유지보수하는 서버들입니다. (예: Docker, Kubernetes, Linear, Snowflake, Spotify, Todoist 등)

## 활용

MCP Server를 사용하면 LLM 기반의 AI Agent가 다음과 같은 작업을 수행할 수 있습니다.

*   로컬 파일 시스템의 파일 읽기/쓰기
*   데이터베이스 쿼리 실행
*   Git 저장소 관리
*   웹 검색 수행 및 웹 페이지 내용 분석
*   외부 API 호출 (GitHub, Slack, Google Maps 등)
*   코드 실행 및 디버깅
*   다양한 생산성 도구 연동 (일정 관리, 메모 작성 등)

MCP Server는 AI Agent가 실제 세계의 정보와 도구를 활용하여 더욱 복잡하고 유용한 작업을 수행할 수 있도록 돕는 핵심적인 구성 요소입니다.

## 참고 자료

*   **MCP 공식 문서:** [https://modelcontextprotocol.io/](https://modelcontextprotocol.io/)
*   **MCP 서버 저장소 (GitHub):** [https://github.com/modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)
*   **MCP 서버 목록 (mcp.so):** [https://mcp.so/](https://mcp.so/) 