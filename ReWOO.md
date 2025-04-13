# ReWOO (Reasoning WithOut Observation) 패턴

## 개요

ReWOO는 Xu 등이 "Reasoning without Observation" 논문에서 제안한 AI Agent 아키텍처 패턴입니다. 이 패턴은 다단계 계획(multi-step planner)과 변수 치환(variable substitution)을 결합하여 효과적인 도구 사용을 가능하게 합니다.

기존의 ReACT 스타일 Agent 아키텍처 대비 다음과 같은 개선점을 목표로 설계되었습니다:

1.  **효율성 향상:** 전체 도구 사용 계획을 한 번의 LLM 호출로 생성하여 토큰 소비량과 실행 시간을 줄입니다. (ReACT는 각 추론 단계마다 시스템 프롬프트와 이전 단계를 포함한 많은 LLM 호출 필요)
2.  **미세 조정 간소화:** 계획 데이터가 도구 출력에 의존하지 않으므로, 이론적으로 실제 도구를 호출하지 않고도 모델 미세 조정이 가능합니다.

## 핵심 모듈

ReWOO 패턴은 일반적으로 세 가지 주요 모듈로 구성됩니다:

1.  🧠 **Planner (계획자):**
    *   주어진 문제를 해결하기 위한 단계별 계획을 생성합니다.
    *   계획은 추론 과정과 함께 각 단계에서 사용할 도구 및 해당 도구의 인수를 명시합니다.
    *   이전 단계의 도구 실행 결과(Evidence)를 다음 단계의 도구 입력으로 활용하기 위한 변수 치환 메커니즘을 포함합니다.
    *   예시 형식:
        ```
        Plan: <추론 내용>
        #E1 = 도구[도구 인수]
        Plan: <추론 내용>
        #E2 = 도구[#E1 변수를 사용한 도구 인수]
        ...
        ```

2.  **Worker (작업자):**
    *   Planner가 생성한 계획에 따라 지정된 인수로 도구를 실행합니다.
    *   도구 실행 결과를 반환합니다. (이 프로젝트에서는 MCP Server 도구를 사용)

3.  **Solver (해결자):**
    *   Planner가 생성한 최종 계획과 Worker가 각 단계에서 실행한 도구의 결과(Evidence)를 종합합니다.
    *   모든 증거를 바탕으로 문제에 대한 최종 응답(Final Answer)을 생성합니다.

## 작동 방식 요약

1.  **Input:** 사용자가 문제를 입력합니다.
2.  **Plan:** Planner는 문제를 해결하기 위한 전체 계획(추론 + 도구 호출)을 생성합니다.
3.  **Tool & Tool Input (Worker):** Worker는 계획에 따라 필요한 도구를 순차적으로 실행하고 결과를 얻습니다(Output). 이때, 이전 단계의 Output(#E1, #E2 등)이 다음 단계의 Tool Input으로 사용될 수 있습니다.
4.  **Evidence:** 각 도구 실행 결과(Output)는 해당 계획 단계를 뒷받침하는 증거(Evidence)가 됩니다.
5.  **Final Answer (Solver):** Solver는 모든 계획 단계에서 얻어진 증거들을 종합하여 최종 응답을 생성합니다.

이러한 구조를 통해 ReWOO Agent는 불필요한 관찰(Observation) 없이 미리 정의된 계획에 따라 효율적으로 도구를 활용하고 추론하여 목표를 달성할 수 있습니다.

## 참고 자료

*   **ReWOO LangGraph Tutorial:** [https://langchain-ai.github.io/langgraph/tutorials/rewoo/rewoo/](https://langchain-ai.github.io/langgraph/tutorials/rewoo/rewoo/) 