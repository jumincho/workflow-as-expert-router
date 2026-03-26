# WaE-Router E2E 종료 보고서

작성일: 2026-03-27  
아카이브 대상 저장소: `/workspace/wae-router-e2e-handoff`

## 1. 한눈에 보는 결론

이 프로젝트는 **WaE-Router (Workflow-as-Expert)** 가 **MasRouter** 대비 실제로 의미 있는 확장인지 검증하려던 실험 패키지입니다.

검증하려던 주장은 두 가지였습니다.

1. **G1: 프레임워크 주장**  
   bare-model expert 대신 workflow expert를 쓰면 비용-정확도-지연 tradeoff가 좋아진다.
2. **G2: 동적 라우팅 주장**  
   동적으로 workflow를 고르는 방식이 `wae_static_cheap`, `wae_cheap_first_escalate`, `wae_dynamic_control_forced_io_general` 같은 강한 WaE baseline보다 더 낫다.

지금까지의 결과를 가장 정직하게 요약하면 다음과 같습니다.

- **G1은 꽤 반복적으로 지지되었습니다.**  
  WaE 방식은 MasRouter 대비 비용 효율 면에서 여러 라운드에서 좋은 신호를 보였습니다.
- **G2는 끝내 설득력 있게 입증되지 못했습니다.**  
  `wae_dynamic`이 가장 강한 WaE baseline들을 안정적으로 이겼다고 말하기 어렵습니다.
- 따라서 이 프로젝트의 최종 산출은 “동적 라우팅이 결정적이다”라기보다, **“workflow를 expert처럼 다루는 WaE 프레임 자체는 유의미할 수 있다”** 에 더 가깝습니다.

이 아카이브는 활성 연구 저장소가 아니라, **종료된 프로젝트의 보존용 handoff 번들**로 보는 것이 맞습니다.

## 2. 이 프로젝트는 무엇을 하려던 것이었나

핵심 아이디어는 모델만 고르는 것이 아니라 **워크플로우 자체를 expert로 취급해 라우팅**하는 것이었습니다.

즉, “어느 모델을 호출할까?”만이 아니라 다음과 같은 workflow를 expert처럼 정의하고 선택합니다.

- 저비용 direct I/O 생성
- refine/coder 계열 workflow
- premium 다중 후보 생성 workflow
- critique/refine 계열 workflow

이 workflow들을 LLM처럼 감싸고, offline library + online routing 조합으로 어떤 workflow를 쓸지 고릅니다. 실험의 핵심 질문은 두 가지였습니다.

- 이런 구조가 MasRouter보다 더 좋은 운영 frontier를 만드는가
- 그중에서도 동적 선택(dynamic routing) 자체가 단순한 정적 baseline보다 더 나은가

주요 평가 태스크는:

- `MBPP`
- `HumanEval`

주요 지표는:

- `accuracy_or_pass1`
- `avg_cost`
- `latency_p50_s`, `latency_p95_s`
- routing / workflow 사용 진단 지표

판정은 주로 다음 기준을 따랐습니다.

- dominance-first
- iso-cost 비교
- 강한 baseline과의 직접 비교

## 3. 저장소에서 중요한 코드가 무엇인가

핵심 구현은 주로 `src/` 아래에 있습니다.

- `src/run_pilot.py`: 메인 실험 러너
- `src/workflow_router_patch.py`: workflow-aware routing 핵심
- `src/workflow_llm.py`: workflow를 LLM 인터페이스처럼 감싸는 래퍼
- `src/workflow_profile.py`: workflow 정의
- `src/offline_pareto_builder.py`: offline library / Pareto 구성
- `src/compare_runs.py`: run 비교, iso-cost 판정, 리포트 생성
- `src/monitor.py`: 실행 상태 모니터링

운영 측면에서는 다음 디렉터리가 중요합니다.

- `config/`: 실험 및 endpoint 설정
- `scripts/`: 실행 및 재개 스크립트
- `docs/`: 개요, runbook, known issues
- `status/`: 마지막 상태 스냅샷
- `artifacts/`: 이전 라운드 보고서와 일부 결과물

## 4. 프로젝트가 어떻게 진행되었는가

### 초기 파일럿 단계

가장 처음에는 WaE 아이디어가 end-to-end로라도 신호가 있는지 보는 것이 목적이었습니다.

결과:

- `MBPP`에서는 가능성 신호가 있었음
- `HumanEval`에서는 그 신호가 깔끔하게 재현되지 않았음
- 당시에는 서빙과 실험 규모 제약도 컸음

즉, “계속 볼 가치는 있다” 정도였지, 강한 결론을 내릴 단계는 아니었습니다.

### Round5q1

Round5q1은 좀 더 결론 지향적인 첫 라운드였습니다. 강한 baseline, dominance-first 판정, routing 분석이 본격적으로 들어갔습니다.

핵심 결과:

- `wae_dynamic`은 MasRouter 대비 효율 신호를 보였음
- 하지만 **가장 중요한 동적 라우팅 주장에는 실패**
- 특히 `wae_static_cheap`, `wae_cheap_first_escalate`가 매우 강한 baseline으로 확인됨

이 라운드에서 보인 그림은 명확했습니다.

- WaE 프레임 자체는 유망
- 하지만 그 이득의 원천이 “dynamic routing 자체”인지에는 의문이 남음

### Round6r1

Round6r1은 multi-seed로 확대한 더 본격적인 검증 라운드였습니다.

이 라운드가 사실상 이 프로젝트의 최종 결론을 가장 안정적으로 만들어 주었습니다.

- `wae_dynamic` vs `masrouter_balanced` 3-seed 평균:
  - `MBPP`: accuracy `+0.0083p`, cost `-42.7%`, p50 latency `-36.9%`
  - `HumanEval`: accuracy `+0.0625p`, cost `-54.3%`, p50 latency `-2.0%`
- 즉, WaE 프레임 계열이 비용 효율 면에서 강하다는 신호는 재확인됨
- 그러나 dynamic routing이 strong baseline을 이긴다는 핵심 주장은 여전히 통과하지 못함

여기서 얻은 가장 중요한 결론은:

- **WaE라는 시스템 아이디어는 살아 있다**
- **dynamic routing이 핵심 차별점이라는 이야기는 아직 약하다**

### Round7r1

Round7r1에서는 dynamic routing 이야기를 더 살리기 위한 시도가 들어갔습니다.

- hardcase-gated 변형 추가
- tracing 강화
- latency 분해
- 사전 원인분해(P0)

하지만 이 라운드는 완주되지 않았습니다.

- Stage A 재현성 게이트까지만 완료
- 본 실험 매트릭스는 실행되지 않음
- 게이트도 token drift와 일부 mismatch 때문에 FAIL

즉, Round7r1은 도구/진단 관점에서는 의미가 있었지만, 최종 주장에 쓸 수 있는 완성 라운드는 아니었습니다.

### Round7r2

Round7r2는 이 handoff 패키지에 남아 있는 마지막 활성 라운드입니다.

계획:

- 총 `28`개 run

실제 상태:

- `19`개 완료
- `9`개 미실행

남아 있는 partial compare 결과를 보면, 이 라운드 역시 큰 방향은 바뀌지 않았습니다.

- dynamic routing이 강한 baseline을 깔끔하게 이기지 못함
- `wae_static_cheap`, `wae_dynamic_control_forced_io_general` 같은 대조군이 여전히 중요함

특히 중요한 partial finding 하나는:

- `HumanEval`에서 `wae_dynamic`이 `wae_dynamic_control_forced_io_general`에 지배됨

이건 “동적 라우팅 자체가 핵심 성분”이라는 서사와는 잘 맞지 않습니다.

## 5. 그래서 무엇을 알아냈는가

### 가장 강하게 말할 수 있는 주장

이 프로젝트가 가장 자신 있게 남기는 결론은 다음입니다.

> workflow를 expert로 쓰는 WaE 프레임은 MasRouter류 설정 대비 비용 효율 측면에서 실제 개선일 가능성이 높다.

이 주장은 다음에 의해 뒷받침됩니다.

- 초기 pilot
- Round5q1
- multi-seed인 Round6r1
- partial Round7 계열의 전반적 경향

### 끝내 강하게 말하지 못한 주장

반면 다음 주장은 끝까지 설득력 있게 서지 못했습니다.

> dynamic workflow routing 자체가 핵심적인 이득의 원천이다.

그 이유는:

- `wae_dynamic`이 강한 WaE baseline을 반복적으로 못 이김
- `wae_static_cheap`, `wae_cheap_first_escalate`가 너무 강했음
- control 변형을 보면 이득의 일부는 dynamic routing이 아니라 workflow/frame 설계 자체에서 온 것으로 보임
- Round7r2 partial도 이 방향을 바꾸지 못함

### 실무적으로 어떻게 해석해야 하나

누군가 이 저장소를 처음 열고 “그래서 무엇을 믿어야 하나?”라고 묻는다면 가장 좋은 답은 이렇습니다.

- **WaE라는 엔지니어링 패턴은 믿을 만하다.**
- **dynamic routing이 결정적이라는 주장은 믿기 어렵다.**

## 6. 왜 여기서 프로젝트를 접는가

이 코드베이스를 아카이브하는 이유는, 지금 상태로는 더 유지할 이유가 충분하지 않기 때문입니다.

핵심 이유:

1. 프레임워크 주장(G1)과 동적 라우팅 주장(G2)이 갈라졌습니다.
2. G1은 의미 있지만, 원래 목표보다 더 좁은 결론입니다.
3. G2는 강한 baseline을 상대로 반복 가능한 승리로 정리되지 못했습니다.
4. 마지막 라운드도 미완성입니다.

즉, 이 프로젝트는 무의미했던 것은 아닙니다. 실제로 배운 것이 있습니다. 다만 처음 기대했던 형태의 강한 결과로 정리되지는 않았습니다.

## 7. 종료 시점의 정직한 상태

종료 시점에서 가장 정직한 상태 요약은 다음과 같습니다.

- 저장소는 이미 작은 handoff 패키지 형태로 잘 정리되어 있음
- 마지막 활성 라운드(`round7r2`)는 미완성
- 가장 안정적인 결론은 **framework-level 결론**
- **routing superiority 결론은 아님**
- 향후 재개하더라도 같은 큰 서사를 그대로 이어 가기보다 더 좁은 framing이 필요함

## 8. 나중에 누가 다시 살린다면

이 작업을 다시 연다면 “조금만 더 튜닝하면 dynamic routing이 이길 것”이라는 가정에서 시작하면 안 됩니다.

더 현실적인 재개 방향은:

1. 주 프레이밍을 **workflow-expert 기반 시스템 효율**로 둔다.
2. dynamic routing은 부가 가설로 다룬다.
3. 반드시 다음 baseline을 강하게 유지한다.
   - `wae_static_cheap`
   - `wae_cheap_first_escalate`
   - forced-control 계열
4. routing-specific 주장을 하려면 더 큰 multi-seed 증거를 먼저 확보한다.

## 9. 이번 보관본에 무엇을 남겼는가

이 저장소는 원래부터 작기 때문에, `.git`만 제외하고 거의 전체 handoff 패키지를 보존하는 방식이 가장 안전합니다.

보관본에는 다음이 포함됩니다.

- 소스 코드
- 설정 파일
- 실행 스크립트
- 문서와 runbook
- 상태 스냅샷
- 과거 라운드 보고서와 partial artifact
- 이 종료 보고서 영문/한글판

## 10. 최종 한 줄 정리

이 프로젝트는 “workflow 단위 라우팅이 plain model routing보다 낫다”와 “그중에서도 dynamic workflow routing이 정적 WaE baseline보다 낫다”를 보여 주려는 진지한 시도였습니다.

최종적으로 남은 가장 좋은 결론은:

- **WaE는 의미 있는 시스템 패턴일 수 있다.**

하지만 끝내 깔끔하게 증명하지 못한 것은:

- **dynamic workflow routing이 결정적 우위라는 주장이다.**

이것이 이 저장소를 처음 읽는 사람에게 가장 정확하고 이해하기 쉬운 종료 시점 해석입니다.
