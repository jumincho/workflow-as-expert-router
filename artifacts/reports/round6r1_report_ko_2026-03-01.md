# WaE-Router Round6r1 상세 실험 보고서 (2026-03-01)

## 1) 실험 개요
- 실험 ID: `round6r1`
- 목적: Round5/피드백 반영 상태에서 `MasRouter 대비 WaE 확장`의 성능-비용-지연 효율 및 `dynamic routing 본질 효과`를 seed 다회 반복으로 검증
- 판정 규칙: `dominance-first` 후 `iso-cost(±5%)에서 +3%p`
- 실행 기간(UTC): `2026-02-28 01:46:18` ~ `2026-03-01 11:24:22`

## 2) 실행 환경/고정 설정
- 서빙: vLLM 로컬 3엔드포인트
  - `Qwen2.5-7B-Instruct` (port 8000)
  - `Qwen2.5-Coder-7B-Instruct` (port 8001)
  - `Qwen2.5-Math-7B-Instruct` (port 8002)
- 데이터: train 64 / calibration 64(mixed) / MBPP test 80 / HumanEval test 80
- 공통: `epochs=1`, `batch_size=2`, `max_agent=3`, `exec_timeout_s=15`, `no_fallback=true`
- seed: `1,2,3`
- 모드 매트릭스
  - `masrouter_balanced` (모든 seed)
  - `masrouter_cheap/premium` (seed1)
  - `wae_static_cheap`, `wae_static_premium`
  - `wae_dynamic`, `wae_dynamic_no_premium`
  - `wae_cheap_first_escalate`
  - `wae_dynamic_control_forced_io_general`

## 3) 운영/안정성 반영 사항
이번 라운드에서 실제 적용/운영된 안정화 요소:
- 오케스트레이터 내구성 강화: 완료 런 자동 skip, 불완전 런 archive 후 재실행, 실패 continue
- 배치 단위 예외 흡수: 배치 실패 시 전체 런 종료 대신 fallback 처리 후 진행
- stale 감시: `status.json` 기반 run watchdog 기록
- 컨텍스트 초과 대응 보강(코드 반영): overflow 시 max_tokens 축소 + 메시지 축약 재시도 로직 추가

장애/복구 이력:
- `round6r1_s1_wae_dynamic_no_premium`: `attempt=1 rc=143` -> 자동 재개 후 완주
- `round6r1_s1_wae_dynamic`: `attempt=1 rc=143` -> 자동 재개 후 완주
- 기존 불완전 런 `round6r1_s1_masrouter_premium`은 archive(`..._failed_014618`) 후 재실행 완주

## 4) 시드별 핵심 결과
### Seed 1
- masrouter_balanced: MBPP `0.7875 / 0.00071352`, HE `0.8125 / 0.00133773`
- wae_dynamic: MBPP `0.7875 / 0.00027931`, HE `0.9000 / 0.00037958`
- wae_static_cheap: MBPP `0.8875 / 0.00027205`, HE `0.8500 / 0.00043303`
- cheap_first_escalate: MBPP `0.8625 / 0.00046099`, HE `0.9125 / 0.00065989`
- compare verdict: MBPP FAIL(지배됨), HE FAIL(iso-cost +0.0125 < +0.03)

### Seed 2
- masrouter_balanced: MBPP `0.8250 / 0.00075927`, HE `0.8375 / 0.00141073`
- wae_dynamic: MBPP `0.7875 / 0.00043361`, HE `0.8875 / 0.00062597`
- wae_static_cheap: MBPP `0.8000 / 0.00038275`, HE `0.8750 / 0.00069114`
- cheap_first_escalate: MBPP `0.8875 / 0.00061752`, HE `0.9000 / 0.00100266`
- compare verdict: MBPP FAIL(지배됨), HE FAIL(iso-cost -0.0250)

### Seed 3
- masrouter_balanced: MBPP `0.7250 / 0.00052673`, HE `0.8375 / 0.00089251`
- wae_dynamic: MBPP `0.7875 / 0.00043258`, HE `0.8875 / 0.00065912`
- wae_static_cheap: MBPP `0.7750 / 0.00044236`, HE `0.8750 / 0.00066464`
- cheap_first_escalate: MBPP `0.8250 / 0.00076030`, HE `0.9125 / 0.00116640`
- compare verdict: MBPP FAIL(지배됨), HE FAIL(iso-cost +0.0125 < +0.03)

## 5) 3-seed 집계(평균)
- `masrouter_balanced`
  - MBPP acc/cost/p50: `0.7792 / 0.00066651 / 17.01s`
  - HE acc/cost/p50: `0.8292 / 0.00121366 / 25.62s`
- `wae_dynamic`
  - MBPP acc/cost/p50: `0.7875 / 0.00038183 / 10.73s`
  - HE acc/cost/p50: `0.8917 / 0.00055489 / 25.12s`
- `wae_static_cheap`
  - MBPP acc/cost/p50: `0.8208 / 0.00036572 / 9.34s`
  - HE acc/cost/p50: `0.8667 / 0.00059627 / 11.90s`
- `wae_cheap_first_escalate`
  - MBPP acc/cost/p50: `0.8583 / 0.00061294 / 11.05s`
  - HE acc/cost/p50: `0.9083 / 0.00094298 / 12.50s`

## 6) 핵심 해석
1. WaE 프레임 효율은 재확인
- `wae_dynamic`은 `masrouter_balanced` 대비 평균적으로
  - MBPP: acc `+0.0083p`, cost `-42.7%`, p50 `-36.9%`
  - HE: acc `+0.0625p`, cost `-54.3%`, p50 `-2.0%`
- 즉, 비용 효율 측면의 시스템 신호는 강함.

2. 그러나 Round 목표(dynamic이 strong baseline 우위)는 미달
- 모든 seed에서 compare verdict FAIL
- 실패 직접 원인
  - MBPP: `dominated_by_baseline`
  - HE: `iso_cost_delta_threshold`(delta 약 +0.0125 또는 음수)

3. 라우팅 본질 효과는 여전히 제한적
- `wae_dynamic` vs `wae_dynamic_control_forced_io_general`의 차이가 제한적/불안정
- 즉, dynamic의 이득 상당 부분이 “routing 자체”보다 프레임/구성 효과일 가능성 지속

## 7) 산출물 경로
- 전체 오케스트레이션 로그: [round6r1_orchestrator.log](/workspace/wae_router_pilot/runs/round6r1_orchestrator.log)
- 실패 로그: [round6r1_failures.log](/workspace/wae_router_pilot/runs/round6r1_failures.log)
- seed별 비교:
  - [s1 compare](/workspace/wae_router_pilot/runs/round6r1_s1_compare_dynamic.md)
  - [s2 compare](/workspace/wae_router_pilot/runs/round6r1_s2_compare_dynamic.md)
  - [s3 compare](/workspace/wae_router_pilot/runs/round6r1_s3_compare_dynamic.md)
- 라운드 요약: [round6r1_summary.md](/workspace/wae_router_pilot/runs/round6r1_summary.md)

## 8) 결론
- Round6r1은 운영 안정성/자동복구/다중 seed 완주 측면에서 목표를 달성.
- 정량적으로 WaE 계열의 비용 효율 우위는 재확인.
- 다만 연구 핵심 주장인 `dynamic routing 우위(강한 baseline 대비)`는 이번 라운드에서 통과하지 못함.
- 다음 라운드는 dynamic 선택 다양성 유도와 hard-case premium 전환 효율 증명이 핵심 과제.

## 9) GPU 자동 해제 적용
- 실행 스크립트 [run_round5_full.sh](/workspace/wae_router_pilot/scripts/run_round5_full.sh)에 `EXIT trap` 기반 자동 GPU 해제 로직을 추가함.
- 기본값:
  - `AUTO_RELEASE_GPU_ON_EXIT=1`
  - `VLLM_RELEASE_PORTS=\"8000 8001 8002\"`
  - `GPU_RELEASE_GRACE_S=8`
- 동작:
  - 실험 스크립트 종료 시 지정 포트의 vLLM 서버를 `TERM -> KILL` 순으로 정리
  - 정상 종료/오류 종료 모두 동일하게 GPU 점유 해제 수행
- 본 라운드 종료 후 수동 검증:
  - vLLM/실험 프로세스 없음
  - `nvidia-smi` 기준 GPU 사용률 0%, 메모리 사용량 GPU당 약 `2 MiB` 수준으로 해제 확인
