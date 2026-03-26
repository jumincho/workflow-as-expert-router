# Round5q1 피드백 반영 현황 (2026-02-27)

## 반영 완료
- `run_pilot.py`
  - `--workflow_entropy_reg` 추가 (dynamic 모드 entropy 정규화)
  - `--deterministic_router_components` 추가 (WaE 라우팅 컴포넌트 argmax 강제)
  - phase 라벨 세분화: `train`, `calibration`, `eval_mbpp`, `eval_humaneval`
  - sample_trace에 `task`, `collab` 필드 추가
  - deterministic 모드에서 torch deterministic 알고리즘 활성화

- `workflow_router_patch.py`
  - deterministic collab/role routing 경로 추가
  - workflow routing entropy 추적(`last_workflow_entropy`) 추가

- `workflow_llm.py`
  - premium workflow에 syntax repair 1회 루프 추가
  - 테스트 통과 수 기반 선택(`max_test_pass_count`) 추가
  - premium_debug 필드 강화(`repaired_once`, `syntax_repair_enabled`, pass count)

- `offline_pareto_builder.py`
  - HumanEval calibration prompt에 테스트 블록 주입
  - calibration 단계 premium 로그에 `phase/split` 명시(`calibration_*`)

- `scripts/run_round5_stageA.sh`
  - deterministic router components 옵션 연결
  - Stage A 분석에 role/task/collab/prompt hash mismatch 추가
  - mode별 token p95 기준 + stage gate pass 판정 추가
  - pass mismatch workflow breakdown 추가

- `scripts/run_round5_full.sh`
  - 기본 prefix를 `round6`로 상향
  - dynamic entropy 정규화 옵션 연결
  - ROI-gated를 기본 core에서 분리(진단 옵션화)
  - 9-way 핵심 매트릭스 중심으로 실행 구조 정리

## 미실행 항목
- 엔드포인트(8000/8001/8002) 오프라인 상태라 실제 재실행 검증은 아직 수행하지 않음.
- 다음 실행 시 Stage A(소규모) -> Full(64/64/80/80, seed=3) 순으로 즉시 검증 필요.
