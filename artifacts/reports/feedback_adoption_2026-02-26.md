# 피드백 반영 내역 (2026-02-26)

## 요약
- 결론: 제안된 피드백은 대부분 타당하며, 다음 라운드 실험 신뢰도를 높이는 핵심 항목(P0/P1 일부)을 코드/스크립트 레벨로 반영했습니다.
- 반영 범위: endpoint fail-fast 게이트, heterogeneity 강제 옵션, iso-cost 판정 강화, premium 검증형 workflow 추가, mixed calibration 지원, seed-sweep 실행 템플릿, 라우팅 분석 산출물.

## 항목별 반영 상태

1. `3x7B hetero endpoint + fallback 금지` (타당, 반영 완료)
- 반영:
  - `run_pilot.py`에 `--no_fallback`, `--require_heterogeneous_endpoints`, readiness/warmup 게이트 추가
  - endpoint 게이트 결과 저장: `metrics/endpoint_gate.json`
  - 이질성 요약 저장: `metrics/endpoint_heterogeneity.json`
  - 미충족 시 fail-fast (`RuntimeError`)
- 관련 파일:
  - `/workspace/wae_router_pilot/src/run_pilot.py`
  - `/workspace/wae_router_pilot/src/workflow_router_patch.py`

2. `iso-cost 판정 로직 강화` (타당, 반영 완료)
- 기존: nearest baseline 1점 비교
- 변경:
  - tolerance-band 비교(기본 ±5%)
  - tolerance-band 불가 시 linear interpolation
  - dominance(cost,acc) / dominance(cost,acc,p50lat) 동시 리포트
- 관련 파일:
  - `/workspace/wae_router_pilot/src/compare_runs.py`

3. `샘플/seed 확대 기반 실행 표준화` (타당, 부분 반영 완료)
- 반영:
  - round2 seed sweep 스크립트 추가 (`SEEDS="1 2 3"` 기본)
  - default config를 다음 라운드 친화적으로 조정 (`calibration_size=64`, `calibration_mix=mixed`)
- 관련 파일:
  - `/workspace/wae_router_pilot/scripts/run_round2_seed_sweep.sh`
  - `/workspace/wae_router_pilot/config/experiment.yaml`

4. `routing 분석 산출물 강화` (타당, 반영 완료)
- 반영:
  - role x workflow 선택 카운트
  - workflow별 conditional success `P(pass|wf)`
  - 난이도(easy/hard, query length median proxy) 버킷별 성능/비용/선택분포
  - case study 10개
  - endpoint usage 분포
- 저장 파일:
  - `metrics/routing_analysis.json`
- 관련 파일:
  - `/workspace/wae_router_pilot/src/run_pilot.py`
  - `/workspace/wae_router_pilot/src/workflow_router_patch.py` (selected roles 반환)

5. `premium workflow 품질-비용 전환 설계` (타당, 반영 완료)
- 반영:
  - `wf_gen3_test_select_coder` 추가
  - 내부 동작: 최대 3개 생성, MBPP inline tests 존재 시 pass candidate 즉시 선택(early exit)
- 관련 파일:
  - `/workspace/wae_router_pilot/src/workflow_profile.py`
  - `/workspace/wae_router_pilot/src/workflow_llm.py`

6. `Pareto calibration 대표성 개선` (타당, 반영 완료)
- 반영:
  - `--calibration_mix {mbpp,humaneval,mixed}` 지원
  - 라이브러리 JSON에 source composition 기록
- 관련 파일:
  - `/workspace/wae_router_pilot/src/offline_pareto_builder.py`
  - `/workspace/wae_router_pilot/src/run_pilot.py`

7. `라운드 전환 시 파일 정리` (타당, 기존 반영 + 유지)
- 유지/적용:
  - cleanup 스크립트 이미 적용 중
  - 이번 세션에서도 임시 검증 산출물 정리 완료
- 관련 파일:
  - `/workspace/wae_router_pilot/scripts/cleanup_round.sh`

## 추가로 반영한 운영 개선
- 3x7B 전용 endpoint 설정 파일 명시 추가
  - `/workspace/wae_router_pilot/config/model_endpoints_3x7b.yaml`
- 7B 실행 스크립트에 strict gate 옵션 반영
  - `/workspace/wae_router_pilot/run_expanded_7b.sh`
- README에 strict gate 및 seed sweep 사용법 추가
  - `/workspace/wae_router_pilot/README.md`

## 다음 실행 제안 (즉시 가능)
- 선행 조건: 3개 vLLM endpoint warmup 완료
- 실행:
  - `bash /workspace/wae_router_pilot/scripts/run_round2_seed_sweep.sh`
- 결과 확인:
  - seed별 `round2_s{seed}_compare.{json,md}`
  - 각 run의 `metrics/endpoint_gate.json`, `metrics/endpoint_heterogeneity.json`, `metrics/routing_analysis.json`
