# WaE-Router Round7r1 실행 결과 상세 보고서 (중간결과)

- 작성 시각: `2026-03-02 02:41:33 UTC`
- 작성자: `Codex`
- 실행 루트: `/workspace/wae_router_pilot`

## 1) 이번 실행 범위와 상태
- 목표: Round7 후속 실험(8-way core + hardcase_gate + tau sweep) end-to-end 수행
- 실제 상태: **Round7 본 매트릭스 실행 전, Stage A(재현성 게이트)까지 완료**
- Round7 오케스트레이터 로그: `/workspace/wae_router_pilot/runs/round7r1_orchestrator.log`
- Stage A 산출물:
  - `/workspace/wae_router_pilot/runs/round5a_0301_142814_stageA_analysis_v2.json`
  - `/workspace/wae_router_pilot/runs/round5a_0301_142814_stageA_analysis_v2.md`
- Round7 seed 실행 산출물(`round7r1_s*_...`)은 생성되지 않음

## 2) 사전 원인분해(P0) 결과 요약 (Round6r1 기반)
- MBPP confusion(dynamic vs static_cheap): `{'both_pass': 180, 'both_fail': 34, 'dynamic_only_pass': 9, 'static_only_pass': 17}`
- Oracle gain vs cheap (avg): MBPP `+0.0667`, HumanEval `+0.0292`
- 해석:
  - dynamic 선택 여지가 있는 headroom은 존재(특히 MBPP 평균 +6.67%p 수준)
  - 다만 Round6r1에서는 선택 다양성 붕괴(`wf_io_general` 수렴)로 실제 이득 전환 실패
- 관련 파일: `/workspace/wae_router_pilot/runs/round6r1_p0_analysis.md`

## 3) Stage A(재현성 게이트) 결과
### wae_static_cheap
- n_common: `24`
- pass/workflow/retry mismatch: `0` / `0` / `0`
- token_rel_diff p95 (target): `0.063693` (`0.030000`)
- stageA_gate_pass: `False`
### wae_dynamic_no_premium
- n_common: `24`
- pass/workflow/retry mismatch: `1` / `0` / `0`
- token_rel_diff p95 (target): `0.088830` (`0.030000`)
- stageA_gate_pass: `False`
### wae_dynamic
- n_common: `24`
- pass/workflow/retry mismatch: `1` / `0` / `0`
- token_rel_diff p95 (target): `0.047460` (`0.005000`)
- stageA_gate_pass: `False`
- 종합: **3개 모드 모두 Stage A gate FAIL**
  - 주요 원인: token drift p95 초과 + 일부 모드 pass mismatch 존재

## 4) 이번 라운드에서 실제 반영한 코드 변경
- `wae_dynamic_hardcase_gate` 모드 추가(cheap 실패확률 기반 분기):
  - `/workspace/wae_router_pilot/src/run_pilot.py`
- hardcase predictor 학습/저장(`hardcase_predictor.json`) 및 tau 파라미터화:
  - `/workspace/wae_router_pilot/src/run_pilot.py`
- latency 분해 지표 추가(`router_overhead`, `llm_infer`, `test_exec`):
  - `/workspace/wae_router_pilot/src/run_pilot.py`
  - `/workspace/wae_router_pilot/src/compare_runs.py`
- sample_trace 확장(`hardcase_fail_prob`, `hardcase_gate_choice`, 호출/재시도 메타):
  - `/workspace/wae_router_pilot/src/run_pilot.py`
  - `/workspace/wae_router_pilot/src/workflow_router_patch.py`
- P0 자동 분석 스크립트 추가:
  - `/workspace/wae_router_pilot/src/analyze_round_p0.py`
- Round7 오케스트레이터 스크립트 추가/수정:
  - `/workspace/wae_router_pilot/scripts/run_round7_feedback.sh`

## 5) Round7 본 실험이 완료되지 않은 이유(현재 확인 범위)
- 로그상 Stage A 종료 후(`STAGE_A_DONE`) 오케스트레이터가 종료됨
- `round7r1_failures.log`는 생성되지 않아 명시적 실패 스텝은 기록되지 않음
- 따라서 본 매트릭스(`round7r1_s1~s3_*`) 실행 결과는 부재

## 6) GPU/서버 상태
- 현 시점 GPU 점유: 4개 GPU 모두 0% (메모리 2MiB 수준)
- Stage A 종료 시 오케스트레이터 cleanup에서 vLLM 서버 자동 해제 확인

## 7) 결론 및 재개 권장 순서
- 결론: 이번 실행은 **Stage A/P0까지 유효**, Round7 본 실험 결과는 아직 없음
- 즉시 재개 시 권장:
  1. `RUN_STAGE_A=0`로 본 매트릭스부터 실행 (이미 Stage A 결과 존재)
  2. seed=1 우선 완주 후 seed=2,3 확장
  3. 완료 후 `compare_runs` + 통합 리포트 재생성

## 8) 참고 파일
- 오케스트레이터 로그: `/workspace/wae_router_pilot/runs/round7r1_orchestrator.log`
- Stage A 리포트: `/workspace/wae_router_pilot/runs/round5a_0301_142814_stageA_analysis_v2.md`
- P0 분석 리포트: `/workspace/wae_router_pilot/runs/round6r1_p0_analysis.md`