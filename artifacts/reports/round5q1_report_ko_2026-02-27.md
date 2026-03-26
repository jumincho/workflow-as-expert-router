# WaE-Router Round5q1 상세 실험 보고서 (2026-02-27, 한국어)

## 0) 한 줄 결론
Round5q1(quick-core, seed=1)에서는 `wae_dynamic`이 MasRouter 대비 비용/지연 효율 신호는 강하게 보였지만, **dominance-first 기준에서 best static/strong baseline(특히 `wae_static_cheap`, `wae_cheap_first_escalate`)을 이기지 못해 핵심 주장(동적 라우팅 우위)은 미통과**입니다.

## 1) 실험 목적과 범위
- 목적: 직전 피드백에서 요구된 항목(재현성 게이트, dominance-first 판정, strong baseline 포함, 라우팅 다양성 확인)을 반영해, Round5 quick-core를 E2E로 실행하고 판정 근거를 확보.
- 실행 일자: 2026-02-27
- 실행 단위:
  - Stage A 재현성 게이트: `round5a_0227_045321`
  - 본 실험: `round5q1_s1` (single seed)
- 스케일(quick-core):
  - `train=12`, `calibration=6`, `test_mbpp=12`, `test_humaneval=12`
  - `epochs=0`, `batch_size=2`, `max_agent=2`, `timeout=12s`
- 서빙/모델:
  - vLLM local endpoints
  - `Qwen2.5-7B-Instruct`, `Qwen2.5-Coder-7B-Instruct`, `Qwen2.5-Math-7B-Instruct`
- 비교군:
  - MasRouter curve: `masrouter_cheap`, `masrouter_balanced`, `masrouter_premium`
  - WaE 계열: `wae_static_cheap`, `wae_static_premium`, `wae_dynamic`, `wae_dynamic_no_premium`, `wae_dynamic_roi_gated`, `wae_dynamic_control_forced_io_general`, `wae_cheap_first_escalate`

## 2) 피드백 반영 체크 (이번 라운드에서 실제 적용된 항목)
1. dominance-first 판정 우선 적용
- compare 결과에서 (cost, acc) 및 (cost, acc, p50) 지배 여부를 먼저 판정하고, 이후 iso-cost(±5%)를 적용.

2. iso-cost tolerance band 적용
- nearest가 아니라 `tolerance_band` 기반 비교 사용.

3. strong baseline 포함
- `wae_cheap_first_escalate`를 고정 baseline으로 포함.

4. Stage A 재현성 게이트 운영
- pass/workflow/retry mismatch와 token/cost mismatch 지표를 산출.

5. 운영 게이트
- endpoint readiness gate + heterogeneity gate 통과 후 실행.

6. 라우팅 분석 및 디버깅 로그
- `routing_analysis.json`, `sample_trace.jsonl`, `premium_debug.jsonl` 산출.
- premium_debug에 `tests_source`, `tests_parse_error`, `final_selection_reason` 포함.

## 3) Stage A 재현성 결과
출처: `round5a_0227_045321_stageA_analysis_v2.md`

### 3.1 핵심 수치
- `wae_static_cheap`
  - n_common=12
  - pass/workflow/retry mismatch=0/0/0
  - output_hash_mismatch=6
  - token_mismatch=9
  - token_rel_diff_avg/p95 = 0.013941 / 0.030338
  - cost_mismatch_rate(>1%/>3%/>5%) = 0.5833 / 0.1667 / 0.0000

- `wae_dynamic_no_premium`
  - n_common=12
  - pass/workflow/retry mismatch=1/0/0
  - output_hash_mismatch=4
  - token_mismatch=9
  - token_rel_diff_avg/p95 = 0.023664 / 0.020710
  - cost_mismatch_rate(>1%/>3%/>5%) = 0.1667 / 0.0833 / 0.0833

- `wae_dynamic`
  - n_common=12
  - pass/workflow/retry mismatch=0/0/0
  - output_hash_mismatch=2
  - token_mismatch=7
  - token_rel_diff_avg/p95 = 0.000564 / 0.001073
  - cost_mismatch_rate(>1%/>3%/>5%) = 0.0000 / 0.0000 / 0.0000

### 3.2 해석
- 이번 quick-core에서 재현성은 `wae_dynamic`이 가장 안정적.
- 단, 모드별 token drift가 남아 있으므로(특히 static/no-premium), 다음 full 라운드에서는 token_rel_diff 기반 게이트를 더 엄격히 유지할 필요가 있음.

## 4) 본 실험 정량 결과 (dominance-first)
출처: `round5q1_s1_compare_dynamic.md`, `round5q1_s1_compare_roi_gated.md`

### 4.1 MBPP (n=12)
| mode | acc | avg_cost | p50 | p95 | calls |
|---|---:|---:|---:|---:|---:|
| masrouter_cheap | 0.9167 | 0.00037076 | 9.583 | 28.125 | 18 |
| masrouter_balanced | 0.8333 | 0.00040810 | 9.709 | 27.179 | 24 |
| masrouter_premium | 0.9167 | 0.00048411 | 14.838 | 20.316 | 30 |
| wae_static_cheap | 1.0000 | 0.00026570 | 7.739 | 13.328 | 24 |
| wae_static_premium | 0.9167 | 0.00034542 | 10.098 | 17.956 | 48 |
| wae_dynamic_no_premium | 0.9167 | 0.00036897 | 12.856 | 20.737 | 26 |
| wae_dynamic_roi_gated | 0.9167 | 0.00026742 | 7.681 | 12.875 | 24 |
| wae_cheap_first_escalate | 1.0000 | 0.00026482 | 6.659 | 16.167 | 27 |
| wae_dynamic | 0.9167 | 0.00026802 | 7.685 | 13.167 | 24 |

`wae_dynamic` 판정:
- iso-cost(±5%) delta_acc = `-0.0833` (ref=`wae_cheap_first_escalate`)
- dominated(cost, acc)=True by `wae_cheap_first_escalate`, `wae_dynamic_roi_gated`, `wae_static_cheap`
- final verdict = `False (dominated_by_baseline)`

### 4.2 HumanEval (n=12)
| mode | acc | avg_cost | p50 | p95 | calls |
|---|---:|---:|---:|---:|---:|
| masrouter_cheap | 0.9167 | 0.00059043 | 9.846 | 48.631 | 19 |
| masrouter_balanced | 0.6667 | 0.00128715 | 33.916 | 80.861 | 24 |
| masrouter_premium | 0.7500 | 0.00112035 | 24.633 | 53.825 | 30 |
| wae_static_cheap | 0.9167 | 0.00027445 | 5.842 | 10.221 | 24 |
| wae_static_premium | 0.9167 | 0.00083182 | 14.291 | 25.403 | 48 |
| wae_dynamic_no_premium | 0.9167 | 0.00065481 | 20.972 | 42.297 | 31 |
| wae_dynamic_roi_gated | 0.9167 | 0.00027572 | 5.902 | 10.160 | 24 |
| wae_cheap_first_escalate | 1.0000 | 0.00048628 | 7.921 | 18.360 | 30 |
| wae_dynamic | 0.9167 | 0.00027562 | 5.906 | 10.172 | 24 |

`wae_dynamic` 판정:
- iso-cost(±5%) delta_acc = `+0.0000` (ref=`wae_static_cheap`)
- dominated(cost, acc)=True by `wae_static_cheap`
- final verdict = `False (dominated_by_baseline)`

### 4.3 `wae_dynamic_roi_gated` 판정
- MBPP: dominated → FAIL
- HumanEval: dominated → FAIL
- 결론: 이번 quick-core에서는 ROI-gated 변형이 개선 신호를 만들지 못함.

## 5) 관찰 포인트 (피드백 관점 핵심)

### 5.1 WaE 파이프라인 vs MasRouter 효율 신호
`wae_dynamic` vs `masrouter_balanced`:
- MBPP: acc +0.0834p, cost -34.32%, p50 -20.85%
- HumanEval: acc +0.2500p, cost -78.59%, p50 -82.59%

해석:
- 이번 스케일에서는 WaE 실행 프레임의 비용/지연 효율 신호가 매우 강함.
- 다만 이는 “dynamic routing 자체의 이득”과는 분리해서 해석해야 함.

### 5.2 라우팅 다양성 붕괴 여부
출처: 각 run의 `metrics/routing_analysis.json`

- `wae_dynamic`
  - MBPP top: `wf::wf_io_general` 12/12
  - HumanEval top: `wf::wf_io_general` 12/12

- `wae_dynamic_roi_gated`
  - MBPP/HumanEval 모두 `wf::wf_io_general` 12/12

- `wae_dynamic_no_premium`
  - MBPP: `wf_io_general` 10, `wf_refine2_coder` 2
  - HumanEval: `wf_refine2_coder` 7, `wf_io_general` 5

해석:
- `wae_dynamic`/`roi_gated`는 사실상 단일 워크플로우로 수렴.
- 즉 이번 라운드의 dynamic 성능은 “상황별 선택 능력”보다 “프레임 자체의 성능” 영향이 큼.

### 5.3 strong baseline(cheap-first-escalate)의 강도
- MBPP: acc 1.0000, avg_cost 0.00026482
- HumanEval: acc 1.0000, avg_cost 0.00048628
- 라우팅 이벤트 기준 escalation 발생:
  - MBPP total_events=13 (샘플 12) → +1회
  - HumanEval total_events=14 (샘플 12) → +2회
  - 합계 +3회/24샘플 (약 12.5%)

해석:
- 낮은 escalation 빈도로 정확도 1.0을 달성해 매우 강한 실전 baseline으로 확인됨.

### 5.4 premium debug (test_select 품질)
로그: `wae_static_premium`, `wae_cheap_first_escalate`, `wae_dynamic`의 `premium_debug.jsonl`

- `wae_static_premium` (66건)
  - first_test_pass=30
  - syntax_filter_or_length=36 (54.5%)
  - tests_present=true 66, false 0

- `wae_cheap_first_escalate` (11건, eval)
  - first_test_pass=5
  - syntax_filter_or_length=6 (54.5%)
  - tests_present=true 11, false 0

- `wae_dynamic` (12건, mostly non-eval diagnostics)
  - first_test_pass=4
  - syntax_filter_or_length=2
  - no_tests_fallback_io=6
  - tests_present=false 6

해석:
- premium 경로에서 `syntax_filter_or_length` 비중이 여전히 높아 “비용→품질 전환”이 안정적이지 않음.
- `wae_dynamic` 로그의 `phase=unknown`/`tests_present=false` 건은 eval 선택과 분리된 경로에서 발생한 것으로 보이며, 다음 라운드에서 trace 구분을 더 명확히 해야 함.

## 6) 종합 판정
1. 확실히 확인된 것
- 운영/게이트/비교 파이프라인은 안정적으로 완주.
- WaE 계열의 비용/지연 효율은 MasRouter curve 대비 강함.

2. 이번 라운드에서 미확인(또는 실패)한 것
- `wae_dynamic`이 best static / cheap-first-escalate를 dominance-first 기준으로 이긴다는 주장: 미통과.
- 동적 라우팅의 본질 효과(다양한 workflow를 상황별로 선택): 미약.

3. 현재 라운드의 공식 결론
- Quick-core 기준으로는 “WaE 프레임 효율”은 긍정 신호.
- 그러나 “dynamic routing 우위”는 아직 근거 부족.

## 7) 다음 라운드 실행 우선순위 제안 (피드백 대응용)
1. 스케일 복원(필수)
- `train=64`, `calibration=64`, `test_mbpp=80`, `test_humaneval=80`, `seed=3`.

2. strong baseline 고정
- `wae_cheap_first_escalate`를 모든 비교에 필수 포함.
- budget-best-under-budget 표를 기본 산출물로 고정.

3. 라우팅 다양성 강제 실험 추가
- `dynamic_control_forced_io`를 유지해 라우팅 효과 분리.
- entropy/규칙형 hard-case gating으로 최소 2-workflow 사용을 유도.

4. premium 안정화
- syntax repair 1회 루프와 selection 기준 강화(pass count 기반) 적용.
- `tests_present=false` 원인을 split별로 분리 기록.

5. Stage A 강화
- token_rel_diff(p95) 기준으로 cost 판정 신뢰 플래그 자동화.

## 8) 주요 산출물 경로
- Stage A 분석
  - `/workspace/wae_router_pilot/runs/round5a_0227_045321_stageA_analysis_v2.json`
  - `/workspace/wae_router_pilot/runs/round5a_0227_045321_stageA_analysis_v2.md`

- 비교 결과
  - `/workspace/wae_router_pilot/runs/round5q1_s1_compare_dynamic.json`
  - `/workspace/wae_router_pilot/runs/round5q1_s1_compare_dynamic.md`
  - `/workspace/wae_router_pilot/runs/round5q1_s1_compare_roi_gated.json`
  - `/workspace/wae_router_pilot/runs/round5q1_s1_compare_roi_gated.md`

- 라우팅/디버그 예시
  - `/workspace/wae_router_pilot/runs/round5q1_s1_wae_dynamic/metrics/routing_analysis.json`
  - `/workspace/wae_router_pilot/runs/round5q1_s1_wae_dynamic_no_premium/metrics/routing_analysis.json`
  - `/workspace/wae_router_pilot/runs/round5q1_s1_wae_cheap_first_escalate/metrics/routing_analysis.json`
  - `/workspace/wae_router_pilot/runs/round5q1_s1_wae_static_premium/logs/premium_debug.jsonl`

