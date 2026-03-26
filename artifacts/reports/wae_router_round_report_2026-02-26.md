# WaE-Router Pilot 상세 실험 보고서 (Round Report)

- 작성일: 2026-02-26 (UTC)
- 작업 경로: `/workspace/wae_router_pilot`
- 목적: MasRouter 대비 WaE-Router 확장 방법론의 가능성 신호를 6시간 내 E2E로 검증
- 성공 기준(사전 정의): iso-cost 기준 `+3%p` 이상 정확도 향상

## 1. 실험 목표와 질문

### 1.1 목표
- `MasRouter-original` 대비 `WaE-Router(Workflow-as-Expert)`가 비용-성능 곡선에서 개선 신호를 보이는지 확인
- 단일 고정 워크플로우(static) 대비 동적 라우팅(dynamic)의 이점 확인

### 1.2 핵심 질문
- Q1: 동적 workflow routing이 모델 라우팅 대비 iso-cost 성능 향상을 보이는가?
- Q2: static-cheap/static-premium 대비 dynamic이 Pareto 측면에서 우세한가?
- Q3: MBPP에서 얻은 신호가 HumanEval에도 일반화되는가?

## 2. 구현/실험 구성

### 2.1 코드 구조
- Upstream 의존(수정 없음): `/workspace/masrouter`
- 확장 구현: `/workspace/wae_router_pilot/src`
  - `workflow_llm.py`: workflow를 LLM-like interface로 래핑
  - `workflow_profile.py`: workflow expert 정의(io/refine2/sc3/critique_refine)
  - `workflow_router_patch.py`: endpoint patch + WaERouter
  - `offline_pareto_builder.py`: role-conditioned Pareto library 생성
  - `run_pilot.py`: 학습/평가 실행 CLI
  - `monitor.py`: stage/heartbeat/GPU 상태 기록
  - `compare_runs.py`: 4-way 비교/iso-cost 판정/플롯 생성

### 2.2 워크플로우 전문가 정의
- `wf_io_general` (cheap)
- `wf_refine2_coder` (balanced)
- `wf_sc3_coder` (premium)
- `wf_critique_refine_math` (premium)

각 workflow는 `base_model + method + params + allowed_roles + budget_tier`로 정의됨.

### 2.3 데이터/지표
- 데이터: MBPP, HumanEval
- 공통 지표: `accuracy_or_pass1`, `avg_cost`, `latency(p50/p95)`, `call_count_est`
- 판정: WaE-dynamic vs nearest iso-cost baseline의 `delta acc`

### 2.4 모델 서빙
- 원 계획: 3x7B (`Qwen2.5-7B`, `Coder-7B`, `Math-7B`)를 vLLM 다중 포트 서빙
- 실제 파일럿: 7B 다운로드/기동 지연으로 `Qwen2.5-0.5B-Instruct` 단일 endpoint로 fallback
  - endpoint config: `/workspace/wae_router_pilot/config/model_endpoints_smoke.yaml`

## 3. 실험 라운드 진행 내역

### 3.1 라운드 A: 초기 소규모 6-sample 파일럿
- 주 목적: 파이프라인 완주/실행 안정성 확인
- 대표 비교 리포트:
  - `/workspace/wae_router_pilot/runs/pilot6h_compare_cli.md`
  - `/workspace/wae_router_pilot/runs/pilot6h_comparison_safe_report.md`
- 관찰:
  - 샘플 수가 매우 작아 분산이 큼
  - HumanEval 신호가 과대/과소로 흔들릴 가능성 큼

### 3.2 라운드 B: 확장 24-sample 재실행 (핵심)
- 실행 묶음(run stamp): `exp24_081044`
- 비교군:
  - `pilot_exp24_masrouter_exp24_081044`
  - `pilot_exp24_wae_dynamic_exp24_081044`
  - `pilot_exp24_wae_static_cheap_exp24_081044`
  - `pilot_exp24_wae_static_premium_exp24_081044`
- 비교 리포트:
  - `/workspace/wae_router_pilot/runs/pilot_exp24_compare_exp24_081044.md`

## 4. 라운드 B 정량 결과

### 4.1 MBPP (n=24)
| mode | acc | avg_cost | p50 | p95 | calls |
|---|---:|---:|---:|---:|---:|
| masrouter | 0.2083 | 0.00019239 | 1.778 | 2.953 | 72 |
| wae_static_cheap | 0.2500 | 0.00018798 | 1.769 | 3.269 | 72 |
| wae_static_premium | 0.3333 | 0.00063908 | 6.664 | 10.658 | 168 |
| wae_dynamic | 0.2917 | 0.00022742 | 2.593 | 4.037 | 81 |

- nearest iso-cost baseline: `masrouter`
- delta acc(dynamic - nearest): `+0.0833`
- 성공 기준(+0.03): `PASS`

### 4.2 HumanEval (n=24)
| mode | acc | avg_cost | p50 | p95 | calls |
|---|---:|---:|---:|---:|---:|
| masrouter | 0.2917 | 0.00020618 | 2.249 | 3.011 | 72 |
| wae_static_cheap | 0.2500 | 0.00020771 | 2.007 | 3.300 | 72 |
| wae_static_premium | 0.2500 | 0.00069422 | 6.561 | 13.225 | 168 |
| wae_dynamic | 0.2500 | 0.00026824 | 3.340 | 3.952 | 81 |

- nearest iso-cost baseline: `wae_static_cheap`
- delta acc(dynamic - nearest): `+0.0000`
- 성공 기준(+0.03): `FAIL`

### 4.3 실행 시간(heartbeat 기준)
- masrouter: 약 `154.5s`
- wae_dynamic: 약 `452.3s`
- wae_static_cheap: 약 `354.2s`
- wae_static_premium: 약 `734.5s`

## 5. 해석 및 결론

### 5.1 결론 요약
- **가능성 신호 있음**: MBPP에서는 WaE-dynamic이 iso-cost 기준에서 유의미한 개선(`+8.33%p`)을 보임
- **일반화 미확인**: HumanEval에서는 동일 개선이 재현되지 않음
- **비용/지연 trade-off 확인**: premium/sc3 계열은 성능 잠재력은 있으나 cost/latency가 크게 증가

### 5.2 원인 가설
- 가설 1: 단일 0.5B endpoint를 general/coder/math 모두 공유해 역할 이질성이 약화됨
- 가설 2: calibration(24) 기반 Pareto library가 HumanEval 분포를 충분히 대표하지 못함
- 가설 3: dynamic policy가 MBPP 편향으로 학습되어 HumanEval에서 overfit 성향 발생

## 6. 운영 이슈 및 대응

- 이슈: 3x7B 동시 로컬 서빙이 실험 시간창 내 안정적으로 API-ready까지 도달하지 못함
- 대응: 0.5B endpoint fallback으로 파이프라인 E2E 완주 우선
- 이슈: long-running 실험 후 GPU 잔여 점유 위험
- 대응: 실험 종료 시 vLLM/EngineCore 강제 종료 후 `nvidia-smi`로 확인

## 7. 라운드 전환 파일 정리(이번에 실제 수행)

정리 원칙: 재현 핵심 산출물(metrics/summary/report/compare)은 보존하고, 스모크/임시/대용량 불필요 파일 제거.

### 7.1 추가된 정리 스크립트
- `/workspace/wae_router_pilot/scripts/cleanup_round.sh`
- 기능:
  - `runs/smoke_*` 삭제
  - stale `.pid` 삭제
  - 빈 `vllm_logs/*.log` 및 빈 디렉터리 삭제
  - 구 라운드 체크포인트(`pilot6h_*/*.pth`) 제거

### 7.2 실제 정리 결과
- 실행: `cleanup_round.sh --execute`
- 용량 변화: `976M -> 356M` (약 `620M` 절감)
- 보존: `exp24` 핵심 런/비교 산출물 및 모든 metrics/report

## 8. 후속 라운드 권장안

1. 3x7B 이질 endpoint 완성 후 동일 프로토콜 재실행
- 목표: WaE의 핵심 가정(heterogeneous experts) 검증

2. 데이터 확대
- 최소 `train/test=64/80`로 복원, seed 3회 반복
- split별 신뢰구간/분산 보고

3. routing 분석 강화
- role별 workflow 선택 분포
- easy/hard 버킷별 비용-성능 변화

4. 판정 고도화
- iso-cost +3% 외에 latency budget 조건 포함
- MBPP/HumanEval 공동 성공 기준 정의

## 9. 재현 커맨드

```bash
cd /workspace/wae_router_pilot
source .venv/bin/activate

# 24-sample 4-way
PYTHONPATH=/workspace/wae_router_pilot:/workspace/masrouter \
python -m src.run_pilot --mode masrouter --train_samples 24 --test_samples_mbpp 24 --test_samples_humaneval 24 --epochs 1 --batch_size 2 --max_agent 3 --calibration_size 24 --exec_timeout_s 10 --model_endpoints /workspace/wae_router_pilot/config/model_endpoints_smoke.yaml --run_id pilot_exp24_masrouter_exp24_081044

PYTHONPATH=/workspace/wae_router_pilot:/workspace/masrouter \
python -m src.compare_runs \
  --masrouter_run /workspace/wae_router_pilot/runs/pilot_exp24_masrouter_exp24_081044 \
  --wae_dynamic_run /workspace/wae_router_pilot/runs/pilot_exp24_wae_dynamic_exp24_081044 \
  --wae_static_cheap_run /workspace/wae_router_pilot/runs/pilot_exp24_wae_static_cheap_exp24_081044 \
  --wae_static_premium_run /workspace/wae_router_pilot/runs/pilot_exp24_wae_static_premium_exp24_081044 \
  --out_prefix /workspace/wae_router_pilot/runs/pilot_exp24_compare_exp24_081044

# 라운드 전환 정리
/workspace/wae_router_pilot/scripts/cleanup_round.sh --execute
```

---

부록: 주요 비교 산출물
- `/workspace/wae_router_pilot/runs/pilot_exp24_compare_exp24_081044.md`
- `/workspace/wae_router_pilot/runs/pilot_exp24_compare_exp24_081044.json`
- `/workspace/wae_router_pilot/runs/pilot_exp24_compare_exp24_081044_cost_acc_mbpp_eval.png`
- `/workspace/wae_router_pilot/runs/pilot_exp24_compare_exp24_081044_cost_acc_humaneval_eval.png`
