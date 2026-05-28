<div align="center">

# workflow-as-expert-router

**모델만 라우팅하지 말고 흐름(workflow)까지 라우팅하자**
**Routing entire workflows, not just models, as the expert unit**

![Status](https://img.shields.io/badge/status-dormant-lightgrey)
![Language](https://img.shields.io/badge/language-Python-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)
![Closure](https://img.shields.io/badge/closure-2026--03-blue)

**한국어** · [English](#english) · [中文](./README.zh-CN.md)

</div>

> 🧊 **휴면(dormant) 중인 연구 파일럿입니다.**

## 무엇을 보려던 연구였나

요즘 LLM 시스템은 종종 "어떤 모델에게 이 질문을 넘길까?" 를 자동으로 고릅니다. 이 프로젝트는 한 걸음 더 들어가, **모델만 고르는 게 아니라 "어떤 흐름(workflow)으로 풀게 할까"까지 같이 고르자** 는 아이디어를 검증한 연구입니다.

여기서 흐름이란 예를 들면 이런 것들입니다.

- 한 번에 답을 만들게 한다
- 답을 만들고 다시 다듬게 한다
- 후보를 여러 개 만들어 비교하게 한다
- 비평 후 다시 쓰게 한다

비교 대상은 흐름 단위 라우팅이 없는 모델-단위 라우팅 기존 방식, 그리고 흐름은 쓰되 항상 같은 흐름만 쓰는 단순 baseline 들입니다. 평가는 코드 생성 벤치마크(`MBPP`, `HumanEval`) 위에서, 정확도·비용·지연시간을 함께 봤습니다.

## 무엇을 알아냈나

- **흐름 단위로 expert 를 정의하고 라우팅한다는 아이디어 자체는 효과가 있었습니다.** 기존 모델-단위 라우팅 대비 같은 정확도에서 비용과 지연이 의미 있게 줄었고, 이 결과는 여러 라운드에서 반복됐습니다.
- **그런데 "흐름을 동적으로 고르는 것 자체가 핵심이다" 라는 더 강한 주장은 끝까지 입증하지 못했습니다.** 단순히 "값싼 흐름 하나만 쓰자" 같은 정적 baseline 이 의외로 강했고, 동적 선택만의 추가 이득이 안정적으로 잡히지 않았습니다.
- 즉 **시스템 패턴으로서는 살아남았지만, "동적 라우팅이 결정적이다" 라는 헤드라인 주장으로는 못 닫혔습니다.**

자세한 숫자가 궁금하시면:

- 🇰🇷 [`closure_reports/project_closure_report_ko_20260327.md`](closure_reports/project_closure_report_ko_20260327.md)
- 🇬🇧 [`closure_reports/project_closure_report_20260327.md`](closure_reports/project_closure_report_20260327.md)

## 왜 잠시 멈춰 두는가

원래 노렸던 결론(동적 흐름 라우팅이 정적 baseline 을 깔끔하게 이긴다) 이 깔끔하게 서지 않았고, 마지막 라운드도 끝까지 돌리지 못했습니다. 다시 시작할 때는 "조금만 더 튜닝하면 동적이 이길 것" 이라는 가정 대신, "흐름 단위 시스템 효율" 을 메인 framing 으로 두고 동적 라우팅은 부수 가설로 다루는 편이 정직합니다. 그 방향이 명확해지는 자극이 오면 다시 열 가치가 있습니다.

## 다시 들여다볼 때는 어디부터

- 📖 [`GLOSSARY.md`](GLOSSARY.md) — 코드와 종료 보고서에 등장하는 내부 용어(`MAR`/MasRouter, `wf_io_general` 등 워크플로우 id, `round7r2`/`expanded_7b` 같은 라운드명, `status/` 스냅샷 포맷, 환경변수, 비용 일치 비교 등)를 일반어로 풀어놓은 사전
- [`docs/EXPERIMENT_OVERVIEW.md`](docs/EXPERIMENT_OVERVIEW.md) — 이 연구가 비교한 것이 정확히 무엇인지
- [`docs/HANDOFF_RUNBOOK.md`](docs/HANDOFF_RUNBOOK.md) — 다시 돌리는 절차
- [`docs/KNOWN_ISSUES_AND_FIXES.md`](docs/KNOWN_ISSUES_AND_FIXES.md) — 반복적으로 빠지던 함정들
- [`docs/PUBLISHING_GUIDE.md`](docs/PUBLISHING_GUIDE.md) — 좁아진 framing 으로 글을 써 본다면
- [`status/`](status/) — 마지막 진행 상태 스냅샷
- [`artifacts/reports/`](artifacts/reports/) — 라운드별 보고서 (한국어판 포함)

## 코드 어디에 뭐가 있나

| 파일 | 하는 일 |
|---|---|
| [`src/run_pilot.py`](src/run_pilot.py) | 메인 실험 러너 |
| [`src/workflow_router_patch.py`](src/workflow_router_patch.py) | 흐름 단위로 라우팅하도록 라우터에 끼우는 패치 |
| [`src/workflow_llm.py`](src/workflow_llm.py) | 흐름을 마치 한 개의 LLM 처럼 감싸는 래퍼 |
| [`src/workflow_profile.py`](src/workflow_profile.py) | 어떤 흐름들이 후보인지 정의 |
| [`src/offline_pareto_builder.py`](src/offline_pareto_builder.py) | 오프라인에서 후보 흐름의 비용-품질 Pareto 를 미리 만든다 |
| [`src/compare_runs.py`](src/compare_runs.py) | 실험 결과를 같은 비용 조건에서 비교하고 보고서를 만든다 |
| [`src/monitor.py`](src/monitor.py) | 실행 상태 모니터링 |

## 폴더 지도

```
.
├── src/                       라우터 패치 / 흐름 래퍼 / 러너 / 비교 / 모니터
├── config/                    실험·엔드포인트 설정
├── scripts/                   라운드별 실행 / 재개 스크립트
├── docs/                      개요 / 런북 / 함정 / 글쓰기 가이드
├── status/                    마지막 진행 상태
├── artifacts/reports/         라운드 보고서 (한국어 / 영문)
├── artifacts/round7r2/        마지막 라운드의 비교 산출물 (일부)
├── artifacts/snapshots/       진행 스냅샷 JSON
├── closure_reports/           종료 보고서 (한국어 / 영문)
├── GLOSSARY.md                내부 용어 사전
└── launch_vllm.sh / setup_env.sh / stop_vllm.sh / run_expanded_7b.sh
```

엔드포인트 설정은 자격증명을 직접 박지 않고 환경변수로만 주입하도록 되어 있습니다.

## 환경

```bash
bash setup_env.sh
bash launch_vllm.sh   # 별도 서빙 환경이 떠 있어야 합니다
```

원본 실행 환경의 절대 경로(`/workspace/wae_router_pilot`, `/workspace/masrouter`)는
다음 환경변수로 덮어쓸 수 있습니다.

| 환경변수 | 의미 | 기본값 |
| --- | --- | --- |
| `WAE_ROUTER_PILOT_ROOT` | 이 저장소의 루트 (run 산출물 위치) | `/workspace/wae_router_pilot` (또는 스크립트 위치 기준 자동) |
| `MASROUTER_PATH` | 외부 MasRouter(`MAR`) 패키지 체크아웃 위치 | `/workspace/masrouter` |
| `WAE_RUNS_ROOT` | run 결과 저장 디렉터리 | `${WAE_ROUTER_PILOT_ROOT}/runs` |

`MAR` 패키지는 본 저장소에 포함되어 있지 않으며, requirements.txt 로도 설치되지 않습니다. 별도로 MasRouter 체크아웃을 마련한 뒤 `MASROUTER_PATH` 로 가리키는 것이 정상적인 셋업입니다.

## 상태

🧊 **휴면 중** — 시스템 패턴 결론은 살아 있고, 동적 라우팅 결론은 더 좁게 다시 framing 해야 하는 상태입니다.

---

<a name="english"></a>

## English

> 🧊 **Dormant research pilot.**

### What this set out to test

Modern LLM systems often pick which model handles a given query. This project pushed the unit of choice one level out: **choose not just the model but the workflow that solves it.**

A workflow here means something like:

- one-shot answer
- answer-then-refine
- generate several candidates, then compare
- critique-then-rewrite

Comparisons: existing model-level routing (no workflow choice), and static baselines that always use the same workflow. Evaluation ran on code-generation benchmarks (`MBPP`, `HumanEval`) measuring accuracy, cost, and latency together.

### What it found

- **Workflow-as-expert routing did show a real effect.** At matched accuracy, cost and latency dropped meaningfully against model-level routing; the result reproduced across rounds.
- **The stronger claim — "dynamic workflow choice is the core driver" — did not close.** A surprisingly strong static baseline ("just use the one cheap workflow") often matched dynamic selection; the dynamic-only marginal gain was not stable.
- So: **the systems pattern survives; the dynamic-routing headline does not.**

Full numbers:

- 🇰🇷 [`closure_reports/project_closure_report_ko_20260327.md`](closure_reports/project_closure_report_ko_20260327.md)
- 🇬🇧 [`closure_reports/project_closure_report_20260327.md`](closure_reports/project_closure_report_20260327.md)

### Why it's on hold

The intended conclusion ("dynamic workflow routing cleanly beats static baselines") did not stand cleanly, and the final round did not run to completion. A restart would reframe the primary claim around "workflow-as-expert systems efficiency" and treat dynamic routing as a secondary hypothesis. Worth reopening when there is a clearer angle for the dynamic part.

### Where to look first when revisiting

- 📖 [`GLOSSARY.md`](GLOSSARY.md) — Decoder ring for the internal vocabulary that survived into the source tree and the closure reports (`MAR` / MasRouter, the four candidate workflows, the `wf_*` id naming, round names like `round7r2` / `expanded_7b`, `status/` snapshot format, env vars, cost-matched comparison rules).
- [`docs/EXPERIMENT_OVERVIEW.md`](docs/EXPERIMENT_OVERVIEW.md) — what was compared, exactly.
- [`docs/HANDOFF_RUNBOOK.md`](docs/HANDOFF_RUNBOOK.md) — how to resume.
- [`docs/KNOWN_ISSUES_AND_FIXES.md`](docs/KNOWN_ISSUES_AND_FIXES.md) — recurring pitfalls.
- [`docs/PUBLISHING_GUIDE.md`](docs/PUBLISHING_GUIDE.md) — writeup notes for the narrowed framing.
- [`status/`](status/) — last progress snapshot.
- [`artifacts/reports/`](artifacts/reports/) — per-round reports (KO included).

### Code map

| File | What it does |
|---|---|
| [`src/run_pilot.py`](src/run_pilot.py) | Main experiment runner |
| [`src/workflow_router_patch.py`](src/workflow_router_patch.py) | Router patch that routes by workflow rather than by model |
| [`src/workflow_llm.py`](src/workflow_llm.py) | Wrapper that exposes a workflow as a single LLM-shaped callable |
| [`src/workflow_profile.py`](src/workflow_profile.py) | Defines candidate workflows |
| [`src/offline_pareto_builder.py`](src/offline_pareto_builder.py) | Precomputes the cost-quality Pareto of candidates offline |
| [`src/compare_runs.py`](src/compare_runs.py) | Cost-matched comparison and report generation |
| [`src/monitor.py`](src/monitor.py) | Run-state monitoring |

### Folder map

```
.
├── src/                       router patch / workflow wrapper / runner / comparison / monitor
├── config/                    experiment + endpoint configs
├── scripts/                   per-round run / resume scripts
├── docs/                      overview / runbook / pitfalls / writeup guide
├── status/                    last progress snapshot
├── artifacts/reports/         round reports (KO / EN)
├── artifacts/round7r2/        partial outputs from the last round
├── artifacts/snapshots/       progress-snapshot JSONs
├── closure_reports/           closure reports (KO / EN)
├── GLOSSARY.md                internal-vocabulary decoder ring
└── launch_vllm.sh / setup_env.sh / stop_vllm.sh / run_expanded_7b.sh
```

Endpoint configs read credentials from environment variables and do not embed them in source.

### Environment

```bash
bash setup_env.sh
bash launch_vllm.sh   # requires a separate vLLM serving environment up
```

The absolute paths from the original execution environment
(`/workspace/wae_router_pilot`, `/workspace/masrouter`) can be overridden via
environment variables:

| Variable | Meaning | Default |
| --- | --- | --- |
| `WAE_ROUTER_PILOT_ROOT` | This repo's root (where run outputs live) | `/workspace/wae_router_pilot` (or auto-detected from script location) |
| `MASROUTER_PATH` | External MasRouter (`MAR`) package checkout | `/workspace/masrouter` |
| `WAE_RUNS_ROOT` | Run output directory | `${WAE_ROUTER_PILOT_ROOT}/runs` |

The `MAR` package is not vendored in this repository and is not installed via `requirements.txt`. The expected setup is a separate MasRouter checkout pointed to by `MASROUTER_PATH`.

### Status

🧊 **Dormant** — the systems-pattern finding survives; the dynamic-routing headline needs a narrower reframing.

### License

Released under [CC BY-NC 4.0](./LICENSE).
