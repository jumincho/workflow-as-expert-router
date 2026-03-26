# workflow-as-expert-router

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
└── launch_vllm.sh / setup_env.sh / stop_vllm.sh / run_expanded_7b.sh
```

엔드포인트 설정은 자격증명을 직접 박지 않고 환경변수로만 주입하도록 돼 있습니다. 그대로 유지하시면 됩니다.

## 환경

```bash
bash setup_env.sh
bash launch_vllm.sh   # 별도 서빙 환경이 떠 있어야 합니다
```

## 상태

🧊 **휴면 중** — 시스템 패턴 결론은 살아 있고, 동적 라우팅 결론은 더 좁게 다시 framing 해야 하는 상태입니다.
