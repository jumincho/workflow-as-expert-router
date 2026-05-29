<div align="center">

# workflow-as-expert-router

**不只是路由模型,而是把整个"工作流(workflow)"作为专家单元来路由**

![Status](https://img.shields.io/badge/status-dormant-lightgrey)
![Language](https://img.shields.io/badge/language-Python-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)
![Closure](https://img.shields.io/badge/closure-2026--03-blue)

[한국어](./README.md) · [English](./README.md#english) · **中文**

</div>

> 🧊 **休眠中的研究试点。**

## ⭐ 核心结果 (TL;DR)

- **以"工作流"为单位路由专家的想法效果得到复现**——在相同准确率下,成本与延迟相比模型级路由明显下降。
- 但**更强的主张("动态选择工作流本身才是关键")没能闭合**——只用一个廉价工作流的静态基线出人意料地强。
- 即:**系统模式留存,而动态路由这一头条结论需要更窄的重新框定。**

## 这项研究想看什么

现在的 LLM 系统经常会自动选择"把这道问题交给哪个模型"。本项目再往前一步:
**不仅选模型,还要同时选"用哪个工作流(workflow)去解决它"。**

这里的"工作流"指的是这样一些东西:

- 一次性给出答案
- 先答一遍,再自我修订
- 生成多个候选,再做比较
- 先批评,再重写

对照对象是: 没有工作流选择的、传统的模型级路由 ; 以及"始终只用一个固定工作流"的简单静态基线。
评估在代码生成基准(`MBPP`、`HumanEval`)上进行,同时观察准确率、成本和延迟。

## 发现了什么

- **"以工作流为专家单元来路由"这件事本身有效。** 相对于传统的模型级路由,在同等准确率下,成本和延迟都显著下降,并且这一结论在多轮实验中可复现。
- **但"动态选择工作流本身才是关键"这个更强的主张没有最终成立。** "干脆只用一个便宜工作流就行"这种静态基线意外地强,动态选择带来的额外收益并不稳定。
- 也就是说: **作为系统模式,这套工作流路由活下来了;但"动态路由才是决定性因素"这个标题级主张没能闭合。**

完整数据可在以下两份关闭报告中查阅:

- 🇰🇷 [`closure_reports/project_closure_report_ko_20260327.md`](closure_reports/project_closure_report_ko_20260327.md)
- 🇬🇧 [`closure_reports/project_closure_report_20260327.md`](closure_reports/project_closure_report_20260327.md)

## 为什么暂停

最初想立住的结论 ——"动态工作流路由能干净地胜过静态基线"—— 并没有干净地立住,
而且最后一轮实验也没跑完。如果未来重启,更诚实的方式不是"再调一调动态肯定能赢",
而是把"以工作流为单元提升系统效率"作为主线 framing,把动态路由本身当成次要假设。
当动态那部分有更清楚的切入点时,值得重新打开这个项目。

## 重启时先看哪里

- 📖 [`GLOSSARY.md`](GLOSSARY.md) —— 把源码和关闭报告里出现的内部术语 (`MAR`/MasRouter、四种候选工作流、`wf_io_general` 这类工作流 id、`round7r2`/`expanded_7b` 这类轮次名、`status/` 快照格式、环境变量、成本对齐比较规则等) 用日常话语解释清楚的词典
- [`docs/EXPERIMENT_OVERVIEW.md`](docs/EXPERIMENT_OVERVIEW.md) —— 这项研究到底比较的是什么
- [`docs/HANDOFF_RUNBOOK.md`](docs/HANDOFF_RUNBOOK.md) —— 怎样重新跑起来
- [`docs/KNOWN_ISSUES_AND_FIXES.md`](docs/KNOWN_ISSUES_AND_FIXES.md) —— 反复踩过的坑
- [`docs/PUBLISHING_GUIDE.md`](docs/PUBLISHING_GUIDE.md) —— 如果以收窄后的 framing 写文章应注意什么
- [`status/`](status/) —— 最后一次的进度快照
- [`artifacts/reports/`](artifacts/reports/) —— 各轮报告(包含韩文版)

## 代码地图

| 文件 | 做什么 |
|---|---|
| [`src/run_pilot.py`](src/run_pilot.py) | 主实验 runner |
| [`src/workflow_router_patch.py`](src/workflow_router_patch.py) | 把"按模型路由"的上游 router 改造成"按工作流路由"的补丁 |
| [`src/workflow_llm.py`](src/workflow_llm.py) | 把一个工作流包装成单个 LLM 形态的可调用对象 |
| [`src/workflow_profile.py`](src/workflow_profile.py) | 定义有哪些候选工作流 |
| [`src/offline_pareto_builder.py`](src/offline_pareto_builder.py) | 离线先把候选工作流的"成本-质量 Pareto"算出来 |
| [`src/compare_runs.py`](src/compare_runs.py) | 把多个 run 在等成本条件下做比较并生成报告 |
| [`src/monitor.py`](src/monitor.py) | run 状态监控 |

## 目录概览

```
.
├── src/                       router 补丁 / 工作流封装 / runner / 比较 / 监控
├── config/                    实验 + 模型端点配置
├── scripts/                   每轮的运行 / 续跑脚本
├── docs/                      概述 / runbook / 已知坑 / 写作指南
├── status/                    最后一次进度快照
├── artifacts/reports/         各轮报告 (KO / EN)
├── artifacts/round7r2/        最后一轮的部分输出
├── artifacts/snapshots/       进度快照 JSON
├── closure_reports/           关闭报告 (KO / EN)
├── GLOSSARY.md                内部术语词典
└── launch_vllm.sh / setup_env.sh / stop_vllm.sh / run_expanded_7b.sh
```

端点配置只从环境变量读取凭据,源码里不会写死。

## 环境

```bash
bash setup_env.sh
bash launch_vllm.sh   # 需要另外有一套 vLLM 服务在运行
```

原始执行环境里的绝对路径(`/workspace/wae_router_pilot`、`/workspace/masrouter`)
可以通过以下环境变量覆盖:

| 环境变量 | 含义 | 默认值 |
| --- | --- | --- |
| `WAE_ROUTER_PILOT_ROOT` | 本仓库根目录(run 输出位置) | `/workspace/wae_router_pilot`(或按脚本所在位置自动检测) |
| `MASROUTER_PATH` | 外部 MasRouter(`MAR`)包的检出位置 | `/workspace/masrouter` |
| `WAE_RUNS_ROOT` | run 输出目录 | `${WAE_ROUTER_PILOT_ROOT}/runs` |

`MAR` 包不在本仓库内,也不会通过 `requirements.txt` 安装。
正常的搭建方式是另外准备一份 MasRouter 检出,并通过 `MASROUTER_PATH` 指向它。

## 状态

🧊 **休眠中** —— 系统模式层面的结论仍然立得住; "动态路由"这条主张需要以更窄的 framing 重新展开。

## 许可证

以 [CC BY-NC 4.0](./LICENSE) 发布。
