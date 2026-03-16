# Enhancing Agent Reliability on τ-bench via Multi-Agent Test-Time Scaling

A multi-agent framework that improves LLM tool-calling agent performance on [τ-bench](https://arxiv.org/abs/2406.12045) through a unified 5-tier adaptive test-time scaling meta-controller. Implements five specialized agent strategies — ABF, Policy Guard, PACE, React Reflection, and Best-of-N — routed by LLM-estimated task difficulty.

Built on top of [Sierra's τ-bench benchmark](https://github.com/sierra-research/tau-bench) for the CSE578 Agentic AI course.

## Architecture

```
                     ┌────────────────────────┐
                     │    MetaController      │
                     │   (HA-TTS Router)      │
                     │                        │
                     │  LLM-based Difficulty   │
                     │     Estimator          │
                     └──────────┬─────────────┘
                                │
       ┌────────────┬───────────┼───────────┬────────────┐
       │            │           │           │            │
   very_easy      easy       medium       hard      very_hard
    (0-1)         (2)        (3-4)       (5-6)       (7+)
       │            │           │           │            │
  ┌────▼────┐ ┌─────▼─────┐ ┌──▼───┐ ┌────▼────────┐ ┌──▼────────┐
  │  ABF    │ │  Policy   │ │ PACE │ │   ReAct +   │ │ Best-of-N │
  │ (min    │ │  Guard    │ │      │ │ Reflection  │ │  (N=2)    │
  │ budget) │ │           │ │      │ │             │ │           │
  │  ~0%    │ │ ~10-20%   │ │~20-30│ │  ~30-50%    │ │   ~2×     │
  └─────────┘ └───────────┘ └──────┘ └─────────────┘ └───────────┘
       │            │           │           │            │
       └────────────┴───────────┴───────────┴────────────┘
                                │
                       All agents include:
                       • Loop Detector
                       • Enhanced System Prompts
```

## Strategies

### Baseline Agents
| Strategy | CLI Flag | Description | Extra Cost |
|---|---|---|---|
| Tool Calling | `tool-calling` | Baseline native function calling via LiteLLM | 1× |
| ReAct | `react` | Text-based reasoning (Thought → Action) | 1× |
| ACT | `act` | Action-only, no explicit reasoning | 1× |

### Phase 3 Agents (New)
| Strategy | CLI Flag | Description | Extra Cost |
|---|---|---|---|
| ABF | `abf` | Adaptive Budget Forcing — S1-style "Wait," reconsideration scaled by difficulty | ~1.5-2× |
| Policy Guard | `policy-guard` | Pre-action policy critic — validates responses against domain policies before execution | ~10-20% |
| PACE | `pace` | Constraint register + pre-action validation — tracks all user requirements systematically | ~20-30% |
| ReAct + Reflection | `react-reflection` | ReAct with periodic reflection checkpoints every N tool calls + loop detection | ~30-50% |
| Best-of-N | `best-of-n` | Multiple full trajectory attempts (N=2), picks best result | ~2× |
| HA-TTS | `ha-tts` | 5-tier meta-controller that routes tasks by LLM-estimated difficulty | Adaptive |

## Key Improvements Over Baseline τ-bench

| Feature | What It Does | Targets |
|---|---|---|
| **Loop Detector** | Detects repeated identical tool calls. Warning at 2×, force-break at 3×. Applied to all agents. | Looping & Inefficient Reasoning (up to 56% of errors) |
| **Enhanced System Prompts** | Behavioral guardrails injected into all agents: anti-escalation, auth-first, constraint tracking, multi-step completion enforcement | Premature Escalation (up to 42%), Auth failures (up to 34%) |
| **Policy Guard** | Lightweight critic validates agent responses against domain policies before execution. Catches surrender phrases and policy violations. | Premature Escalation, Policy & Confirmation violations |
| **PACE** | Constraint register extracts all user requirements upfront, validates each action against the register before execution. | Constraint Misinterpretation (up to 33%), Incomplete Multi-Step |
| **ReactReflectionAgent** | Every 4 tool calls, forces the agent to review progress, check constraints, and plan remaining steps | All error categories simultaneously |
| **Best-of-N** | Runs N=2 full trajectories for the hardest tasks, picks best. Reserved for very_hard tier only. | All categories (brute-force retry) |
| **LLM Difficulty Estimator** | 5-level LLM classification (falls back to keyword scoring). Routes tasks to optimal strategy. | Better routing accuracy for HA-TTS |

## 5-Tier Routing Logic

| Tier | Difficulty Score | Strategy | Rationale |
|---|---|---|---|
| very_easy | 0-1 | ABF (minimal budget) | Simple lookups — minimal overhead |
| easy | 2 | Policy Guard | Single-step modifications — policy check catches common errors cheaply |
| medium | 3-4 | PACE | Multi-constraint tasks — register tracks all requirements |
| hard | 5-6 | React Reflection | Complex multi-step — reflection catches drift, loop detector prevents spinning |
| very_hard | 7+ | Best-of-N (N=2) | Most complex — when single attempts frequently fail |

## Phase 1 Results (Baselines)

**Setup**: User agent = Qwen3-30B-a3b-instruct-2507 (fixed). Tool-calling agent = Qwen3 (4B/8B/14B/32B).

### Airline Domain

| Strategy | Model | pass^1 | pass^2 | pass^3 | pass^4 | pass^5 |
|---|---|---|---|---|---|---|
| ACT | Qwen3-4B | 0.273 | 0.213 | 0.180 | 0.158 | 0.142 |
| ACT | Qwen3-8B | 0.310 | 0.190 | 0.150 | 0.130 | 0.110 |
| ACT | Qwen3-14B | 0.256 | 0.162 | 0.138 | 0.128 | 0.120 |
| ACT | Qwen3-32B | 0.370 | 0.240 | 0.190 | 0.160 | 0.146 |
| ReAct | Qwen3-4B | 0.356 | 0.316 | 0.294 | 0.280 | 0.270 |
| ReAct | Qwen3-8B | 0.287 | 0.183 | 0.145 | 0.126 | 0.113 |
| ReAct | Qwen3-14B | 0.344 | 0.268 | 0.222 | 0.188 | 0.160 |
| ReAct | Qwen3-32B | 0.350 | 0.220 | 0.180 | 0.150 | 0.140 |
| FC | Qwen3-4B | 0.125 | 0.077 | 0.065 | 0.060 | 0.006 |
| FC | Qwen3-8B | 0.269 | 0.157 | 0.126 | 0.110 | 0.097 |
| FC | Qwen3-14B | 0.236 | 0.132 | 0.092 | 0.072 | 0.060 |
| FC | Qwen3-32B | 0.331 | 0.238 | 0.190 | 0.160 | 0.122 |

### Retail Domain

| Strategy | Model | pass^1 | pass^2 | pass^3 | pass^4 | pass^5 |
|---|---|---|---|---|---|---|
| ACT | Qwen3-4B | 0.104 | 0.049 | 0.035 | 0.030 | 0.026 |
| ACT | Qwen3-8B | 0.117 | 0.070 | 0.056 | 0.048 | 0.043 |
| ACT | Qwen3-14B | 0.292 | 0.174 | 0.119 | 0.085 | 0.061 |
| ACT | Qwen3-32B | 0.139 | 0.081 | 0.065 | 0.057 | 0.051 |
| ReAct | Qwen3-4B | 0.075 | 0.059 | 0.056 | 0.054 | 0.052 |
| ReAct | Qwen3-8B | 0.070 | 0.041 | 0.033 | 0.028 | 0.025 |
| ReAct | Qwen3-14B | 0.324 | 0.190 | 0.130 | 0.094 | 0.070 |
| ReAct | Qwen3-32B | 0.304 | 0.177 | 0.142 | 0.124 | 0.111 |
| FC | Qwen3-4B | 0.040 | 0.026 | 0.021 | 0.019 | 0.017 |
| FC | Qwen3-8B | 0.270 | 0.157 | 0.126 | 0.110 | 0.098 |
| FC | Qwen3-14B | 0.348 | 0.220 | 0.159 | 0.125 | 0.104 |
| FC | Qwen3-32B | 0.374 | 0.218 | 0.175 | 0.152 | 0.136 |

## Phase 2 Error Analysis

We classified failures into 9 categories across all model sizes and strategies:

1. **Premature Escalation / Surrender** — Agent gives up before exhausting available tools
2. **Tool Selection & Schema Errors** — Wrong tool or hallucinated function calls
3. **Tool Argument / ID Errors** — Correct tool, wrong arguments
4. **Authentication & Missing Info Failures** — Skipping auth or ignoring available data
5. **Policy & Confirmation Violations** — Skipping confirmation steps or violating domain rules
6. **Constraint & Preference Misinterpretation** — Ignoring user requirements
7. **Incomplete Multi-Step Execution** — Starting but not finishing multi-step workflows
8. **Looping & Inefficient Reasoning** — Repeated identical tool calls with no progress
9. **System / Infrastructure Failures** — Context overflow, API timeouts

### Qwen3-32B Error Distribution (the model used for Phase 3)

**Airline:**

| Error Category | ReAct (%) | ACT (%) | FC (%) |
|---|---|---|---|
| Premature Escalation | 42.0 | 38.0 | 20.0 |
| Constraint Misinterpretation | 22.0 | 23.0 | 30.0 |
| Tool Argument / ID Errors | 10.0 | 12.0 | 16.0 |
| Policy & Confirmation | 9.0 | 11.0 | 10.0 |
| Incomplete Multi-Step | 6.0 | 6.0 | 10.0 |

**Retail:**

| Error Category | ReAct (%) | ACT (%) | FC (%) |
|---|---|---|---|
| Looping & Inefficient Reasoning | 56.2 | 13.1 | 19.4 |
| Tool Selection & Schema Errors | 33.8 | 2.0 | 1.4 |
| Tool Argument / ID Errors | 6.2 | 42.4 | 6.9 |
| Constraint Misinterpretation | 0.0 | 3.0 | 33.3 |
| Authentication & Missing Info | 1.2 | 34.3 | 13.9 |

## Setup

1. Clone this repository:

```bash
git clone https://github.com/Samudyata/Multi-Agent-Framework-TauBench.git
cd Multi-Agent-Framework-TauBench
```

2. Install from source:

```bash
pip install -e .
```

3. Start a vLLM server for the agent model:

```bash
vllm serve Qwen/Qwen3-32B --port 8005
```

4. Set your API key for the user simulator:

```bash
export OPENAI_API_KEY=...
```

## Usage

### Run individual strategies

```bash
# Baseline tool-calling
python run.py --agent-strategy tool-calling --env airline --model Qwen/Qwen3-32B \
  --model-provider openai --user-model gpt-4o --user-model-provider openai \
  --user-strategy llm --num-trials 5

# Policy Guard
python run.py --agent-strategy policy-guard --env airline --model Qwen/Qwen3-32B \
  --model-provider openai --user-model gpt-4o --user-model-provider openai \
  --user-strategy llm --num-trials 5

# PACE (constraint tracking)
python run.py --agent-strategy pace --env airline --model Qwen/Qwen3-32B \
  --model-provider openai --user-model gpt-4o --user-model-provider openai \
  --user-strategy llm --num-trials 5

# ReAct + Reflection
python run.py --agent-strategy react-reflection --env airline --model Qwen/Qwen3-32B \
  --model-provider openai --user-model gpt-4o --user-model-provider openai \
  --user-strategy llm --num-trials 5

# Best-of-N (N=2)
python run.py --agent-strategy best-of-n --env airline --model Qwen/Qwen3-32B \
  --model-provider openai --user-model gpt-4o --user-model-provider openai \
  --user-strategy llm --num-trials 5
```

### Run HA-TTS (5-tier meta-controller, auto-routes by difficulty)

```bash
python run.py --agent-strategy ha-tts --env airline --model Qwen/Qwen3-32B \
  --model-provider openai --user-model gpt-4o --user-model-provider openai \
  --user-strategy llm --num-trials 5
```

### Run specific tasks

```bash
python run.py --agent-strategy ha-tts --env retail --model Qwen/Qwen3-32B \
  --model-provider openai --user-model gpt-4o --user-model-provider openai \
  --user-strategy llm --task-ids 0 5 10 15 20
```

## Project Structure

```
tau_bench/agents/
├── base.py                      # Abstract Agent base class
├── prompts.py                   # Shared ENHANCED_GUIDELINES & REFLECTION_PROMPT
├── difficulty.py                # 5-tier difficulty estimator (LLM + keyword fallback)
├── meta_controller_agent.py     # HA-TTS 5-tier meta-controller
│
├── tool_calling_agent.py        # Baseline: native function calling + loop detector
├── chat_react_agent.py          # ReAct/ACT agent + loop detector
├── few_shot_agent.py            # Few-shot in-context learning agent
│
├── adaptive_budget_agent.py     # ABF agent + loop detector (very_easy tier)
├── policy_guard_agent.py        # Policy Guard agent (easy tier)
├── pace_agent.py                # PACE orchestrator (medium tier)
├── pace/                        # PACE internals
│   ├── register.py              #   Constraint register
│   ├── executor.py              #   Pre-action validator
│   ├── prompts.py               #   PACE-specific prompts
│   └── tools.py                 #   PACE tool wrappers
├── react_reflection_agent.py    # ReAct + Reflection agent (hard tier)
└── best_of_n_agent.py           # Best-of-N agent (very_hard tier)

tau_bench/envs/
├── airline/                     # Airline domain (14 tools, ~61 tasks)
├── retail/                      # Retail domain (16 tools, ~100+ tasks)
├── base.py                      # Environment logic + SHA-256 evaluation
└── user.py                      # User simulator (5 strategies)
```

## Team

Gursparsh Singh Sodhi, Hithaishi Surendra, Jahnvi Seth, Samhitha Harish, Samudyata Sudarshan Jagirdar

## Acknowledgments

Built on [τ-bench](https://github.com/sierra-research/tau-bench) by Sierra Research.

```bibtex
@misc{yao2024tau,
      title={$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains},
      author={Shunyu Yao and Noah Shinn and Pedram Razavi and Karthik Narasimhan},
      year={2024},
      eprint={2406.12045},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2406.12045},
}
```
