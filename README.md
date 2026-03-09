# Enhancing Agent Reliability on τ-bench via Multi-Agent Test-Time Scaling

A multi-agent framework that improves LLM tool-calling agent performance on [τ-bench](https://arxiv.org/abs/2406.12045) through adaptive test-time scaling strategies, loop detection, enhanced prompting, and periodic self-reflection.

Built on top of [Sierra's τ-bench benchmark](https://github.com/sierra-research/tau-bench) for the CSE578 Agentic AI course.

## Architecture

```
                    ┌──────────────────────┐
                    │   MetaController     │
                    │  (HA-TTS Router)     │
                    │                      │
                    │  LLM-based Difficulty │
                    │     Estimator        │
                    └──────┬───────────────┘
                           │
            ┌──────────────┼──────────────┬──────────────┐
            │              │              │              │
         easy           medium          hard        very_hard
            │              │              │              │
   ┌────────▼───┐  ┌───────▼──────┐ ┌────▼────────┐ ┌──▼───────┐
   │ToolCalling │  │    ABF       │ │   ReAct +   │ │  Stub    │
   │  Agent     │  │   Agent      │ │ Reflection  │ │(fallback)│
   │  (1x)      │  │ (~1.5-2x)   │ │ (~1.2-1.5x) │ │          │
   └────────────┘  └──────────────┘ └─────────────┘ └──────────┘
         │              │              │              │
         └──────────────┴──────────────┴──────────────┘
                           │
                  All agents include:
                  • Loop Detector
                  • Enhanced System Prompts
```

## Strategies

| Strategy | CLI Flag | Description | Extra Cost |
|---|---|---|---|
| Tool Calling | `tool-calling` | Baseline native function calling via LiteLLM | 1x |
| ReAct | `react` | Text-based reasoning (Thought → Action) | 1x |
| ACT | `act` | Action-only, no explicit reasoning | 1x |
| ABF | `abf` | Adaptive Budget Forcing — S1-style "Wait," reconsideration scaled by difficulty | ~1.5-2x |
| ReAct + Reflection | `react-reflection` | ReAct with periodic reflection checkpoints every N tool calls | ~1.2-1.5x |
| HA-TTS | `ha-tts` | Meta-controller that routes tasks to the best strategy based on LLM-estimated difficulty | Adaptive |

## Key Improvements Over Baseline τ-bench

| Feature | What It Does | Targets |
|---|---|---|
| **Loop Detector** | Detects repeated identical tool calls. Warning at 2x, force-break at 3x. Applied to all agents. | Looping & Inefficient Reasoning (up to 56% of errors) |
| **Enhanced System Prompts** | Behavioral guardrails injected into all agents: anti-escalation, auth-first, constraint tracking, multi-step completion enforcement | Premature Escalation (up to 42%), Auth failures (up to 34%) |
| **ReactReflectionAgent** | Every 4 tool calls, forces the agent to review progress, check constraints, and plan remaining steps | All error categories simultaneously |
| **LLM Difficulty Estimator** | Replaces keyword-only heuristics with an LLM classification call (falls back to keywords on failure) | Better routing accuracy for HA-TTS |

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
git clone https://github.com/Samudyata/tau-bench-multi-agent-approach.git
cd tau-bench-multi-agent-approach
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

# ReAct + Reflection (new)
python run.py --agent-strategy react-reflection --env airline --model Qwen/Qwen3-32B \
  --model-provider openai --user-model gpt-4o --user-model-provider openai \
  --user-strategy llm --num-trials 5

# Adaptive Budget Forcing
python run.py --agent-strategy abf --env airline --model Qwen/Qwen3-32B \
  --model-provider openai --user-model gpt-4o --user-model-provider openai \
  --user-strategy llm --num-trials 5
```

### Run HA-TTS (auto-routes by difficulty)

```bash
python run.py --agent-strategy ha-tts --env airline --model Qwen/Qwen3-32B \
  --model-provider openai --user-model gpt-4o --user-model-provider openai \
  --user-strategy llm --num-trials 5
```

### Run specific tasks

```bash
python run.py --agent-strategy react-reflection --env retail --model Qwen/Qwen3-32B \
  --model-provider openai --user-model gpt-4o --user-model-provider openai \
  --user-strategy llm --task-ids 0 5 10 15 20
```

## Project Structure

```
tau_bench/agents/
├── base.py                      # Abstract Agent base class
├── prompts.py                   # Shared ENHANCED_GUIDELINES & REFLECTION_PROMPT
├── difficulty.py                # Difficulty estimator (LLM-based + keyword fallback)
├── tool_calling_agent.py        # Baseline: native function calling + loop detector
├── chat_react_agent.py          # ReAct/ACT agent + loop detector
├── adaptive_budget_agent.py     # ABF agent + loop detector
├── react_reflection_agent.py    # ReAct + Reflection agent (NEW)
├── meta_controller_agent.py     # HA-TTS meta-controller
└── few_shot_agent.py            # Few-shot in-context learning agent

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
