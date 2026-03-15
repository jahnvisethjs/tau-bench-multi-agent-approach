# run_policy_guard.py
# Replace your existing run_baseline.py with this.
# Usage: python run_policy_guard.py --domain airline --model Qwen3-32B --split test

import argparse
from tau_bench.envs import get_env
from tau_bench.run import run_eval          # tau-bench's existing eval loop
from policy_guard_agent import PolicyGuardAgent

# Import whichever base agent worked best for you in Phase 1
from tau_bench.agents.react_agent import ReactAgent   # swap for ACT/FC if needed

POLICY_PATHS = {
    "airline": "tau_bench/envs/airline/data/policy.md",
    "retail":  "tau_bench/envs/retail/data/policy.md",
}

def build_agent(args, model_client):
    base_agent = ReactAgent(
        model=args.model,
        client=model_client,
        tools=get_env(args.domain).tools,
    )
    return PolicyGuardAgent(
        base_agent=base_agent,
        model_client=model_client,
        domain=args.domain,
        policy_path=POLICY_PATHS[args.domain],
        max_retries=2,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["airline", "retail"], required=True)
    parser.add_argument("--model", default="Qwen3-32B")
    parser.add_argument("--num-trials", type=int, default=5)  # for pass^k
    parser.add_argument("--output-dir", default="results/policy_guard")
    args = parser.parse_args()

    # Your existing model client setup
    from tau_bench.model_utils import get_client
    client = get_client(args.model)

    agent = build_agent(args, client)

    # tau-bench's eval loop handles pass^k automatically
    run_eval(
        agent=agent,
        domain=args.domain,
        num_trials=args.num_trials,
        output_dir=args.output_dir,
        reset_fn=agent.reset,   # clear turn log between tasks
    )
