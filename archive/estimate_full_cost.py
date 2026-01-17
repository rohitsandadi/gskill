#!/usr/bin/env python3
"""
Full cost estimation for GEPA optimization.
Runs a mini GEPA loop (1 generation, small val set) to measure:
- Agent cost (rollouts)
- Proposer/reflection cost

Usage:
    python src/estimate_full_cost.py --agent-model openai/gpt-5.2 --reflection-model openai/gpt-5.2-pro
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

import litellm

# Track costs using LiteLLM's built-in tracking
total_cost = 0.0
call_count = 0
agent_cost = 0.0
reflection_cost = 0.0

def track_cost(kwargs, completion_response, start_time, end_time):
    """LiteLLM success callback to track costs."""
    global total_cost, call_count, agent_cost, reflection_cost
    call_count += 1
    try:
        cost = litellm.completion_cost(completion_response=completion_response)
        total_cost += cost
        
        # Try to identify model type
        model = kwargs.get('model', '')
        if 'pro' in model.lower():
            reflection_cost += cost
            print(f"    [REFLECTION] Call {call_count}: ${cost:.4f}")
        else:
            agent_cost += cost
            print(f"    [AGENT] Call {call_count}: ${cost:.4f}")
    except Exception as e:
        print(f"    Call {call_count}: cost unknown ({e})")

# Register the callback
litellm.success_callback = [track_cost]

def main():
    global total_cost, call_count, agent_cost, reflection_cost
    
    parser = argparse.ArgumentParser(description="Estimate full GEPA cost (agent + proposer)")
    parser.add_argument("--val-tasks", type=int, default=2, help="Number of validation tasks")
    parser.add_argument("--agent-model", type=str, default="openai/gpt-5.2", help="Agent model")
    parser.add_argument("--reflection-model", type=str, default="openai/gpt-5.2-pro", help="Reflection/proposer model")
    parser.add_argument("--budget", type=float, default=200.0, help="Total budget in dollars")
    args = parser.parse_args()

    print(f"=" * 70)
    print(f"FULL GEPA COST ESTIMATION")
    print(f"=" * 70)
    print(f"Agent model: {args.agent_model}")
    print(f"Reflection model: {args.reflection_model}")
    print(f"Val tasks: {args.val_tasks}")
    print(f"Budget: ${args.budget:.2f}")
    print()
    print("Running 1 generation of GEPA to measure full cost...")
    print(f"=" * 70)
    print()

    # Import GEPA and our adapter
    from gepa.api import optimize
    from src.adapters.swe_adapter import SWEAdapter

    # Load small val set
    val_file = Path("data/pygments_val.json")
    if not val_file.exists():
        print("ERROR: data/pygments_val.json not found.")
        return
    
    with open(val_file) as f:
        val_data = json.load(f)[:args.val_tasks]
    
    # Use same data for train (GEPA needs both)
    train_data = val_data
    
    # Setup adapter
    adapter = SWEAdapter(model_name=args.agent_model)
    
    # Baseline prompt
    seed_candidate = {"system_prompt": """You are an autonomous software engineer.
You will be given a specific issue to fix in a software repository.
You have access to the codebase and can write files and run commands.
Your goal is to reproduce the issue (if possible) and then fix it.
You MUST generate a patch using `git diff` implicitly by changing files.
When you are done, submit your changes."""}

    # Run 1 generation
    result = optimize(
        seed_candidate=seed_candidate,
        trainset=train_data,
        valset=val_data,
        adapter=adapter,
        reflection_lm=args.reflection_model,
        max_metric_calls=args.val_tasks * 3,  # Small budget
        reflection_minibatch_size=1,
        candidate_selection_strategy="pareto",
        skip_perfect_score=True,
        perfect_score=1.0,
        use_wandb=False,
        run_dir="gepa_results",
        display_progress_bar=False,
        seed=42,
    )

    # Results
    print(f"\n{'=' * 70}")
    print(f"RESULTS")
    print(f"{'=' * 70}")
    print(f"Total API calls: {call_count}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"  - Agent cost: ${agent_cost:.4f}")
    print(f"  - Reflection cost: ${reflection_cost:.4f}")
    
    if args.val_tasks > 0 and total_cost > 0:
        per_rollout = agent_cost / args.val_tasks
        # Reflection happens once per generation, agent runs on all val tasks
        # For larger runs: total = (per_rollout * val_size) + reflection_per_gen
        
        print(f"\nPer-rollout (agent): ${per_rollout:.4f}")
        print(f"Per-generation reflection: ${reflection_cost:.4f}")
        
        # Estimate for full runs
        val_50_gen_cost = (per_rollout * 50) + reflection_cost
        val_100_gen_cost = (per_rollout * 100) + reflection_cost
        
        max_gens_50 = int(args.budget / val_50_gen_cost)
        max_gens_100 = int(args.budget / val_100_gen_cost)
        
        print(f"\nWith ${args.budget:.2f} budget:")
        print(f"  50 val tasks: ~${val_50_gen_cost:.2f}/gen → {max_gens_50} generations")
        print(f"  100 val tasks: ~${val_100_gen_cost:.2f}/gen → {max_gens_100} generations")
        
        # Recommended settings
        print(f"\nRecommended for 10-20 generations with 50 val tasks:")
        print(f"  Estimated cost: ${val_50_gen_cost * 15:.2f}")
        print(f"  max_metric_calls: {50 * 15} = 750")
    else:
        print("\nNo cost data captured. Check model names.")

if __name__ == "__main__":
    main()
