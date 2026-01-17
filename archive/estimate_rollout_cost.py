#!/usr/bin/env python3
"""
Cost estimation script for GEPA rollouts.
Uses LiteLLM's built-in cost tracking for accurate measurement.

Usage:
    python src/estimate_rollout_cost.py --rollouts 3 --model openai/gpt-5-mini
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
from src.swe_harness import SWEHarness

# Track costs using LiteLLM's built-in tracking
total_cost = 0.0
call_count = 0

def track_cost(kwargs, completion_response, start_time, end_time):
    """LiteLLM success callback to track costs."""
    global total_cost, call_count
    call_count += 1
    try:
        cost = litellm.completion_cost(completion_response=completion_response)
        total_cost += cost
        print(f"    Call {call_count}: ${cost:.4f} (total: ${total_cost:.4f})")
    except Exception as e:
        print(f"    Call {call_count}: cost unknown ({e})")

# Register the callback
litellm.success_callback = [track_cost]

def main():
    global total_cost, call_count
    
    parser = argparse.ArgumentParser(description="Estimate per-rollout cost")
    parser.add_argument("--rollouts", type=int, default=3, help="Number of test rollouts")
    parser.add_argument("--model", type=str, default="openai/gpt-5-mini", help="Model to test")
    parser.add_argument("--budget", type=float, default=200.0, help="Total budget in dollars")
    args = parser.parse_args()

    # Load a few validation tasks
    val_file = Path("data/pygments_val.json")
    if not val_file.exists():
        print("ERROR: data/pygments_val.json not found. Run split_dataset.py first.")
        return
    
    with open(val_file) as f:
        tasks = json.load(f)[:args.rollouts]
    
    print(f"=" * 60)
    print(f"COST ESTIMATION TEST")
    print(f"=" * 60)
    print(f"Model: {args.model}")
    print(f"Rollouts: {args.rollouts}")
    print(f"Budget: ${args.budget:.2f}")
    print()

    harness = SWEHarness()
    
    baseline_prompt = """You are an autonomous software engineer.
You will be given a specific issue to fix in a software repository.
You have access to the codebase and can write files and run commands.
Your goal is to reproduce the issue (if possible) and then fix it.
You MUST generate a patch using `git diff` implicitly by changing files.
When you are done, submit your changes."""

    print(f"Running {args.rollouts} test rollouts...\n")
    
    for i, task in enumerate(tasks):
        rollout_start_cost = total_cost
        rollout_start_calls = call_count
        
        print(f"[Rollout {i+1}/{args.rollouts}] {task.get('instance_id', 'unknown')[:50]}...")
        
        # Setup
        base_commit = task.get("base_commit")
        if not base_commit:
            repo_field = task.get("repo", "")
            if "." in repo_field:
                base_commit = repo_field.split(".")[-1]
            else:
                base_commit = "master"
        
        bug_patch = task.get("patch", "")
        harness.setup_task(base_commit, bug_patch=bug_patch)
        
        # Run agent
        problem = task["problem_statement"]
        patch, trace = harness.run_agent(problem, baseline_prompt, model_name=args.model)
        
        # Cleanup
        harness.cleanup()
        
        rollout_cost = total_cost - rollout_start_cost
        rollout_calls = call_count - rollout_start_calls
        print(f"  Rollout complete: {rollout_calls} calls, ${rollout_cost:.4f}")

    # Results
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Total API calls: {call_count}")
    print(f"Total cost: ${total_cost:.4f}")
    
    if args.rollouts > 0 and total_cost > 0:
        per_rollout = total_cost / args.rollouts
        max_rollouts = int(args.budget / per_rollout)
        
        print(f"\nPer-rollout cost: ${per_rollout:.4f}")
        print(f"\nWith ${args.budget:.2f} budget:")
        print(f"  Max rollouts: {max_rollouts}")
        print(f"  With 50 val tasks: {max_rollouts // 50} generations")
        print(f"  With 100 val tasks: {max_rollouts // 100} generations")
        print(f"\nRecommended max_metric_calls: {max_rollouts}")
    else:
        print("\nNo cost data captured. Check if the model name is correct.")

if __name__ == "__main__":
    main()
