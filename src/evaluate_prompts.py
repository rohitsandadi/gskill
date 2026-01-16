"""
Evaluate and compare baseline vs optimized prompts.
This script validates that GEPA is actually improving the prompts.
"""

import json
import argparse
from pathlib import Path
from src.adapters.swe_adapter import SWEAdapter

def load_split(split_name="test"):
    """Load a pre-split dataset."""
    data_file = Path("data") / f"pygments_{split_name}.json"
    if not data_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {data_file}\n"
            f"Run: python split_dataset.py first"
        )

    with open(data_file) as f:
        return json.load(f)

def evaluate_prompt(prompt_text, split_name="test", limit=None, model_name="gemini/gemini-2.5-flash-lite"):
    """
    Evaluate a prompt on a dataset split.
    Returns (pass_rate, total_tasks, results).
    """

    print(f"\nEvaluating prompt on {split_name} split...")
    print(f"Model: {model_name}")

    # Load data
    data = load_split(split_name)
    if limit:
        data = data[:limit]

    print(f"Tasks: {len(data)}")
    print("-" * 60)

    # Create adapter
    adapter = SWEAdapter(
        workspace_root="/tmp/gepa_workenvs/pygments",
        model_name=model_name
    )

    # Evaluate
    candidate = {"system_prompt": prompt_text}
    eval_batch = adapter.evaluate(
        batch=data,
        candidate=candidate,
        capture_traces=False  # Don't need traces for evaluation
    )

    # Calculate metrics
    scores = eval_batch.scores
    pass_count = sum(1 for s in scores if s == 1.0)
    total = len(scores)
    pass_rate = (pass_count / total) * 100 if total > 0 else 0

    print(f"\nResults:")
    print(f"  Passed: {pass_count}/{total}")
    print(f"  Pass Rate: {pass_rate:.1f}%")

    return pass_rate, total, eval_batch

def compare_prompts(baseline_prompt, optimized_prompt, split_name="test", limit=None, model_name="gemini/gemini-2.5-flash-lite"):
    """
    Compare baseline vs optimized prompt.
    Shows if GEPA actually improved performance.
    """

    print("=" * 70)
    print("PROMPT COMPARISON")
    print("=" * 70)

    # Evaluate baseline
    print("\n[1/2] BASELINE PROMPT")
    print("-" * 70)
    baseline_rate, total, _ = evaluate_prompt(baseline_prompt, split_name, limit, model_name)

    # Evaluate optimized
    print("\n[2/2] OPTIMIZED PROMPT")
    print("-" * 70)
    optimized_rate, _, _ = evaluate_prompt(optimized_prompt, split_name, limit, model_name)

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"\nBaseline Pass Rate:  {baseline_rate:.1f}%")
    print(f"Optimized Pass Rate: {optimized_rate:.1f}%")

    improvement = optimized_rate - baseline_rate
    if improvement > 0:
        print(f"\n✓ IMPROVEMENT: +{improvement:.1f} percentage points")
        print(f"  Relative gain: +{(improvement/baseline_rate)*100:.1f}%")
    elif improvement < 0:
        print(f"\n✗ REGRESSION: {improvement:.1f} percentage points")
    else:
        print(f"\n- NO CHANGE")

    print(f"\nTasks evaluated: {total}")

    return baseline_rate, optimized_rate

def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare prompts")
    parser.add_argument("--baseline", type=str, help="Path to baseline prompt file")
    parser.add_argument("--optimized", type=str, help="Path to optimized prompt file")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--limit", type=int, help="Limit number of tasks (for quick testing)")
    parser.add_argument("--model", type=str, default="gemini/gemini-2.5-flash-lite",
                        help="Model to use for evaluation")
    args = parser.parse_args()

    # Load prompts
    if args.baseline and args.optimized:
        with open(args.baseline) as f:
            baseline = f.read().strip()
        with open(args.optimized) as f:
            optimized = f.read().strip()

        compare_prompts(baseline, optimized, args.split, args.limit, args.model)

    else:
        # Use default baseline for testing
        baseline = """
You are an autonomous software engineer.
You will be given a specific issue to fix in a software repository.
You have access to the codebase and can write files and run commands.
Your goal is to reproduce the issue (if possible) and then fix it.
You MUST generate a patch using `git diff` implicitly by changing files.
When you are done, submit your changes.
        """.strip()

        print("No prompts specified. Evaluating baseline prompt only...")
        evaluate_prompt(baseline, args.split, args.limit, args.model)

if __name__ == "__main__":
    main()
