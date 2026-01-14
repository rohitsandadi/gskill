import os
import argparse
from datasets import load_dataset
from gepa.api import optimize
from src.adapters.pygments_adapter import PygmentsAdapter

def load_pygments_data(limit=10, split="train"):
    print(f"Loading {limit} Pygments examples from SWESmith...")
    ds = load_dataset("SWE-bench/SWE-smith", split=split, streaming=True)
    
    data = []
    count = 0
    target_repo = "swesmith/pygments__pygments"
    
    for item in ds:
        repo_name = item.get('repo', '')
        if target_repo in repo_name:
            data.append(item)
            count += 1
            if count >= limit:
                break
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=2, help="Number of GEPA generations")
    parser.add_argument("--train-size", type=int, default=5, help="Number of training examples")
    parser.add_argument("--workspace", type=str, default="/tmp/gepa_workenvs/pygments")
    parser.add_argument("--model", type=str, default="gemini/gemini-1.5-flash")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test")
    args = parser.parse_args()

    if args.smoke_test:
        args.generations = 1
        args.train_size = 2
        print("🔥 SMOKE TEST MODE 🔥")

    # 1. Load Data
    train_data = load_pygments_data(limit=args.train_size)
    print(f"Loaded {len(train_data)} examples.")

    # 2. Setup Adapter
    adapter = PygmentsAdapter(workspace_root=args.workspace, model_name=args.model)

    # 3. Initial Candidate (The Baseline Prompt)
    # We use a simplified version of the standard SWE-agent prompt
    initial_prompt = """
You are an autonomous software engineer.
You will be given a specific issue to fix in the 'pygments' repository.
You have access to the codebase and can write files and run commands.
Your goal is to reproduce the issue (if possible) and then fix it.
You MUST generate a patch using `git diff` implicitly by changing files.
When you are done, submit your changes.
    """.strip()
    
    seed_candidate = {"system_prompt": initial_prompt}

    # 4. Run GEPA
    print("Starting GEPA Optimization...")
    result = optimize(
        seed_candidate=seed_candidate,
        trainset=train_data,
        adapter=adapter,
        # GEPA hyperparameters
        reflection_lm=args.model,
        max_metric_calls=args.generations * args.train_size * 2, # Rough budget
        reflection_minibatch_size=1, # Small batch for reflection
        use_wandb=False,
        run_dir="gepa_results",
        stop_callbacks=[] # handled by max_metric_calls
    )
    
    print("Optimization Complete!")
    print("Best Candidate Found:")
    print(result.best_candidate)

if __name__ == "__main__":
    main()
