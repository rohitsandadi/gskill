"""
SWE-smith Dataset Example

This script demonstrates how to load and explore the SWE-smith dataset
from HuggingFace. The dataset contains 52k+ task instances for training
software engineering agents.

No API key required - this only loads data from HuggingFace.
"""

from datasets import load_dataset


def main():
    print("Loading SWE-smith dataset from HuggingFace...")
    print("(This may take a while on first run as it downloads the data)")
    print()
    
    # Load the dataset - streaming mode for faster initial access
    ds = load_dataset("SWE-bench/SWE-smith", split="train", streaming=True)
    
    # Take a few examples to explore
    print("=" * 60)
    print("EXPLORING SWE-SMITH TASK INSTANCES")
    print("=" * 60)
    
    for i, task in enumerate(ds):
        if i >= 3:  # Just show 3 examples
            break
            
        print(f"\n--- Task {i + 1} ---")
        print(f"Instance ID: {task.get('instance_id', 'N/A')}")
        print(f"Repository: {task.get('repo', 'N/A')}")
        print(f"Base Commit: {task.get('base_commit', 'N/A')[:12]}...")
        
        # Show problem statement preview
        problem = task.get('problem_statement', '')
        if problem:
            preview = problem[:200] + "..." if len(problem) > 200 else problem
            print(f"Problem Statement:\n{preview}")
        
        print()
    
    print("=" * 60)
    print("Dataset loaded successfully!")
    print()
    print("Available fields in each task instance:")
    print("  - instance_id: Unique identifier for the task")
    print("  - repo: GitHub repository (e.g., 'django/django')")
    print("  - base_commit: Git commit hash for the task")
    print("  - problem_statement: Natural language description of the issue")
    print("  - patch: The gold solution patch")
    print("  - test_patch: Tests that verify the solution")
    print()
    print("For more info, visit: https://swesmith.com")


if __name__ == "__main__":
    main()
