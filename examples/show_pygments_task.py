from datasets import load_dataset

def main():
    print("Searching for a Pygments task in SWESmith...")
    ds = load_dataset("SWE-bench/SWE-smith", split="train", streaming=True)
    
    # The repo name logic uses double underscores usually in SWESmith
    target_repo = "swesmith/pygments__pygments"
    
    found = False
    for i, task in enumerate(ds):
        if task['repo'] == target_repo:
            print("\nFOUND IT! Here is a sample task:")
            print("-" * 50)
            print(f"Instance ID: {task['instance_id']}")
            print(f"Repo Field:  {task['repo']}")
            print(f"Base Commit: {task['base_commit']}")
            print("-" * 50)
            print("Problem Statement Snippet:")
            print(task['problem_statement'][:300] + "...")
            found = True
            break
            
        if i > 10000: # Safety break
            break
            
    if not found:
        print(f"Could not find {target_repo} in the first 10,000 examples.")

if __name__ == "__main__":
    main()
