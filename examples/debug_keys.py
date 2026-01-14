from datasets import load_dataset

ds = load_dataset("SWE-bench/SWE-smith", split="train", streaming=True)
for task in ds:
    print("Keys found in task:", task.keys())
    print("Base Commit value:", task.get('base_commit'))
    print("Environment setup commit:", task.get('environment_setup_commit'))
    break
