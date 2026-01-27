"""
Fitness function wrapper for SWE tasks using the optimize_anything API.

This module provides a FitnessFn-compatible interface for evaluating SWE-smith tasks,
wrapping the existing SWE harness logic in the new optimize_anything format.
"""

from typing import Any, Dict, List, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from src.swe_harness import SWEHarness


def create_swe_fitness_fn(
    workspace_root: str = "/tmp/gepa_workenvs/pygments",
    model_name: str = "gpt-4o",
    n_workers: int = 6
):
    """
    Factory function to create a FitnessFn for SWE-smith tasks.
    
    Args:
        workspace_root: Base directory for worker workspaces
        model_name: LLM model to use for the agent
        n_workers: Number of parallel workers
        
    Returns:
        A FitnessFn that evaluates candidates on SWE tasks
    """
    
    # Create harness pool - one per worker
    harness_pool = []
    for i in range(n_workers):
        worker_workspace = f"{workspace_root}_{i}"
        harness_pool.append(SWEHarness(workspace_root=worker_workspace))
    
    # Track which harnesses are available
    harness_available = [True] * n_workers
    harness_lock = threading.Lock()
    task_count = 0
    task_count_lock = threading.Lock()
    
    def get_harness():
        """Get an available harness from the pool."""
        with harness_lock:
            for i, available in enumerate(harness_available):
                if available:
                    harness_available[i] = False
                    return i, harness_pool[i]
        raise RuntimeError("No harness available - pool exhausted")
    
    def release_harness(idx: int):
        """Release a harness back to the pool."""
        with harness_lock:
            harness_available[idx] = True
    
    def process_single_task(
        task: Dict[str, Any],
        system_prompt: str,
        task_idx: int,
        total_tasks: int
    ):
        """Process a single SWE task and return (score, output, side_info)."""
        nonlocal task_count
        
        # Get a harness from the pool
        harness_idx, harness = get_harness()
        
        try:
            with task_count_lock:
                task_count += 1
                current_count = task_count
            
            instance_id = task.get('instance_id', 'unknown')[:50]
            print(f"[Task {task_idx+1}/{total_tasks}] {instance_id}...", flush=True)
            
            # 1. Setup Environment
            base_commit = task.get("base_commit")
            if not base_commit:
                repo_field = task.get("repo", "")
                if "." in repo_field:
                    base_commit = repo_field.split(".")[-1]
                else:
                    print(f"  WARNING: Could not determine base commit. Using HEAD.")
                    base_commit = "master"
            
            bug_patch = task.get("patch", "")
            harness.setup_task(base_commit, bug_patch=bug_patch)

            # 2. Run Agent
            problem = task["problem_statement"]
            patch, agent_reasoning_trace, agent_metrics = harness.run_agent(
                problem, system_prompt, model_name=model_name
            )

            # 3. Verify with tests
            has_patch = len(patch.strip()) > 0
            passed = False
            feedback_msg = ""
            test_verification_output = ""

            if not has_patch:
                passed = False
                feedback_msg = "Agent did not produce any valid patch (git diff was empty)."
                test_verification_output = "No patch to test."
            else:
                # FAIL_TO_PASS Tests
                fail_to_pass = task.get("FAIL_TO_PASS", [])
                if fail_to_pass:
                    fail_test_cmd = f"pytest {' '.join(fail_to_pass)} -v"
                else:
                    fail_test_cmd = "pytest"
                
                f2p_passed, f2p_output = harness.verify(test_cmd=fail_test_cmd)
                test_verification_output = f"=== FAIL_TO_PASS TESTS ===\n{f2p_output}"
                
                if not f2p_passed:
                    passed = False
                    feedback_msg = "FAIL_TO_PASS tests failed - the fix does not solve the issue."
                else:
                    # PASS_TO_PASS Tests (Regression Check)
                    pass_to_pass = task.get("PASS_TO_PASS", [])
                    
                    if pass_to_pass:
                        sample_size = min(10, len(pass_to_pass))
                        sampled_tests = pass_to_pass[:sample_size]
                        pass_test_cmd = f"pytest {' '.join(sampled_tests)} -v"
                        
                        p2p_passed, p2p_output = harness.verify(test_cmd=pass_test_cmd)
                        test_verification_output += f"\n\n=== PASS_TO_PASS TESTS (sampled {sample_size}/{len(pass_to_pass)}) ===\n{p2p_output}"
                        
                        if not p2p_passed:
                            passed = False
                            feedback_msg = "PASS_TO_PASS regression failed - the fix breaks existing tests."
                        else:
                            passed = True
                            feedback_msg = "All tests passed (FAIL_TO_PASS fixed + PASS_TO_PASS still passing)."
                    else:
                        passed = True
                        feedback_msg = "FAIL_TO_PASS tests passed. No PASS_TO_PASS tests to check."

            score = 1.0 if passed else 0.0
            
            # Rollout output - what the agent actually produced
            rollout_output = {
                "patch": patch,
                "success": passed,
                "steps": agent_metrics.get("steps", 0),
                "estimated_tokens": agent_metrics.get("estimated_tokens", 0)
            }
            
            # Side info - rich feedback for GEPA reflection
            # Truncate traces to avoid token limits while preserving diagnostic info
            agent_trace_truncated = agent_reasoning_trace[:10000]
            test_output_truncated = test_verification_output[:5000]
            
            side_info = {
                # Core evaluation information
                "Input": {
                    "Task ID": task["instance_id"],
                    "Problem Statement": task["problem_statement"][:500] + "..."
                },
                "Generated Outputs": {
                    "Agent Reasoning & Actions": agent_trace_truncated,
                    "Patch Generated": patch[:2000] if patch else "No patch generated"
                },
                "Feedback": {
                    "Test Verification Output": test_output_truncated,
                    "Status": feedback_msg,
                    "Score": score
                },
                # Metrics for multi-objective optimization (optional)
                "scores": {
                    "correctness": score,
                    "efficiency": 1.0 - (agent_metrics.get("steps", 30) / 30.0),  # Normalize steps
                }
            }
            
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  [{instance_id[:30]}] {status}", flush=True)

            # Cleanup
            harness.cleanup()
            
            return score, rollout_output, side_info
            
        finally:
            release_harness(harness_idx)
    
    def swe_fitness_fn(
        candidate: Dict[str, str],
        batch: Sequence[Dict[str, Any]]
    ) -> List[tuple[float, Dict[str, Any], Dict[str, Any]]]:
        """
        FitnessFn for SWE-smith tasks.
        
        Args:
            candidate: Dictionary containing 'system_prompt' key
            batch: List of SWE task instances
            
        Returns:
            List of (score, rollout_output, side_info) tuples
        """
        system_prompt = candidate.get("system_prompt", "You are a helpful software engineering assistant.")

        print(f"\n{'='*70}")
        print(f"Evaluating batch of {len(batch)} tasks with {n_workers} workers...")
        print(f"{'='*70}\n")

        # Use ThreadPoolExecutor for parallel execution
        results = [None] * len(batch)  # Preserve order
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(
                    process_single_task,
                    task, system_prompt, i, len(batch)
                ): i
                for i, task in enumerate(batch)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    score, output, side_info = future.result()
                    results[idx] = (score, output, side_info)
                except Exception as e:
                    print(f"  Task {idx} failed with error: {e}")
                    # Return failure result
                    results[idx] = (
                        0.0,
                        {"patch": "", "success": False, "steps": 0, "estimated_tokens": 0},
                        {
                            "Input": {"Task ID": f"task_{idx}", "Problem Statement": "Error"},
                            "Generated Outputs": {"Error": str(e)},
                            "Feedback": {"Status": f"Task failed with exception: {e}", "Score": 0.0},
                            "scores": {"correctness": 0.0, "efficiency": 0.0}
                        }
                    )
        
        # Print batch summary
        scores = [r[0] for r in results]
        outputs = [r[1] for r in results]
        pass_count = sum(1 for s in scores if s == 1.0)
        avg_steps = sum(o.get("steps", 0) for o in outputs) / len(outputs) if outputs else 0
        avg_tokens = sum(o.get("estimated_tokens", 0) for o in outputs) / len(outputs) if outputs else 0
        print(f"\nBatch complete: {pass_count}/{len(batch)} passed ({pass_count/len(batch)*100:.1f}%)")
        print(f"Avg steps: {avg_steps:.1f}, Avg tokens: {avg_tokens:.0f}")
        print(f"{'='*70}\n")

        return results
    
    return swe_fitness_fn

