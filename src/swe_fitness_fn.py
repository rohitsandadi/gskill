"""
Fitness function for GEPA optimize_anything API.

This wraps the SWE harness logic into a simple fitness function that:
- Takes a candidate (prompt) and batch of tasks
- Returns (score, output, side_info) tuples for each task
"""

from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading

from src.swe_harness import SWEHarness


def create_swe_fitness_fn(model_name: str = "gpt-4o", n_workers: int = 6):
    """Create a fitness function for SWE tasks with Docker containers.
    
    Args:
        model_name: LiteLLM model name
        n_workers: Number of parallel workers
        
    Returns:
        fitness_fn: Function that evaluates candidates on batches
    """
    
    # Create harness pool - one per worker
    harness_pool = [SWEHarness() for _ in range(n_workers)]
    harness_available = [True] * n_workers
    harness_lock = threading.Lock()
    
    def get_harness() -> Tuple[int, SWEHarness]:
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
        skills: str,
        task_idx: int,
        total_tasks: int
    ) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
        """Process a single task and return (score, output, side_info)."""
        
        # Get a harness from the pool
        harness_idx, harness = get_harness()
        
        try:
            instance_id = task.get('instance_id', 'unknown')[:50]
            print(f"[Task {task_idx+1}/{total_tasks}] {instance_id}...", flush=True)
            
            # 1. Setup Docker Container
            harness.setup_task(task_instance=task)

            # 2. Run Agent
            problem = task["problem_statement"]
            patch, agent_trace, agent_metrics = harness.run_agent(
                problem, skills, model_name=model_name
            )
            
            # Debug: Quick summary of agent run
            print(f"  Agent finished: steps={agent_metrics.get('steps', 0)}, msgs={agent_metrics.get('num_messages', 0)}, patch_len={len(patch)}")

            # 3. Verify with tests
            has_patch = len(patch.strip()) > 0
            passed = False
            feedback_msg = ""
            test_output = ""

            if not has_patch:
                passed = False
                feedback_msg = "Agent did not produce any valid patch (git diff was empty)."
                test_output = "No patch to test."
                # Debug: Print first part of agent trace to see what happened
                print(f"  DEBUG: No patch produced. Agent trace preview:")
                print(f"  {agent_trace[:500]}..." if len(agent_trace) > 500 else f"  {agent_trace}")
            else:
                # FAIL_TO_PASS Tests
                fail_to_pass = task.get("FAIL_TO_PASS", [])
                if fail_to_pass:
                    fail_test_cmd = f"pytest {' '.join(fail_to_pass)} -v"
                else:
                    fail_test_cmd = "pytest"
                
                f2p_passed, f2p_output = harness.verify(test_cmd=fail_test_cmd)
                test_output = f"=== FAIL_TO_PASS TESTS ===\n{f2p_output}"
                
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
                        test_output += f"\n\n=== PASS_TO_PASS TESTS (sampled {sample_size}/{len(pass_to_pass)}) ===\n{p2p_output}"
                        
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
            
            # Rollout output
            output = {
                "patch": patch,
                "success": passed,
                "steps": agent_metrics.get("steps", 0),
                "estimated_tokens": agent_metrics.get("estimated_tokens", 0)
            }
            
            # Side info for reflection
            side_info = {
                "Input": {
                    "Task ID": instance_id,
                    "Problem": problem[:200] + "..." if len(problem) > 200 else problem,
                },
                "Generated Outputs": {
                    "Patch": patch[:500] + "..." if len(patch) > 500 else patch,
                    "Agent Trace": agent_trace[:1000] + "..." if len(agent_trace) > 1000 else agent_trace,
                },
                "Feedback": {
                    "Status": feedback_msg,
                    "Test Output": test_output[:1000] + "..." if len(test_output) > 1000 else test_output,
                },
                "scores": {
                    "correctness": score,
                }
            }
            
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  [{instance_id[:30]}] {status}", flush=True)
            
            # Cleanup
            harness.cleanup()
            
            return score, output, side_info
        
        finally:
            release_harness(harness_idx)
    
    def fitness_fn(
        candidate: Dict[str, str],
        batch: List[Dict[str, Any]]
    ) -> List[Tuple[float, Dict[str, Any], Dict[str, Any]]]:
        """Evaluate candidate on batch of tasks.
        
        Args:
            candidate: Dict with 'skills' key
            batch: List of task instances
            
        Returns:
            List of (score, output, side_info) tuples
        """
        skills = candidate["skills"]
        
        # Process tasks in parallel
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for idx, task in enumerate(batch):
                future = executor.submit(
                    process_single_task,
                    task,
                    skills,
                    idx,
                    len(batch)
                )
                futures.append(future)
            
            # Collect results in order
            results = [future.result() for future in futures]
        
        return results
    
    return fitness_fn
