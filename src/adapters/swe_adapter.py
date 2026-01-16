from typing import Any, Mapping, Sequence, List, Dict
from dataclasses import dataclass
import json
import sys

from gepa.core.adapter import EvaluationBatch
from src.swe_harness import SWEHarness
from src.cost_tracker import get_tracker

# Define types
DataInst = Dict[str, Any]  # SWESmith task instance
Trajectory = Dict[str, Any] # Agent trace and logs
RolloutOutput = Dict[str, Any] # Patch and status

class SWEAdapter:
    def __init__(self, workspace_root: str = "/tmp/gepa_workenvs/pygments", model_name: str = "gpt-4o"):
        self.harness = SWEHarness(workspace_root=workspace_root)
        self.model_name = model_name
        self.propose_new_texts = None # Default GEPA proposer
        self.tracker = get_tracker()
        self.task_count = 0


    def evaluate(
        self,
        batch: List[DataInst],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, RolloutOutput]:
        
        outputs: List[RolloutOutput] = []
        scores: List[float] = []
        trajectories: List[Trajectory] | None = [] if capture_traces else None
        
        system_prompt = candidate.get("system_prompt", "You are a helpful software engineering assistant.")

        print(f"\n{'='*70}")
        print(f"Evaluating batch of {len(batch)} tasks...")
        print(f"{'='*70}\n")

        for i, task in enumerate(batch):
            self.task_count += 1
            print(f"[Task {i+1}/{len(batch)}] {task.get('instance_id', 'unknown')}", end="", flush=True)
            # 1. Setup Environment
            # SWESmith 'repo' field often looks like 'swesmith/pygments__pygments.27649ebb'
            # We can extract the commit hash from there if base_commit is missing.
            base_commit = task.get("base_commit")
            
            if not base_commit:
                # Try to parse from repo name or instance_id
                # Example repo: swesmith/pygments__pygments.27649ebb
                repo_field = task.get("repo", "")
                if "." in repo_field:
                    base_commit = repo_field.split(".")[-1]
                else:
                    # Fallback to HEAD if we really can't find it (dangerous but keeps it running)
                    print(f"WARNING: Could not determine base commit for {task.get('instance_id')}. Using HEAD.")
                    base_commit = "master"
            
            # Get the bug patch from the task (SWE-smith's synthetic bug)
            bug_patch = task.get("patch", "")

            self.harness.setup_task(base_commit, bug_patch=bug_patch)

            # 2. Run Agent
            problem = task["problem_statement"]
            # harness.run_agent returns: (patch, agent_reasoning_trace)

            # Estimate tokens for logging (rough approximation)
            input_tokens = len(system_prompt.split()) + len(problem.split()) * 3

            patch, agent_reasoning_trace = self.harness.run_agent(problem, system_prompt, model_name=self.model_name)

            # Estimate output tokens
            output_tokens = len(agent_reasoning_trace.split()) + len(patch.split())

            # Log API usage
            self.tracker.log_api_call(
                model_name=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                operation=f"agent_run_task_{self.task_count}"
            )

            # 3. Verify with tests (captures environment feedback)
            has_patch = len(patch.strip()) > 0

            passed = False
            feedback_msg = ""
            test_verification_output = ""

            if not has_patch:
                passed = False
                feedback_msg = "Agent did not produce any valid patch (git diff was empty)."
                test_verification_output = "No patch to test."
            else:
                # === FAIL_TO_PASS Tests ===
                # These tests should fail before the fix and pass after
                fail_to_pass = task.get("FAIL_TO_PASS", [])
                if fail_to_pass:
                    fail_test_cmd = f"pytest {' '.join(fail_to_pass)} -v"
                else:
                    print(f"  WARNING: No FAIL_TO_PASS tests for {task.get('instance_id')}. Using generic pytest.")
                    fail_test_cmd = "pytest"
                
                # Run FAIL_TO_PASS tests
                f2p_passed, f2p_output = self.harness.verify(test_cmd=fail_test_cmd)
                test_verification_output = f"=== FAIL_TO_PASS TESTS ===\n{f2p_output}"
                
                if not f2p_passed:
                    passed = False
                    feedback_msg = "FAIL_TO_PASS tests failed - the fix does not solve the issue."
                else:
                    # === PASS_TO_PASS Tests (Regression Check) ===
                    # These tests should continue to pass after the fix (no regressions)
                    pass_to_pass = task.get("PASS_TO_PASS", [])
                    
                    if pass_to_pass:
                        # Sample up to 10 tests to avoid timeout (full test would take too long)
                        sample_size = min(10, len(pass_to_pass))
                        sampled_tests = pass_to_pass[:sample_size]
                        pass_test_cmd = f"pytest {' '.join(sampled_tests)} -v --timeout=60"
                        
                        # Run PASS_TO_PASS tests
                        p2p_passed, p2p_output = self.harness.verify(test_cmd=pass_test_cmd)
                        test_verification_output += f"\n\n=== PASS_TO_PASS TESTS (sampled {sample_size}/{len(pass_to_pass)}) ===\n{p2p_output}"
                        
                        if not p2p_passed:
                            passed = False
                            feedback_msg = f"PASS_TO_PASS regression failed - the fix breaks existing tests."
                        else:
                            passed = True
                            feedback_msg = "All tests passed (FAIL_TO_PASS fixed + PASS_TO_PASS still passing)."
                    else:
                        # No PASS_TO_PASS tests provided - just use FAIL_TO_PASS result
                        passed = True
                        feedback_msg = "FAIL_TO_PASS tests passed. No PASS_TO_PASS tests to check."

            score = 1.0 if passed else 0.0

            outputs.append({"patch": patch, "success": passed})
            scores.append(score)

            # Print result
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f" - {status}")
            sys.stdout.flush()

            if capture_traces:
                trajectories.append({
                    "agent_reasoning_trace": agent_reasoning_trace,
                    "test_verification_output": test_verification_output,
                    "patch": patch,
                    "feedback": feedback_msg,
                    "instance_id": task["instance_id"],
                    "problem_statement": task["problem_statement"]
                })
                
            # Cleanup
            self.harness.cleanup()

        # Print batch summary
        pass_count = sum(1 for s in scores if s == 1.0)
        print(f"\nBatch complete: {pass_count}/{len(batch)} passed ({pass_count/len(batch)*100:.1f}%)")
        print(f"{'='*70}\n")

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories
        )

    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch: EvaluationBatch[Trajectory, RolloutOutput],
        components_to_update: List[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """
        Prepare reflection data for GEPA.

        This combines TWO sources of feedback (as per GEPA paper Section 3.2):
        1. Agent reasoning traces (LLM's chain of thought, tool calls, decisions)
        2. Environment/evaluation feedback (test output, errors, stack traces)

        GEPA uses this rich textual feedback to reflect on failures and propose
        targeted prompt improvements - this is what gives it an advantage over RL.
        """
        updates = []

        if not eval_batch.trajectories:
            return {"system_prompt": []}

        for i, traj in enumerate(eval_batch.trajectories):
            # Combine agent reasoning + environment feedback
            # Truncate to avoid token limits while preserving diagnostic info
            agent_trace_truncated = traj["agent_reasoning_trace"][:10000]
            test_output_truncated = traj["test_verification_output"][:5000]

            record = {
                "Inputs": {
                    "Task ID": traj["instance_id"],
                    "Problem Statement": traj["problem_statement"][:500] + "..."
                },
                "Generated Outputs": {
                    "Agent Reasoning & Actions": agent_trace_truncated,
                    "Patch Generated": traj["patch"][:2000] if traj["patch"] else "No patch generated"
                },
                "Environment Feedback": {
                    "Test Verification Output": test_output_truncated,
                    "Status": traj['feedback'],
                    "Score": eval_batch.scores[i]
                }
            }
            updates.append(record)

        return {"system_prompt": updates}
