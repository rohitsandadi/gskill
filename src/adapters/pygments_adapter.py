from typing import Any, Mapping, Sequence, List, Dict
from dataclasses import dataclass
import json

from gepa.core.adapter import EvaluationBatch
from src.harness import PygmentsHarness

# Define types
DataInst = Dict[str, Any]  # SWESmith task instance
Trajectory = Dict[str, Any] # Agent trace and logs
RolloutOutput = Dict[str, Any] # Patch and status

class PygmentsAdapter:
    def __init__(self, workspace_root: str = "/tmp/gepa_workenvs/pygments", model_name: str = "gpt-4o"):
        self.harness = PygmentsHarness(workspace_root=workspace_root)
        self.model_name = model_name
        self.propose_new_texts = None # Default GEPA proposer


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

        for task in batch:
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
            
            self.harness.setup_task(base_commit)
            
            # 2. Run Agent
            problem = task["problem_statement"]
            # harness.run_agent returns: (patch, agent_reasoning_trace)
            patch, agent_reasoning_trace = self.harness.run_agent(problem, system_prompt, model_name=self.model_name)

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
                # Extract test command from task if available, default to pytest
                test_cmd = task.get("test_cmd", "pytest")
                # verify now returns (passed, test_output)
                passed, test_verification_output = self.harness.verify(test_cmd=test_cmd)
                feedback_msg = "Tests passed." if passed else "Tests failed after applying patch."

            score = 1.0 if passed else 0.0

            outputs.append({"patch": patch, "success": passed})
            scores.append(score)

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
