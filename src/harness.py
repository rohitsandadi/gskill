import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple

from minisweagent.agents.default import DefaultAgent
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.environments.local import LocalEnvironment

@dataclass
class TaskResult:
    passed: bool
    trace: str
    output: str

class PygmentsHarness:
    def __init__(self, workspace_root: str = "/tmp/gepa_workenvs/pygments"):
        self.workspace_root = workspace_root
        
    def setup_task(self, base_commit: str):
        """Checkout the specific commit for the task."""
        # Reset any changes
        subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=self.workspace_root, check=True)
        subprocess.run(["git", "clean", "-fd"], cwd=self.workspace_root, check=True)
        # Checkout commit
        subprocess.run(["git", "checkout", base_commit], cwd=self.workspace_root, check=True)

    def run_agent(self, problem_statement: str, system_prompt: str, model_name: str = "gemini/gemma-3-4b-it") -> Tuple[str, str]:
        """Run the agent and return (patch, conversation_trace)."""
        
        # Initialize Agent with the optimized prompt
        # system_template is passed to DefaultAgent which uses it to override AgentConfig
        agent = DefaultAgent(
            model=LitellmModel(model_name=model_name),
            env=LocalEnvironment(cwd=self.workspace_root),
            system_template=system_prompt
        )

        try:
            # We wrap in try/except to ensure we capture trace even if it crashes
            result = agent.run(problem_statement)

            # Extract the full conversation trace from agent.messages
            # This contains the agent's reasoning, actions, and tool outputs
            trace = "\n\n".join([
                f"[{msg['role'].upper()}]\n{msg['content']}"
                for msg in agent.messages
            ])

            # Generate patch of changes
            diff_proc = subprocess.run(
                ["git", "diff"],
                cwd=self.workspace_root,
                capture_output=True,
                text=True
            )
            return diff_proc.stdout, trace
            
        except Exception as e:
            return "", f"Agent crashed: {str(e)}"

    def verify(self, test_cmd: str = "pytest") -> Tuple[bool, str]:
        """Run verification tests and return (passed, output).

        Returns both the pass/fail status and the full test output,
        which includes rich diagnostic information like:
        - Test failure messages
        - Stack traces
        - Compilation errors
        - Expected vs actual values
        """
        try:
            proc = subprocess.run(
                test_cmd,
                cwd=self.workspace_root,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60 # Fail fast
            )

            # Combine stdout and stderr for full diagnostic output
            test_output = f"=== TEST COMMAND: {test_cmd} ===\n"
            test_output += f"=== RETURN CODE: {proc.returncode} ===\n\n"

            if proc.stdout:
                test_output += "=== STDOUT ===\n" + proc.stdout + "\n"
            if proc.stderr:
                test_output += "=== STDERR ===\n" + proc.stderr + "\n"

            passed = proc.returncode == 0
            return passed, test_output

        except subprocess.TimeoutExpired:
            return False, f"TEST TIMEOUT: Command '{test_cmd}' exceeded 60 second limit"

    def cleanup(self):
        """Cleanup after run."""
        subprocess.run(["git", "checkout", "master"], cwd=self.workspace_root, check=False)
