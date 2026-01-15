"""
Real-time cost and API call tracking during GEPA optimization.
"""

import time
import json
from pathlib import Path
from datetime import datetime

class CostTracker:
    def __init__(self, log_dir="gepa_results/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.session_start = datetime.now()
        self.api_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.estimated_cost = 0.0

        self.log_file = self.log_dir / f"cost_log_{self.session_start.strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.summary_file = self.log_dir / "cost_summary.txt"

        # Model pricing (per 1M tokens)
        self.pricing = {
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gemini-2.5-flash-lite": {"input": 0.0, "output": 0.0},
            "gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0},
            "gemma-3-27b": {"input": 0.0, "output": 0.0},  # Free tier
        }

    def log_api_call(self, model_name, input_tokens, output_tokens, operation="unknown"):
        """Log a single API call."""
        self.api_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        # Estimate cost
        pricing = self.pricing.get(model_name, {"input": 0.0, "output": 0.0})
        call_cost = (
            (input_tokens / 1_000_000) * pricing["input"] +
            (output_tokens / 1_000_000) * pricing["output"]
        )
        self.estimated_cost += call_cost

        # Log entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "call_number": self.api_calls,
            "model": model_name,
            "operation": operation,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "call_cost": call_cost,
            "cumulative_cost": self.estimated_cost
        }

        # Write to JSONL log
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        # Update summary
        self.write_summary()

    def write_summary(self):
        """Write human-readable summary."""
        elapsed = (datetime.now() - self.session_start).total_seconds()

        summary = f"""
================================================================================
GEPA COST TRACKER - Real-time Summary
================================================================================

Session Start: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}
Elapsed Time:  {elapsed/60:.1f} minutes

API Calls:     {self.api_calls}
Input Tokens:  {self.total_input_tokens:,}
Output Tokens: {self.total_output_tokens:,}
Total Tokens:  {self.total_input_tokens + self.total_output_tokens:,}

Estimated Cost: ${self.estimated_cost:.4f}

Rate: {self.api_calls / (elapsed / 60):.1f} calls/min

Log file: {self.log_file}
================================================================================
"""

        with open(self.summary_file, 'w') as f:
            f.write(summary)

        return summary

    def get_summary(self):
        """Return current summary string."""
        return self.write_summary()

# Global tracker instance
_tracker = None

def get_tracker():
    """Get or create global tracker."""
    global _tracker
    if _tracker is None:
        _tracker = CostTracker()
    return _tracker

def log_call(model_name, input_tokens, output_tokens, operation="unknown"):
    """Convenience function to log a call."""
    tracker = get_tracker()
    tracker.log_api_call(model_name, input_tokens, output_tokens, operation)
