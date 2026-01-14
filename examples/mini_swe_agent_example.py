"""
Mini-SWE-Agent Example

This script demonstrates how to use mini-SWE-agent programmatically
with Python bindings. Mini-SWE-agent is a lightweight 100-line agent
that achieves 74%+ on SWE-bench.

Requires: OPENAI_API_KEY or ANTHROPIC_API_KEY in environment
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Check for API key
api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    print("=" * 60)
    print("API KEY REQUIRED")
    print("=" * 60)
    print()
    print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in your environment.")
    print("You can:")
    print("  1. Copy .env.example to .env and add your key")
    print("  2. Or export the key: export OPENAI_API_KEY='your-key'")
    print()
    exit(1)


def main():
    """
    Demonstrate mini-SWE-agent usage.
    
    For interactive use, run from terminal:
        mini        # Simple text interface
        mini -v     # Visual UI interface
    """
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.models.litellm import LitellmModel
    from minisweagent.environments.local import LocalEnvironment
    
    print("=" * 60)
    print("MINI-SWE-AGENT EXAMPLE")
    print("=" * 60)
    print()
    
    # Choose model based on available API key
    if os.environ.get("ANTHROPIC_API_KEY"):
        model_name = "claude-sonnet-4-20250514"
        print(f"Using Anthropic: {model_name}")
    else:
        model_name = "gpt-4o"
        print(f"Using OpenAI: {model_name}")
    
    print()
    print("Initializing agent...")
    
    # Create the agent with a local environment
    agent = DefaultAgent(
        model=LitellmModel(model_name=model_name),
        environment=LocalEnvironment(),
    )
    
    # Example: Ask the agent to do a simple task
    task = "List all Python files in the current directory and show their sizes."
    
    print(f"Task: {task}")
    print()
    print("Running agent...")
    print("-" * 60)
    
    # Run the agent
    result = agent.run(task)
    
    print("-" * 60)
    print()
    print("Agent completed!")
    print()
    print("For interactive sessions, use the CLI:")
    print("  mini        # Simple text interface")
    print("  mini -v     # Visual UI interface")


if __name__ == "__main__":
    main()
