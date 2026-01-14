"""
GEPA Prompt Optimization Example

This script demonstrates how to use GEPA (Genetic-Pareto) to optimize
prompts for LLM-based systems. GEPA uses evolutionary algorithms with
LLM reflection to systematically improve prompts.

Requires: OPENAI_API_KEY or ANTHROPIC_API_KEY in environment
"""

import os

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
    Demonstrate GEPA prompt optimization.
    
    This example shows the basic structure of using GEPA to optimize
    a simple classification prompt.
    """
    print("=" * 60)
    print("GEPA PROMPT OPTIMIZATION EXAMPLE")
    print("=" * 60)
    print()
    
    # Import GEPA components
    try:
        import gepa
        print(f"GEPA version: {gepa.__version__ if hasattr(gepa, '__version__') else 'installed'}")
    except ImportError:
        print("GEPA not installed. Run: pip install gepa")
        return
    
    print()
    print("GEPA Overview:")
    print("-" * 60)
    print("""
GEPA (Genetic-Pareto) optimizes text components of systems using:

1. Reflective Evolution: LLMs analyze execution traces and propose
   targeted improvements to prompts, code, or specifications.

2. Pareto Optimization: Balances multiple objectives (accuracy, cost,
   latency) to find the best trade-offs.

3. Iterative Refinement: Evolves prompts through mutation, crossover,
   and selection over multiple generations.

Key Use Cases:
- Optimizing classification/extraction prompts
- Improving agent system prompts  
- Fine-tuning code generation instructions
- Multi-objective prompt balancing

For detailed usage, see:
- GitHub: https://github.com/gepa-ai/gepa
- Paper: https://arxiv.org/abs/2507.19457

DSPy Integration (recommended):
  GEPA integrates with DSPy for easy prompt optimization.
  See the GEPA docs for DSPy examples.
""")
    
    print("-" * 60)
    print()
    print("To run a full optimization experiment, see the GEPA documentation:")
    print("https://github.com/gepa-ai/gepa#using-gepa-to-optimize-your-system")


if __name__ == "__main__":
    main()
