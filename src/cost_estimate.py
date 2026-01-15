"""
Estimate API costs for different experiment configurations.
Helps avoid surprises with API spending.
"""

# Pricing as of Jan 2025 (per 1M tokens)
PRICING = {
    # OpenAI GPT-4 family
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},

    # Google Gemini (FREE tier has limits, then paid)
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30, "free_limit": "1500 requests/day"},
    "gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0, "free_limit": "Free tier (experimental)"},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},

    # Anthropic Claude
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

def estimate_tokens_per_task():
    """
    Estimate token usage per task based on typical SWE-smith tasks.
    """
    # Conservative estimates
    return {
        "problem_statement": 500,      # Task description
        "agent_turns": 10,              # Number of agent reasoning steps
        "tokens_per_turn": 1000,        # Agent reasoning + tool calls
        "code_context": 2000,           # Reading files from codebase
        "reflection_context": 5000,     # For GEPA reflection on failures
    }

def estimate_cost(
    model_name,
    num_tasks,
    num_generations,
    include_reflection=True
):
    """
    Estimate total API cost for an experiment.
    """

    if model_name not in PRICING:
        print(f"Warning: Pricing unknown for {model_name}, using GPT-4o as estimate")
        model_name = "gpt-4o"

    pricing = PRICING[model_name]
    tokens = estimate_tokens_per_task()

    # Tokens per task evaluation
    input_per_task = (
        tokens["problem_statement"] +
        tokens["code_context"] +
        (tokens["tokens_per_turn"] * tokens["agent_turns"] * 0.3)  # 30% input in conversation
    )

    output_per_task = (
        tokens["tokens_per_turn"] * tokens["agent_turns"] * 0.7  # 70% output (agent generates)
    )

    # Total evaluation cost
    total_evaluations = num_tasks * num_generations
    total_input_tokens = input_per_task * total_evaluations
    total_output_tokens = output_per_task * total_evaluations

    eval_cost = (
        (total_input_tokens / 1_000_000) * pricing["input"] +
        (total_output_tokens / 1_000_000) * pricing["output"]
    )

    # Reflection cost (GEPA uses LLM to reflect on failures)
    reflection_cost = 0
    if include_reflection:
        # Assume reflection on 30% of tasks (failures)
        num_reflections = int(total_evaluations * 0.3)
        reflection_input = tokens["reflection_context"]
        reflection_output = 500  # Generated mutation suggestions

        reflection_cost = (
            (reflection_input * num_reflections / 1_000_000) * pricing["input"] +
            (reflection_output * num_reflections / 1_000_000) * pricing["output"]
        )

    total_cost = eval_cost + reflection_cost

    return {
        "evaluation_cost": eval_cost,
        "reflection_cost": reflection_cost,
        "total_cost": total_cost,
        "total_evaluations": total_evaluations,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }

def print_cost_table():
    """Print cost estimates for different experiment sizes."""

    configs = [
        ("Smoke test", 2, 1),
        ("Small experiment", 20, 5),
        ("Medium experiment", 100, 10),
        ("Large experiment", 500, 20),
    ]

    models = ["gpt-4o-mini", "gpt-4o", "gemini-2.0-flash-exp", "gemini-1.5-flash", "claude-3-5-haiku"]

    print("=" * 80)
    print("COST ESTIMATES FOR GEPA + SWE-SMITH EXPERIMENTS")
    print("=" * 80)
    print("\nAssumptions:")
    print("  - ~10 agent turns per task")
    print("  - ~1000 tokens per turn (reasoning + tool calls)")
    print("  - GEPA reflects on ~30% of tasks (failures)")
    print("  - Costs are approximate and may vary")
    print("\n" + "=" * 80)

    for config_name, num_tasks, num_gens in configs:
        print(f"\n{config_name.upper()}")
        print(f"  Tasks: {num_tasks}, Generations: {num_gens}")
        print(f"  Total evaluations: {num_tasks * num_gens}")
        print(f"\n  {'Model':<25} {'Cost':<15} {'Notes':<30}")
        print(f"  {'-'*70}")

        for model in models:
            result = estimate_cost(model, num_tasks, num_gens)
            cost = result["total_cost"]

            notes = ""
            if "gemini" in model and "free_limit" in PRICING[model]:
                if result["total_evaluations"] < 1500:
                    notes = "✓ Within free tier"
                else:
                    notes = "⚠ Exceeds free tier"

            print(f"  {model:<25} ${cost:>6.2f}        {notes}")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
1. START WITH GEMINI (FREE):
   - gemini-1.5-flash has generous free tier (1500 requests/day)
   - Perfect for validation that GEPA is working
   - Run small/medium experiments first

2. VALIDATE IMPROVEMENT:
   - Use evaluate_prompts.py to confirm baseline vs optimized
   - Only move to paid models if you see improvement

3. THEN USE GPT-4O-MINI:
   - Very cheap ($0.15/$0.60 per 1M tokens)
   - Good quality for agent tasks
   - Medium experiment costs ~$5-10

4. RESERVE GPT-4O FOR FINAL RUNS:
   - Use for publication-quality results
   - Large experiment costs ~$50-100
   - Only after validating approach works
""")

if __name__ == "__main__":
    print_cost_table()
