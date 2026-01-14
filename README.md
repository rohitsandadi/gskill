# GEPA + SWE-smith Integration

> **Research Project**: Using GEPA (Genetic-Pareto) prompt optimization for inference-time search on software engineering tasks.

## 🎯 Research Question

> Can training GEPA on problems from one repository produce effective prompts for held-out test problems from the same repository?

This follows the **inference-time search** approach from [GEPA paper Section 6](https://arxiv.org/abs/2507.19457).

## 📋 Project Overview

| Component | Description | Role |
|-----------|-------------|------|
| **GEPA** | Genetic-Pareto prompt optimization framework | Evolves prompts using LLM reflection |
| **SWE-smith** | Dataset of 52k+ GitHub issues | Provides training/test tasks |
| **mini-SWE-agent** | Lightweight 100-line coding agent | Executes tasks with optimized prompts |


## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GEPA Optimization Loop                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Load SWE-smith tasks (filtered to single repo)          │
│  2. Seed with baseline system prompt                        │
│  3. For each generation:                                    │
│     a. Evaluate prompt on tasks using mini-SWE-agent        │
│     b. Collect rich feedback (agent traces + test output)   │
│     c. GEPA reflects on failures                            │
│     d. Mutate prompt based on reflection                    │
│  4. Output: Optimized prompt for that repository            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
GEPA+SWESMITH/
├── train.py                 # Main entry point for GEPA optimization
├── src/
│   ├── harness.py           # PygmentsHarness: runs agent on tasks
│   └── adapters/
│       └── pygments_adapter.py  # GEPA adapter for SWE-smith
├── examples/                # Example scripts and utilities
├── scripts/                 # Setup scripts
├── gepa_results/            # Output directory for optimization runs
├── requirements.txt         # Python dependencies
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy template and add your API key
cp .env.example .env
# Edit .env with your GOOGLE_API_KEY (Gemini) or OPENAI_API_KEY
```

### 3. Run Smoke Test

```bash
python train.py --smoke-test
```

### 4. Run Full Optimization

```bash
python train.py --generations 5 --train-size 10
```

## 🔧 Core Components

### `Harness` (src/harness.py)

Manages the execution of mini-SWE-agent on individual tasks:

```python
harness.setup_task(base_commit)    # Checkout bug-introducing commit
harness.run_agent(problem, prompt) # Run agent, get (patch, trace)
harness.verify(test_cmd)           # Run tests, get (passed, output)
```

### `Adapter` (src/adapters/pygments_adapter.py)

Bridges GEPA and the harness, implementing:

- `evaluate()`: Run agent on batch of tasks, return scores + traces
- `make_reflective_dataset()`: Format feedback for GEPA reflection

The adapter captures **dual-source feedback** as described in GEPA paper Section 3.2:
1. **Agent reasoning traces** (LLM's chain of thought, tool calls)
2. **Environment feedback** (test errors, stack traces, compilation errors)


## 📚 References

- **GEPA Paper**: [arXiv:2507.19457](https://arxiv.org/abs/2507.19457)
- **GEPA GitHub**: [github.com/gepa-ai/gepa](https://github.com/gepa-ai/gepa)
- **SWE-smith**: [swesmith.com](https://swesmith.com)
- **mini-SWE-agent**: [github.com/SWE-agent/mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)



