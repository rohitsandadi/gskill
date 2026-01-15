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
├── requirements.txt              # Python dependencies
├── GUIDE.md                      # Quick reference (no fluff)
├── README.md                     # This file
├── QUICKSTART.md                 # Detailed workflow
├── CLAUDE.md                     # Technical documentation
│
├── src/                          # All source code
│   ├── train.py                 # Main: GEPA optimization
│   ├── split_dataset.py         # Create train/val/test splits
│   ├── evaluate_prompts.py      # Validate improvements
│   ├── cost_estimate.py         # Estimate API costs
│   ├── harness.py               # Runs mini-SWE-agent on tasks
│   ├── cost_tracker.py          # Track API costs during runs
│   └── adapters/
│       └── pygments_adapter.py  # GEPA adapter for SWE-smith
│
├── data/                         # Dataset (generated)
│   ├── pygments_train.json      # ~780 tasks (65%)
│   ├── pygments_val.json        # ~60 tasks (5%)
│   └── pygments_test.json       # ~360 tasks (30%)
│
├── gepa_results/                 # Optimization outputs
├── examples/                     # Example scripts
├── scripts/                      # Setup scripts
└── archive/                      # Old development files
```

## 🚀 Quick Start

### ⚡ New: Cost-Aware Workflow

**IMPORTANT**: Start with the free tier to validate GEPA works before spending money!

```bash
# 1. See estimated costs for different experiment sizes
python src/cost_estimate.py

# 2. Setup environment (one-time)
bash scripts/setup_envs.sh
python src/split_dataset.py  # Creates ~780 train, ~60 val, ~360 test

# 3. Validate with FREE Gemini (no cost!)
python src/train.py --use-split --train-size 10 --generations 3

# 4. Check if GEPA actually improved the prompts
python src/evaluate_prompts.py --split val --limit 10

# 5. If it works, scale up!
python src/train.py --use-split --train-size 100 --generations 10 --model "gpt-4o-mini"
```

📖 **See [QUICKSTART.md](QUICKSTART.md) for detailed walkthrough**

### 📚 Documentation

- **[GUIDE.md](GUIDE.md)** - Quick reference (no fluff)
- **[QUICKSTART.md](QUICKSTART.md)** - Detailed workflow
- **[CLAUDE.md](CLAUDE.md)** - Technical documentation

### 🛠️ Key Scripts (in src/)

- **train.py** - Run GEPA optimization
- **split_dataset.py** - Create train/val/test splits
- **evaluate_prompts.py** - Validate improvements
- **cost_estimate.py** - Estimate API costs

### 1. Setup Environment

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your GOOGLE_API_KEY (Gemini - FREE!) or OPENAI_API_KEY
```

### 2. Prepare Dataset

```bash
# Clone Pygments repository
bash scripts/setup_envs.sh

# Split into train/val/test (creates ~360 test samples)
python src/split_dataset.py
```

### 3. Run GEPA Optimization

```bash
# Start with free tier validation
python src/train.py --use-split --train-size 10 --generations 3

# Evaluate improvement
python src/evaluate_prompts.py --split val --limit 10
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



