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

### Key Insight: No Training Required!

This is **purely inference-time optimization**:
- mini-SWE-agent uses an LLM (GPT-4, Gemini, etc.) at inference time
- GEPA uses an LLM to reflect on failures and propose prompt improvements
- No model weights are trained—only prompts are evolved

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
├── implementation_plan.md   # Development roadmap
└── PRD.md                   # Product requirements document
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

### `PygmentsHarness` (src/harness.py)

Manages the execution of mini-SWE-agent on individual tasks:

```python
harness.setup_task(base_commit)    # Checkout bug-introducing commit
harness.run_agent(problem, prompt) # Run agent, get (patch, trace)
harness.verify(test_cmd)           # Run tests, get (passed, output)
```

### `PygmentsAdapter` (src/adapters/pygments_adapter.py)

Bridges GEPA and the harness, implementing the required interface:

- `evaluate()`: Run agent on batch of tasks, return scores + traces
- `make_reflective_dataset()`: Format feedback for GEPA reflection

The adapter captures **dual-source feedback** as described in GEPA paper Section 3.2:
1. **Agent reasoning traces** (LLM's chain of thought, tool calls)
2. **Environment feedback** (test errors, stack traces, compilation errors)

## 📊 Supported Models

This project uses LiteLLM, supporting multiple providers:

| Provider | Model | Free Tier |
|----------|-------|-----------|
| Google | `gemini/gemini-2.0-flash` | ✅ Yes |
| Groq | `groq/llama-3.3-70b-versatile` | ✅ Yes |
| OpenAI | `gpt-4o` | ❌ No |
| Anthropic | `claude-3-5-sonnet-20241022` | ❌ No |

## 📚 References

- **GEPA Paper**: [arXiv:2507.19457](https://arxiv.org/abs/2507.19457)
- **GEPA GitHub**: [github.com/gepa-ai/gepa](https://github.com/gepa-ai/gepa)
- **SWE-smith**: [swesmith.com](https://swesmith.com)
- **mini-SWE-agent**: [github.com/SWE-agent/mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)

## ⚠️ Notes

- **Docker + Linux required** for full SWE-smith execution environments
- **macOS compatibility**: Data loading and basic testing work, but full container-based execution requires Linux
- **API costs**: A full optimization run (~5 generations, ~10 tasks) costs approximately $1-5 depending on the model

---

*This project integrates three cutting-edge tools for AI-assisted software engineering research.*
