# Implementation Plan: GEPA + SWESmith Integration

## Phase 1: Environment & Data Preparation
**Goal**: Ensure the `pygments` repository is locally available and we can efficiently load tasks from SWESmith.

1.  **Repo Setup Script** (`scripts/setup_repo.py`):
    *   Clones `https://github.com/pygments/pygments`.
    *   Creates a stable environment (virtualenv) for running its tests.
    *   Verifies `pytest` runs correctly on the HEAD of the repo.

2.  **Data Loader Module** (`src/data_loader.py`):
    *   Uses `datasets` to stream `SWE-bench/SWE-smith`.
    *   Filters for `repo == 'swesmith/pygments__pygments'`.
    *   Implements a deterministic Train/Test split (e.g., 80/20 based on instance ID hash).
    *   *Output*: Returns list of `TaskInstance` objects.

## Phase 2: The GEPA Adapter
**Goal**: Connect GEPA's abstract `evolve` loop to the concrete execution of `mini-swe-agent`.

1.  **Adapter Class** (`src/adapters/pygments_adapter.py`):
    *   Inherits from `gepa.GEPAAdapter`.
    *   **Method `evaluate(candidate, inputs)`**:
        *   Iterates over `inputs` (Batch of SWESmith tasks).
        *   For each task:
            *   Resets the local `pygments` repo to `base_commit`.
            *   Instantiates `DefaultAgent` (MiniSWE) with `candidate['system_prompt']`.
            *   Runs agent on `problem_statement`.
            *   Applies patch.
            *   Runs verification tests.
    *   **Method `get_traces`**: Returns the conversation history for GEPA reflection.

## Phase 3: Optimization Loop Construction
**Goal**: Create the main entry point that runs the evolution.

1.  **Training Script** (`train.py`):
    *   Loads data using `Data Loader`.
    *   Initializes `PygmentsAdapter`.
    *   Configures GEPA (Proposer: Reflective, Population Size: Small to start).
    *   Runs `gepa.optimize(...)`.
    *   Saves the detailed log of generations and the final Pareto front of prompts.

## Phase 4: Evaluation & Analysis
**Goal**: Verify the results on the hold-out test set.

1.  **Evaluation Script** (`evaluate.py`):
    *   Loads the *Best Prompt* found in Phase 3.
    *   Loads the *Default Prompt* as a baseline.
    *   Runs both against the `Test Set` from Phase 1.
    *   Generates a report comparing Pass Rates and Costs.

## Timeline / Step-by-Step
1.  [ ] **Step 1**: Write `scripts/setup_pygments.sh` to clone and prep the target repo.
2.  [ ] **Step 2**: Create `src/harness.py` containing the core logic to run one agent task on one repository state.
3.  [ ] **Step 3**: Implement `src/adapters/pygments_adapter.py` connecting the Harness to GEPA.
4.  [ ] **Step 4**: Write `train.py` and run a "Smoke Test" (1 generation, 2 tasks).
5.  [ ] **Step 5**: Run full training (e.g., 10 generations).
