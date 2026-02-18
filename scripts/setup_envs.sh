#!/bin/bash
set -e

# Directory to hold the repositories
WORK_DIR="/tmp/gepa_workenvs"
REPO_URL="https://github.com/pygments/pygments.git"
REPO_DIR="$WORK_DIR/pygments"

echo "Setting up work environment in $WORK_DIR..."
mkdir -p "$WORK_DIR"

if [ -d "$REPO_DIR" ]; then
    echo "Repo already exists at $REPO_DIR. Cleaning..."
    cd "$REPO_DIR"
    git reset --hard HEAD
    git clean -fd
    git fetch origin
else
    echo "Cloning Pygments..."
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

echo "Installing Pygments in edit mode..."
# We use the parent virtualenv so the agent can run tests
pip install -e .

echo "Verifying test runner (pytest)..."
if command -v pytest > /dev/null; then
    echo "pytest is available."
else
    echo "Installing pytest..."
    pip install pytest
fi

echo "Ready! Repository is located at: $REPO_DIR"
