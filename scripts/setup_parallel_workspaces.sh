#!/bin/bash
# Setup script for parallel GEPA workers
# Creates multiple pygments workspace clones for parallel task execution

set -e

BASE_DIR="/tmp/gepa_workenvs"
REPO_URL="https://github.com/pygments/pygments.git"
N_WORKERS=${1:-6}  # Default 6 workers

echo "=============================================="
echo "Setting up $N_WORKERS parallel workspaces"
echo "=============================================="

# Create base directory
mkdir -p "$BASE_DIR"

# Check if master clone exists
if [ ! -d "$BASE_DIR/pygments" ]; then
    echo "Cloning master pygments repo..."
    git clone "$REPO_URL" "$BASE_DIR/pygments"
else
    echo "Master repo exists, updating..."
    (cd "$BASE_DIR/pygments" && git fetch origin)
fi

# Create worker clones
for i in $(seq 0 $((N_WORKERS - 1))); do
    WORKER_DIR="$BASE_DIR/pygments_$i"
    
    if [ ! -d "$WORKER_DIR" ]; then
        echo "Creating worker $i: $WORKER_DIR"
        # Use local clone for speed (--reference)
        git clone --reference "$BASE_DIR/pygments" "$REPO_URL" "$WORKER_DIR"
    else
        echo "Worker $i exists: $WORKER_DIR"
    fi
done

echo ""
echo "=============================================="
echo "Setup complete! Created $N_WORKERS workspaces:"
for i in $(seq 0 $((N_WORKERS - 1))); do
    echo "  - $BASE_DIR/pygments_$i"
done
echo "=============================================="
echo ""
echo "Disk usage:"
du -sh "$BASE_DIR"/*
