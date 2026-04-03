#!/usr/bin/env bash
# Run Phase 2 experiments on the desktop GPU via SSH.
#
# Usage:
#   ./scripts/run_on_desktop.sh              # Run all experiments
#   ./scripts/run_on_desktop.sh sync         # Only sync code, don't run
#   ./scripts/run_on_desktop.sh check        # Check desktop status
#   ./scripts/run_on_desktop.sh fetch        # Fetch results back to MacBook

set -euo pipefail

DESKTOP="umutakin-desktop"
REMOTE_DIR="~/workspace/ua/seds500-graduation-project"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[desktop]${NC} $1"; }
warn() { echo -e "${YELLOW}[desktop]${NC} $1"; }
err() { echo -e "${RED}[desktop]${NC} $1"; }

check_connection() {
    log "Checking connection to $DESKTOP..."
    if ! ssh -o ConnectTimeout=5 "$DESKTOP" "echo ok" &>/dev/null; then
        err "Cannot reach $DESKTOP. Is it on? Is Tailscale running?"
        echo "  Run: tailscale status"
        exit 1
    fi
    log "Connected."
}

check_env() {
    log "Checking desktop environment..."
    ssh "$DESKTOP" bash -l <<'EOF'
echo "=== System ==="
hostname
uname -a

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi not found"

echo ""
echo "=== Python ==="
python --version 2>/dev/null || python3 --version 2>/dev/null || echo "Python not found"

echo ""
echo "=== uv ==="
uv --version 2>/dev/null || echo "uv not found"

echo ""
echo "=== PyTorch CUDA ==="
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "PyTorch not importable"

echo ""
echo "=== Project dir ==="
if [ -d "$HOME/workspace/ua/seds500-graduation-project" ]; then
    echo "Project exists at ~/workspace/ua/seds500-graduation-project"
    cd "$HOME/workspace/ua/seds500-graduation-project"
    git log --oneline -3
else
    echo "Project NOT found"
fi
EOF
}

sync_code() {
    log "Syncing code to desktop..."

    # Sync source files (not data/checkpoints — those are large)
    rsync -avz --delete \
        --include='src/***' \
        --include='pyproject.toml' \
        --include='scripts/***' \
        --include='experiments/phase2/***' \
        --exclude='data/' \
        --exclude='checkpoints/' \
        --exclude='.venv/' \
        --exclude='__pycache__/' \
        --exclude='*.pt' \
        --exclude='*.pth' \
        --exclude='.git/' \
        "$LOCAL_DIR/" "$DESKTOP:$REMOTE_DIR/"

    log "Syncing dependencies..."
    ssh "$DESKTOP" bash -l <<EOF
cd $REMOTE_DIR
uv sync 2>&1 | tail -5
EOF

    log "Sync complete."
}

run_experiments() {
    local DATASET="${1:-all}"
    local METHOD="${2:-all}"

    log "Starting experiments: dataset=$DATASET method=$METHOD"

    # Run in a tmux session so it survives SSH disconnect
    ssh "$DESKTOP" bash -l <<EOF
cd $REMOTE_DIR

# Create or attach to tmux session
tmux has-session -t phase2 2>/dev/null && tmux kill-session -t phase2
tmux new-session -d -s phase2

tmux send-keys -t phase2 "cd $REMOTE_DIR && uv run python src/run_experiment.py --dataset $DATASET --method $METHOD --device cuda 2>&1 | tee experiments/phase2/run.log" C-m

echo "Experiments running in tmux session 'phase2'"
echo "To monitor: ssh $DESKTOP -t 'tmux attach -t phase2'"
echo "To check progress: ssh $DESKTOP 'tail -20 $REMOTE_DIR/experiments/phase2/run.log'"
EOF

    log "Experiments launched in background tmux session."
    log "Monitor with: ssh $DESKTOP -t 'tmux attach -t phase2'"
}

fetch_results() {
    log "Fetching results from desktop..."

    rsync -avz \
        "$DESKTOP:$REMOTE_DIR/experiments/phase2/" \
        "$LOCAL_DIR/experiments/phase2/"

    log "Results fetched to $LOCAL_DIR/experiments/phase2/"
}

# Main
case "${1:-run}" in
    check)
        check_connection
        check_env
        ;;
    sync)
        check_connection
        sync_code
        ;;
    run)
        check_connection
        sync_code
        run_experiments "${2:-all}" "${3:-all}"
        ;;
    fetch)
        check_connection
        fetch_results
        ;;
    monitor)
        ssh "$DESKTOP" -t "tmux attach -t phase2"
        ;;
    *)
        echo "Usage: $0 {check|sync|run|fetch|monitor}"
        echo ""
        echo "Commands:"
        echo "  check              Check desktop connection and environment"
        echo "  sync               Sync code to desktop"
        echo "  run [dataset] [method]  Sync + run experiments (default: all all)"
        echo "  fetch              Fetch results back to MacBook"
        echo "  monitor            Attach to tmux session on desktop"
        ;;
esac
