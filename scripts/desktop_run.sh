#!/usr/bin/env bash
# Launch/monitor/check experiments on the desktop GPU.
#
# Usage:
#   ./scripts/desktop_run.sh launch    # Start experiments (detached)
#   ./scripts/desktop_run.sh status    # Check progress
#   ./scripts/desktop_run.sh results   # Show results summary
#   ./scripts/desktop_run.sh privacy   # Show privacy AUCs
#   ./scripts/desktop_run.sh fetch     # Copy results to MacBook
#   ./scripts/desktop_run.sh kill      # Stop running experiments
#   ./scripts/desktop_run.sh cleanup   # Clean old files

set -euo pipefail
DESKTOP="ydran@umutakin-desktop"
PROJ='C:\Users\ydran\workspace\seds\seds500-graduation-project'
SSH="ssh -o ConnectTimeout=10 $DESKTOP"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

strip() { grep -v "WARNING\|vulnerable\|upgraded\|Shim\|oh-my-posh"; }

ps() { $SSH "powershell -Command \"$1\"" 2>&1 | strip; }

case "${1:-status}" in
    launch)
        echo "Launching experiments on desktop (detached)..."
        ps "Start-Process powershell -ArgumentList '-ExecutionPolicy','Bypass','-File','${PROJ}\run_final.ps1' -WindowStyle Hidden -PassThru | Select-Object Id"
        sleep 10
        $0 status
        ;;

    status)
        echo "=== GPU ==="
        ps "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
        echo ""
        echo "=== Experiments ==="
        ps "(Get-ChildItem '${PROJ}\experiments\phase2\*\RESULTS.json' -ErrorAction SilentlyContinue).Count.ToString() + ' of 44 complete'"
        echo ""
        echo "=== Process ==="
        ps "if (Test-Path '${PROJ}\experiments\phase2\.running') { Get-Content '${PROJ}\experiments\phase2\.running' } else { 'Not running' }"
        echo ""
        ps "nvidia-smi 2>&1 | Select-String 'python'" || echo "No python on GPU"
        ;;

    results)
        echo "=== Replacement Utility (% of baseline) ==="
        ps "
            Get-ChildItem '${PROJ}\experiments\phase2\*\RESULTS.json' | ForEach-Object {
                \\\$name = \\\$_.Directory.Name
                \\\$json = Get-Content \\\$_.FullName | ConvertFrom-Json
                \\\$repl = \\\$json.utility.summary.replacement.pct_of_baseline
                \\\$aug = \\\$json.utility.summary.augmentation.pct_of_baseline
                \\\$priv = \\\$json.privacy.attack_auc
                Write-Output ('{0,-40} Repl:{1,8:F1}%  Aug:{2,8:F1}%  Privacy:{3,7:F4}' -f \\\$name, \\\$repl, \\\$aug, \\\$priv)
            } | Sort-Object
        "
        ;;

    privacy)
        echo "=== Privacy AUCs ==="
        ps "
            Get-ChildItem '${PROJ}\experiments\phase2\*\RESULTS.json' | ForEach-Object {
                \\\$name = \\\$_.Directory.Name
                \\\$json = Get-Content \\\$_.FullName | ConvertFrom-Json
                \\\$auc = \\\$json.privacy.attack_auc
                \\\$interp = \\\$json.privacy.interpretation
                Write-Output ('{0,-40} AUC:{1,7:F4}  {2}' -f \\\$name, \\\$auc, \\\$interp)
            } | Sort-Object
        "
        ;;

    fetch)
        echo "Fetching results..."
        scp -r -o ConnectTimeout=10 "${DESKTOP}:${PROJ//\\//}/experiments/phase2/*" "$LOCAL_DIR/experiments/phase2/" 2>&1 | strip | tail -5
        echo "Fetched to $LOCAL_DIR/experiments/phase2/"
        ;;

    kill)
        echo "Killing experiments..."
        ps "Get-Process python* -ErrorAction SilentlyContinue | Stop-Process -Force; 'Done'"
        ;;

    cleanup)
        "$(dirname "$0")/desktop_cleanup.sh"
        ;;

    *)
        echo "Usage: $0 {launch|status|results|privacy|fetch|kill|cleanup}"
        ;;
esac
