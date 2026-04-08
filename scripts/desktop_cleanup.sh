#!/usr/bin/env bash
# Clean up old runner scripts, caches, and logs on the desktop.
# Run from MacBook: ./scripts/desktop_cleanup.sh

set -euo pipefail
DESKTOP="ydran@umutakin-desktop"
PROJ="C:\\Users\\ydran\\workspace\\seds\\seds500-graduation-project"

echo "Cleaning up desktop..."

# Upload a PowerShell cleanup script then run it
cat > /tmp/cleanup.ps1 << 'PSEOF'
$proj = "C:\Users\ydran\workspace\seds\seds500-graduation-project"

# Kill python
Get-Process python* -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Output "Killed python"

# Old scripts
$old = @(
    "scripts\run_all_phase2.py", "scripts\run_ames.py", "scripts\run_final.py",
    "scripts\run_remaining.py", "scripts\run_remaining2.py", "scripts\run_rest.py",
    "scripts\test_adult.py", "scripts\test_news.py", "scripts\test_single.py",
    "scripts\test_resume.py", "scripts\gpu_verify.py", "scripts\smoke_test.py",
    "run_debug.bat", "run_remaining.bat", "run_rest.bat", "launch_phase2.bat",
    "check_env.py", "read_results.bat", "read_privacy.bat",
    "experiments\phase2\.running"
)
foreach ($f in $old) {
    $path = Join-Path $proj $f
    if (Test-Path $path) { Remove-Item $path -Force; Write-Output "  Deleted $f" }
}

# Old logs
Get-ChildItem (Join-Path $proj "experiments\phase2\*.log") -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-Item $_.FullName -Force; Write-Output "  Deleted $($_.Name)"
}

# Pycache
$cache = Join-Path $proj "src\__pycache__"
if (Test-Path $cache) { Remove-Item $cache -Recurse -Force; Write-Output "  Deleted pycache" }

Write-Output "Cleanup complete"
PSEOF

scp -o ConnectTimeout=10 /tmp/cleanup.ps1 "${DESKTOP}:C:/Users/ydran/cleanup.ps1"
ssh -o ConnectTimeout=10 "$DESKTOP" "powershell -ExecutionPolicy Bypass -File C:\Users\ydran\cleanup.ps1"

echo "Done."
