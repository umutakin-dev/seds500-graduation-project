# Launch run_all_final.py as a truly detached process using WMI.
# This survives SSH disconnection.

$proj = "C:\Users\ydran\workspace\seds\seds500-graduation-project"
$ps1 = "$proj\run_final.ps1"

# Clean stale lock file
Remove-Item "$proj\experiments\phase2\.running" -Force -ErrorAction SilentlyContinue

# Launch via WMI (creates process outside current session)
$proc = ([wmiclass]"Win32_Process").Create("powershell.exe -ExecutionPolicy Bypass -File $ps1")
Write-Output "Launched PID: $($proc.ProcessId), Return: $($proc.ReturnValue)"
