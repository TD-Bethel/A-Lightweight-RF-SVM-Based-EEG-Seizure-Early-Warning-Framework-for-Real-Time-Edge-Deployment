$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$generator = Join-Path $projectRoot "scripts\generate_readme.py"

if (-not (Test-Path $generator)) {
    throw "generate_readme.py not found in $projectRoot"
}

Write-Host "Watching project files for README updates..."
Write-Host "Project root: $projectRoot"
Write-Host "Press Ctrl+C to stop."

$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $projectRoot
$watcher.IncludeSubdirectories = $true
$watcher.EnableRaisingEvents = $true
$watcher.NotifyFilter = [System.IO.NotifyFilters]'FileName, LastWrite, CreationTime, Size'
$watcher.Filter = "*.*"

$script:lastRun = [datetime]::MinValue
$debounceMs = 1200

$action = {
    $changedPath = $Event.SourceEventArgs.FullPath
    $name = [System.IO.Path]::GetFileName($changedPath)
    $extension = [System.IO.Path]::GetExtension($changedPath).ToLowerInvariant()

    if ($changedPath -match "\\__pycache__\\") { return }
    if ($name -eq "README.md") { return }
    if ($name -eq "neurowatch_status.json") { return }
    if ($name -eq "neurowatch_patients.json") { return }
    if ($name -eq "neurowatch_metrics.json") { return }
    if ($name -eq "neurowatch_metrics_output.json") { return }
    if ($name -like "neurowatch_history_*.json") { return }

    $allowed = @(".py", ".ps1", ".json", ".md")
    if ($allowed -notcontains $extension) { return }

    $now = Get-Date
    if (($now - $script:lastRun).TotalMilliseconds -lt $debounceMs) { return }
    $script:lastRun = $now

    Write-Host ""
    Write-Host "Change detected:" $changedPath
    & python $generator
}

$subscriptions = @(
    Register-ObjectEvent -InputObject $watcher -EventName Changed -Action $action
    Register-ObjectEvent -InputObject $watcher -EventName Created -Action $action
    Register-ObjectEvent -InputObject $watcher -EventName Renamed -Action $action
    Register-ObjectEvent -InputObject $watcher -EventName Deleted -Action $action
)

try {
    & python $generator
    while ($true) {
        Wait-Event -Timeout 1 | Out-Null
    }
}
finally {
    foreach ($subscription in $subscriptions) {
        Unregister-Event -SourceIdentifier $subscription.Name -ErrorAction SilentlyContinue
        Remove-Job -Id $subscription.Id -Force -ErrorAction SilentlyContinue
    }
    $watcher.Dispose()
}
