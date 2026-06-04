param(
    [int]$Port = 8080,
    [int]$TopK = 3,
    [switch]$SkipRefresh
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Cli = Join-Path $Root "lotto_scraper.py"
$Csv = Join-Path $Root "xsmb_full.csv"
$Json = Join-Path $Root "xsmb_full.json"
$DashboardDir = Join-Path $Root "dashboard"
$Payload = Join-Path $DashboardDir "dashboard-payload.json"

if (-not (Test-Path $Python)) {
    throw "Python venv not found: $Python"
}

if (-not $SkipRefresh) {
    Write-Host "Refreshing data and dashboard payload..."
    & $Python $Cli refresh-dashboard --csv $Csv --json $Json --dashboard-output $Payload --top-k $TopK --pretty
}

$ExistingServer = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $ExistingServer) {
    Write-Host "Starting dashboard server on port $Port..."
    Start-Process -FilePath $Python -ArgumentList "-m http.server $Port" -WorkingDirectory $DashboardDir -WindowStyle Hidden
    Start-Sleep -Seconds 1
}

$Url = "http://localhost:$Port"
Write-Host ""
Write-Host "Dashboard ready: $Url"
Write-Host "Open the URL and click Load payload to view evaluations."
