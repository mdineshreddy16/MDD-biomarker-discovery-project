# Download DAIC-WOZ Training Sessions Only
# This downloads sessions specified in train_split_Depression_AVEC2017.csv

$baseUrl = "http://dcapswoz.ict.usc.edu/wwwdaicwoz/"
$dataDir = "M:\5th sem\ML2-project\data"
$trainSplitPath = "$dataDir\splits\train_split_Depression_AVEC2017.csv"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  DAIC-WOZ Training Set Downloader" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Check if train split exists
if (-not (Test-Path $trainSplitPath)) {
    Write-Host "`n✗ Error: train_split_Depression_AVEC2017.csv not found!" -ForegroundColor Red
    Write-Host "  Please run download_daicwoz.ps1 first to get CSV files." -ForegroundColor Yellow
    exit
}

# Read training session IDs
Write-Host "`nReading training session IDs..." -ForegroundColor Yellow
$trainData = Import-Csv $trainSplitPath
$sessionIds = $trainData.Participant_ID

Write-Host "  Found $($sessionIds.Count) training sessions" -ForegroundColor Green

# Ask for confirmation
Write-Host "`n⚠️  WARNING: This will download ~50-55 GB of data!" -ForegroundColor Yellow
Write-Host "  Number of sessions: $($sessionIds.Count)" -ForegroundColor White
Write-Host "  Average size per session: ~450 MB" -ForegroundColor White
Write-Host "  Estimated time: 2-6 hours (depending on connection)" -ForegroundColor White

$confirm = Read-Host "`nContinue with download? (yes/no)"
if ($confirm -ne "yes") {
    Write-Host "Download cancelled." -ForegroundColor Yellow
    exit
}

# Download options
Write-Host "`nDownload Options:" -ForegroundColor Cyan
Write-Host "  1. Download ALL training sessions (~107 sessions)" -ForegroundColor White
Write-Host "  2. Download FIRST 10 sessions (for testing)" -ForegroundColor White
Write-Host "  3. Download SMALL sessions only (< 400 MB)" -ForegroundColor White
Write-Host "  4. Custom range" -ForegroundColor White

$option = Read-Host "`nSelect option (1-4)"

switch ($option) {
    "1" { 
        $sessionsToDownload = $sessionIds 
        Write-Host "  Downloading all $($sessionIds.Count) training sessions" -ForegroundColor Green
    }
    "2" { 
        $sessionsToDownload = $sessionIds | Select-Object -First 10
        Write-Host "  Downloading first 10 sessions" -ForegroundColor Green
    }
    "3" {
        # Small sessions (< 400MB based on directory listing)
        $smallSessions = @(300,309,318,319,324,326,327,338,340,347,348,351,352,354,355,357,358,360,361,362,375,385,387,388,389,391,392,393)
        $sessionsToDownload = $sessionIds | Where-Object { $smallSessions -contains $_ }
        Write-Host "  Downloading $($sessionsToDownload.Count) small sessions" -ForegroundColor Green
    }
    "4" {
        $start = Read-Host "  Start index (0-based)"
        $count = Read-Host "  Number of sessions"
        $sessionsToDownload = $sessionIds | Select-Object -Skip $start -First $count
        Write-Host "  Downloading $count sessions starting from index $start" -ForegroundColor Green
    }
    default {
        Write-Host "Invalid option. Exiting." -ForegroundColor Red
        exit
    }
}

# Create raw data directory
New-Item -ItemType Directory -Path "$dataDir\raw" -Force | Out-Null

# Download and extract sessions
$successCount = 0
$failCount = 0
$currentSession = 0

foreach ($sessionId in $sessionsToDownload) {
    $currentSession++
    $filename = "${sessionId}_P.zip"
    $url = $baseUrl + $filename
    $output = "$dataDir\raw\$filename"
    $extractPath = "$dataDir\raw\${sessionId}_P"
    
    Write-Host "`n[$currentSession/$($sessionsToDownload.Count)] Processing session $sessionId..." -ForegroundColor Cyan
    
    # Skip if already exists
    if (Test-Path $extractPath) {
        Write-Host "  ⏩ Already exists, skipping" -ForegroundColor Gray
        $successCount++
        continue
    }
    
    try {
        # Download
        Write-Host "  ⬇️  Downloading..." -ForegroundColor White
        Invoke-WebRequest -Uri $url -OutFile $output -ErrorAction Stop -TimeoutSec 600
        
        $fileSize = (Get-Item $output).Length / 1MB
        Write-Host "  ✓ Downloaded ($([math]::Round($fileSize, 1)) MB)" -ForegroundColor Green
        
        # Extract
        Write-Host "  📦 Extracting..." -ForegroundColor White
        Expand-Archive -Path $output -DestinationPath $extractPath -Force
        Write-Host "  ✓ Extracted" -ForegroundColor Green
        
        # Remove zip to save space (optional)
        Remove-Item $output -Force
        Write-Host "  🗑️  Removed zip file" -ForegroundColor Gray
        
        $successCount++
        
    } catch {
        Write-Host "  ✗ Failed: $_" -ForegroundColor Red
        $failCount++
        
        # Clean up partial downloads
        if (Test-Path $output) {
            Remove-Item $output -Force
        }
    }
    
    # Progress summary every 10 sessions
    if ($currentSession % 10 -eq 0) {
        Write-Host "`n  📊 Progress: $successCount successful, $failCount failed" -ForegroundColor Yellow
    }
}

# Final summary
Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host "  Download Complete!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Successful: $successCount" -ForegroundColor Green
Write-Host "  Failed: $failCount" -ForegroundColor Red
Write-Host "  Total sessions: $($sessionsToDownload.Count)" -ForegroundColor White
Write-Host "`n  Data location: $dataDir\raw\" -ForegroundColor Yellow

# Check total size
$totalSize = (Get-ChildItem "$dataDir\raw" -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host "  Total size: $([math]::Round($totalSize, 2)) GB" -ForegroundColor Yellow

Write-Host "`n📊 Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Verify data in: $dataDir\raw\" -ForegroundColor White
Write-Host "  2. Run DAIC-WOZ analysis notebook" -ForegroundColor White
Write-Host "  3. Extract features from downloaded sessions" -ForegroundColor White

Write-Host "`n==================================================" -ForegroundColor Cyan
