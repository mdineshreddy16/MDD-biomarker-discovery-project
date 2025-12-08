# DAIC-WOZ Dataset Download Script
# This script helps you download the DAIC-WOZ Depression Database files

$baseUrl = "http://dcapswoz.ict.usc.edu/wwwdaicwoz/"
$dataDir = "M:\5th sem\ML2-project\data"

# Create directories
Write-Host "Creating directory structure..." -ForegroundColor Green
New-Item -ItemType Directory -Path "$dataDir\splits" -Force | Out-Null
New-Item -ItemType Directory -Path "$dataDir\raw" -Force | Out-Null
New-Item -ItemType Directory -Path "$dataDir\documentation" -Force | Out-Null

Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host "  DAIC-WOZ Depression Database Downloader" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Download CSV splits (CRITICAL - Small files)
Write-Host "`n[Step 1] Downloading CSV Split Files..." -ForegroundColor Yellow

$csvFiles = @(
    "train_split_Depression_AVEC2017.csv",
    "dev_split_Depression_AVEC2017.csv",
    "test_split_Depression_AVEC2017.csv",
    "full_test_split.csv"
)

foreach ($file in $csvFiles) {
    $url = $baseUrl + $file
    $output = "$dataDir\splits\$file"
    
    Write-Host "  Downloading $file..." -ForegroundColor White
    try {
        Invoke-WebRequest -Uri $url -OutFile $output -ErrorAction Stop
        Write-Host "  ✓ $file downloaded" -ForegroundColor Green
    } catch {
        Write-Host "  ✗ Failed to download $file" -ForegroundColor Red
        Write-Host "    Error: $_" -ForegroundColor Red
    }
}

# Download documentation
Write-Host "`n[Step 2] Downloading Documentation..." -ForegroundColor Yellow

$docFiles = @(
    "DAICWOZDepression_Documentation_AVEC2017.pdf",
    "documents.zip",
    "util.zip"
)

foreach ($file in $docFiles) {
    $url = $baseUrl + $file
    $output = "$dataDir\documentation\$file"
    
    Write-Host "  Downloading $file..." -ForegroundColor White
    try {
        Invoke-WebRequest -Uri $url -OutFile $output -ErrorAction Stop
        Write-Host "  ✓ $file downloaded" -ForegroundColor Green
    } catch {
        Write-Host "  ✗ Failed to download $file" -ForegroundColor Red
    }
}

# Extract documentation zips
Write-Host "`n[Step 3] Extracting documentation..." -ForegroundColor Yellow
try {
    Expand-Archive -Path "$dataDir\documentation\documents.zip" -DestinationPath "$dataDir\documentation\documents" -Force
    Expand-Archive -Path "$dataDir\documentation\util.zip" -DestinationPath "$dataDir\documentation\util" -Force
    Write-Host "  ✓ Documentation extracted" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Extraction failed: $_" -ForegroundColor Red
}

# Download sample sessions (optional)
Write-Host "`n[Step 4] Download Session Data (Optional)" -ForegroundColor Yellow
Write-Host "  Session files are large (200-900 MB each)" -ForegroundColor White
Write-Host "  Total dataset: ~85-90 GB for all 189 sessions" -ForegroundColor White
Write-Host ""

$downloadSessions = Read-Host "  Download sample sessions for testing? (y/n)"

if ($downloadSessions -eq 'y') {
    Write-Host "`n  Downloading 5 small sample sessions for testing..." -ForegroundColor Yellow
    
    # Small sample sessions
    $sampleSessions = @(
        @{id=300; size="327M"},
        @{id=309; size="346M"},
        @{id=318; size="287M"},
        @{id=319; size="310M"},
        @{id=357; size="187M"}
    )
    
    foreach ($session in $sampleSessions) {
        $sessionId = $session.id
        $filename = "${sessionId}_P.zip"
        $url = $baseUrl + $filename
        $output = "$dataDir\raw\$filename"
        
        Write-Host "`n  Downloading session $sessionId ($($session.size))..." -ForegroundColor White
        Write-Host "  This may take several minutes..." -ForegroundColor Gray
        
        try {
            Invoke-WebRequest -Uri $url -OutFile $output -ErrorAction Stop
            Write-Host "  ✓ Session $sessionId downloaded" -ForegroundColor Green
            
            # Extract
            Write-Host "  Extracting session $sessionId..." -ForegroundColor Gray
            Expand-Archive -Path $output -DestinationPath "$dataDir\raw\${sessionId}_P" -Force
            Write-Host "  ✓ Session $sessionId extracted" -ForegroundColor Green
            
        } catch {
            Write-Host "  ✗ Failed to download session $sessionId" -ForegroundColor Red
        }
    }
}

# Summary
Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host "  Download Summary" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

Write-Host "`n✓ CSV splits saved to: $dataDir\splits\" -ForegroundColor Green
Write-Host "✓ Documentation saved to: $dataDir\documentation\" -ForegroundColor Green

if ($downloadSessions -eq 'y') {
    Write-Host "✓ Sample sessions saved to: $dataDir\raw\" -ForegroundColor Green
}

Write-Host "`n📊 Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Check CSV files in data/splits/" -ForegroundColor White
Write-Host "  2. Review documentation PDF" -ForegroundColor White
Write-Host "  3. Run: jupyter notebook notebooks/03_DAIC_WOZ_analysis.ipynb" -ForegroundColor White

Write-Host "`n💡 To download more sessions:" -ForegroundColor Yellow
Write-Host "  - Check train_split.csv for session IDs" -ForegroundColor White
Write-Host "  - Download from: $baseUrl" -ForegroundColor White
Write-Host "  - Each session: http://dcapswoz.ict.usc.edu/wwwdaicwoz/XXX_P.zip" -ForegroundColor White

Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host "  Download complete!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
