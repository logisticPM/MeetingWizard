# Run the Meeting Minutes Application with AnythingLLM and LanceDB Integration
Write-Host "Starting Meeting Minutes Application with AnythingLLM and LanceDB Integration..." -ForegroundColor Green

# Check Python installation
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python 3") {
        Write-Host "Python detected: $pythonVersion" -ForegroundColor Green
    } else {
        Write-Host "Python 3 is required but a different version was found: $pythonVersion" -ForegroundColor Red
        exit
    }
} catch {
    Write-Host "Python not found. Please install Python 3.8 or higher." -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit
}

# Check for required packages
Write-Host "Checking for required packages..." -ForegroundColor Green
$requiredPackages = @("gradio", "openai", "lancedb", "pandas", "numpy")
$missingPackages = @()

foreach ($package in $requiredPackages) {
    try {
        $null = python -c "import $package"
        Write-Host "✓ $package is installed" -ForegroundColor Green
    } catch {
        Write-Host "✗ $package is not installed" -ForegroundColor Yellow
        $missingPackages += $package
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Host "Installing missing packages..." -ForegroundColor Yellow
    python -m pip install $missingPackages
}

# Check if AnythingLLM is running
$anythingLLMPort = 3001
$anythingLLMRunning = $false

try {
    $testConnection = Invoke-WebRequest -Uri "http://localhost:$anythingLLMPort/health" -UseBasicParsing -ErrorAction SilentlyContinue
    if ($testConnection.StatusCode -eq 200) {
        $anythingLLMRunning = $true
        Write-Host "AnythingLLM detected running on port $anythingLLMPort" -ForegroundColor Green
    }
} catch {
    Write-Host "AnythingLLM not detected on port $anythingLLMPort" -ForegroundColor Yellow
}

if (-not $anythingLLMRunning) {
    Write-Host "For full functionality, please start AnythingLLM before using the chat feature." -ForegroundColor Yellow
    Write-Host "You can download AnythingLLM from: https://github.com/Mintplex-Labs/anything-llm" -ForegroundColor Yellow
    Write-Host "The application will still run, but chat functionality will be limited." -ForegroundColor Yellow
    Write-Host ""
    $startAnyway = Read-Host "Do you want to continue without AnythingLLM? (Y/N)"
    if ($startAnyway -ne "Y" -and $startAnyway -ne "y") {
        Write-Host "Exiting. Please start AnythingLLM and try again." -ForegroundColor Red
        exit
    }
}

# Create data directories if they don't exist
if (-not (Test-Path "data")) {
    Write-Host "Creating data directory..." -ForegroundColor Green
    New-Item -ItemType Directory -Path "data" | Out-Null
}

# Display information about the application
Write-Host "" -ForegroundColor Cyan
Write-Host "Meeting Minutes Application with AnythingLLM and LanceDB Integration" -ForegroundColor Cyan
Write-Host "----------------------------------------------------------------" -ForegroundColor Cyan
Write-Host "This application provides the following features:" -ForegroundColor Cyan
Write-Host "" -ForegroundColor Cyan
Write-Host "1. Record or upload audio of a meeting" -ForegroundColor Cyan
Write-Host "2. Transcribe the audio using OpenAI's Whisper API" -ForegroundColor Cyan
Write-Host "3. Generate structured meeting minutes" -ForegroundColor Cyan
Write-Host "4. Store transcripts and meeting minutes with vector embeddings using LanceDB" -ForegroundColor Cyan
Write-Host "5. Chat with your meeting data using semantic search and AnythingLLM" -ForegroundColor Cyan
Write-Host "" -ForegroundColor Cyan
Write-Host "Prerequisites:" -ForegroundColor Cyan
Write-Host "- OpenAI API key (for transcription, summarization, and embeddings)" -ForegroundColor Cyan
Write-Host "- AnythingLLM running locally (for enhanced chat functionality)" -ForegroundColor Cyan
Write-Host "" -ForegroundColor Cyan
Write-Host "New Features:" -ForegroundColor Green
Write-Host "- Vector search using LanceDB for more accurate and contextual responses" -ForegroundColor Green
Write-Host "- Improved chat interface with semantic search capabilities" -ForegroundColor Green
Write-Host "- Enhanced context retrieval for better question answering" -ForegroundColor Green
Write-Host "" -ForegroundColor Cyan
Write-Host "The application will open in your default web browser." -ForegroundColor Cyan
Write-Host "" -ForegroundColor Cyan

# Run the application
Write-Host "Starting the application..." -ForegroundColor Green
try {
    Start-Process -NoNewWindow -FilePath "python" -ArgumentList "app_with_minutes.py"
    
    # Wait for the application to start
    Write-Host "Waiting for the application to start..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
    
    # Open the browser
    Start-Process "http://localhost:7680"
    
    Write-Host "Application started successfully!" -ForegroundColor Green
    Write-Host "If the browser didn't open automatically, navigate to: http://localhost:7680" -ForegroundColor Yellow
} catch {
    Write-Host "Error starting the application: $_" -ForegroundColor Red
}

# Keep the window open after the script completes
Read-Host "Press Enter to exit"
